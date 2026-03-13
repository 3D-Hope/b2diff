import torch


def compute_aabb_penetration_depth(centers1, sizes1, centers2, sizes2):
    """
    Compute penetration depth between two sets of AABBs.

    Penetration depth is defined as the minimum translation distance needed to
    separate two overlapping objects. For AABBs, this is the minimum overlap
    across all axes.

    Args:
        centers1: (B, N1, 3) - centers of first set of boxes
        sizes1: (B, N1, 3) - sizes of first set of boxes
        centers2: (B, N2, 3) - centers of second set of boxes
        sizes2: (B, N2, 3) - sizes of second set of boxes

    Returns:
        penetration_depth: (B, N1, N2) - penetration depth (0 if no overlap)
    """
    batch_size = centers1.shape[0]
    device = centers1.device

    # Expand dimensions for pairwise comparison
    c1 = centers1.unsqueeze(2)  # (B, N1, 1, 3)
    s1 = sizes1.unsqueeze(2)  # (B, N1, 1, 3)
    c2 = centers2.unsqueeze(1)  # (B, 1, N2, 3)
    s2 = sizes2.unsqueeze(1)  # (B, 1, N2, 3)

    # NOTE: sizes are already half-extents (sx/2, sy/2, sz/2)
    # So we use them directly
    half1 = s1  # (B, N1, 1, 3) - already half-extents
    half2 = s2  # (B, 1, N2, 3) - already half-extents

    # Compute center distance for each axis
    center_dist = torch.abs(c1 - c2)  # (B, N1, N2, 3)

    # Compute sum of half-extents for each axis
    sum_half_extents = half1 + half2  # (B, N1, N2, 3)

    # Overlap on each axis = sum_of_half_extents - center_distance
    # If positive, there's overlap; if negative or zero, no overlap
    overlap_per_axis = sum_half_extents - center_dist  # (B, N1, N2, 3)

    # For AABBs to overlap, they must overlap on ALL axes
    # Check if overlapping on all axes
    is_overlapping_all_axes = (overlap_per_axis > 0).all(dim=3)  # (B, N1, N2)

    # Penetration depth is the MINIMUM overlap across all axes
    # (the smallest amount needed to separate the objects)
    min_overlap_per_pair = overlap_per_axis.min(dim=3)[0]  # (B, N1, N2)

    # Only consider penetration where objects actually overlap on all axes
    penetration_depth = torch.where(
        is_overlapping_all_axes,
        min_overlap_per_pair,
        torch.zeros_like(min_overlap_per_pair),
    )

    # Clamp to ensure non-negative
    penetration_depth = torch.clamp(penetration_depth, min=0.0)

    return penetration_depth


def compute_non_penetration_reward(parsed_scenes, **kwargs):
    """
    Calculate reward based on non-penetration constraint using penetration depth.

    Following the approach from original authors: reward = sum of negative signed distances.
    When objects overlap, we get positive penetration depth, so reward is negative.

    Args:
        parsed_scenes: Dict returned by parse_and_descale_scenes()

    Returns:
        rewards: Tensor of shape (B,) with non-penetration rewards for each scene
    """
    room_type = kwargs["room_type"]
    positions = parsed_scenes["positions"]
    sizes = parsed_scenes["sizes"]
    object_indices = parsed_scenes["object_indices"]
    is_empty = parsed_scenes["is_empty"]
    batch_size = positions.shape[0]
    device = positions.device

    # Identify ceiling objects (they don't participate in ground-level collisions)
    ceiling_objects = ["ceiling_lamp", "pendant_lamp"]
    idx_to_labels_bedroom = {
        0: "armchair",
        1: "bookshelf",
        2: "cabinet",
        3: "ceiling_lamp",
        4: "chair",
        5: "children_cabinet",
        6: "coffee_table",
        7: "desk",
        8: "double_bed",
        9: "dressing_chair",
        10: "dressing_table",
        11: "kids_bed",
        12: "nightstand",
        13: "pendant_lamp",
        14: "shelf",
        15: "single_bed",
        16: "sofa",
        17: "stool",
        18: "table",
        19: "tv_stand",
        20: "wardrobe",
    }
    ceiling_indices = [
        idx for idx, label in idx_to_labels_bedroom.items() if label in ceiling_objects
    ]
    is_ceiling = torch.zeros_like(is_empty, dtype=torch.bool)
    for ceiling_idx in ceiling_indices:
        is_ceiling |= object_indices == ceiling_idx

    # Create mask for ground objects (non-empty, non-ceiling)
    is_ground_object = ~is_empty & ~is_ceiling

    # Compute pairwise penetration depths
    penetration_depth = compute_aabb_penetration_depth(
        positions, sizes, positions, sizes
    )  # (B, N, N)

    # Create mask to ignore self-overlaps (diagonal), empty objects, and ceiling objects
    mask = is_ground_object.unsqueeze(2) & is_ground_object.unsqueeze(1)  # (B, N, N)

    # Remove self-overlaps (diagonal)
    eye = torch.eye(positions.shape[1], device=device, dtype=torch.bool)
    eye = eye.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, N)
    mask = mask & ~eye

    # Apply mask to penetration depths
    masked_penetration = torch.where(
        mask, penetration_depth, torch.zeros_like(penetration_depth)
    )

    # Sum total penetration depth per scene
    # Divide by 2 because each pair is counted twice (i,j) and (j,i)
    total_penetration = masked_penetration.sum(dim=[1, 2]) / 2.0  # (B,)

    # Convert to reward: +1 if no penetration, else -penetration_depth
    # This provides a clear positive signal for valid scenes and negative for invalid ones
    rewards = torch.where(
        total_penetration == 0, torch.ones_like(total_penetration), -total_penetration
    )
    return rewards
