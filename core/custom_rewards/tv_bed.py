import torch
import torch.nn.functional as F


_IDX_TO_LABEL_BEDROOM = {
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


def compute_tv_bed_presence_reward(parsed_scene, ideal=3.0, sigma=1.0, **kwargs):
    """
    Gaussian-shaped reward for ideal bed–TV distance.
    """
    device = parsed_scene["device"]
    object_indices = parsed_scene["object_indices"]
    positions = parsed_scene["positions"]
    orientations = parsed_scene["orientations"]
    is_empty = parsed_scene["is_empty"]
    idx_to_labels = kwargs.get("idx_to_labels", _IDX_TO_LABEL_BEDROOM)

    # Handle both integer and string keys
    if idx_to_labels and isinstance(list(idx_to_labels.keys())[0], str):
        idx_to_labels = {int(k): v for k, v in idx_to_labels.items()}

    idx_tv = next((k for k, v in idx_to_labels.items() if "tv_stand" in v), None)
    idx_bed = next((k for k, v in idx_to_labels.items() if "bed" in v), None)
    # print(f"TV index: {idx_tv}, Bed index: {idx_bed}")
    if idx_tv is None or idx_bed is None:
        return torch.zeros(len(object_indices), device=device)

    rewards = torch.zeros(len(object_indices), device=device)
    for b in range(len(object_indices)):
        try:
            # Get valid mask - ensure boolean tensor
            if isinstance(is_empty, torch.Tensor):
                valid_mask = ~is_empty[b]
            else:
                valid_mask = ~torch.tensor(is_empty[b], dtype=torch.bool, device=device)

            # Convert to boolean explicitly
            if isinstance(valid_mask, torch.Tensor):
                if valid_mask.dtype != torch.bool:
                    valid_mask = valid_mask.bool()
            else:
                # valid_mask is a Python bool - no valid objects
                continue

            # Check if we have any valid objects
            if valid_mask.sum().item() == 0:
                continue

            valid_indices = object_indices[b][valid_mask]
            valid_pos = positions[b][valid_mask]
            valid_orient = orientations[b][valid_mask]

            if not isinstance(valid_indices, torch.Tensor):
                continue

            if valid_indices.dim() == 0:
                valid_indices = valid_indices.unsqueeze(0)
                valid_pos = valid_pos.unsqueeze(0)
                valid_orient = valid_orient.unsqueeze(0)

            if valid_indices.numel() == 0:
                continue

            # Check for TV and bed - ensure tensor comparisons
            tv_mask = (valid_indices == idx_tv)
            bed_mask = (valid_indices == idx_bed)

            if isinstance(tv_mask, torch.Tensor):
                has_tv = tv_mask.any().item()
            else:
                has_tv = bool(tv_mask)

            if isinstance(bed_mask, torch.Tensor):
                has_bed = bed_mask.any().item()
            else:
                has_bed = bool(bed_mask)

            if not has_bed:
                rewards[b] += -5

            if not has_tv:
                rewards[b] += -5

            if not (has_tv and has_bed):
                continue

            tv_pos = valid_pos[valid_indices == idx_tv][0]
            bed_pos = valid_pos[valid_indices == idx_bed][0]

            dist = torch.norm(tv_pos - bed_pos)
            rewards[b] += torch.exp(-((dist - ideal) ** 2) / (2 * sigma**2))

            # facing reward
            bed_dir = valid_orient[valid_indices == idx_bed][0]

            # Compute direction from bed to TV (in XZ plane, ignore Y)
            dir_bed_to_tv = tv_pos - bed_pos
            # Project to 2D (XZ plane) to match orientation which is [cos, sin] in XZ
            dir_bed_to_tv_2d = torch.tensor([dir_bed_to_tv[0], dir_bed_to_tv[2]], device=device)
            dir_bed_to_tv_2d = dir_bed_to_tv_2d / (torch.norm(dir_bed_to_tv_2d) + 1e-6)

            alignment = F.cosine_similarity(bed_dir.unsqueeze(0), dir_bed_to_tv_2d.unsqueeze(0)).clamp(0, 1)
            rewards[b] += alignment.item()

        except Exception as e:
            print(f"[ERROR] reward_tv_distance batch {b}: {e}")
            continue

    return rewards


def compute_reward(parsed_scene, **kwargs):
    """Generic custom-reward entrypoint used by dynamic loader."""
    return compute_tv_bed_presence_reward(parsed_scene, **kwargs)