"""
Object Count Reward - Encourages realistic number of objects in a scene.

Uses KL divergence between the batch's object count distribution and the 
empirical distribution from training data. This is superior to per-scene NLL
because it:
1. Prevents mode collapse (can't spam the most common count)
2. Encourages distributional diversity within batches
3. Matches the training data distribution at the batch level

### **Object Count Distribution for Training Set (3,722 scenes):**

- **Mean (Average) Object Count: 4.99**
- **Median:** 5 objects
- **Standard Deviation:** 1.43
- **Range:** 3-12 objects per scene

### **Distribution Breakdown:**

- 3 objects: 14.05% (523 scenes)
- 4 objects: 25.36% (944 scenes)
- 5 objects: 30.52% (1,136 scenes) Most common
- 6 objects: 15.80% (588 scenes)
- 7 objects: 9.30% (346 scenes)
- 8-12 objects: <5% combined

Reward = -KL(batch_dist || training_dist)
Perfect match: KL = 0 -> reward = 0
Mode collapse: high KL -> large negative reward
"""

import torch
import torch.nn.functional as F


def compute_object_count_reward(
    parsed_scene, mode="kl", target_count=5.21, std_dev=1, **kwargs
):
    """
    Calculate reward based on object count distribution.

    Three modes:
    1. KL mode (default, recommended): Uses KL divergence between batch distribution
       and empirical training distribution. Encourages diversity and prevents mode collapse.
    2. NLL mode: Uses negative log-likelihood per scene (legacy)
    3. Gaussian mode: Uses squared deviation from mean (legacy)

    Args:
        parsed_scene: Dict returned by parse_and_descale_scenes() with batch dimension
        mode: 'kl' (default), 'nll', or 'gaussian'
        target_count: Expected number of objects (used only for gaussian mode)
        std_dev: Standard deviation (used only for gaussian mode)
        **kwargs: Additional arguments (ignored, for compatibility)

    Returns:
        rewards: Tensor of shape (B,) with object count rewards for each scene
    """
    is_empty = parsed_scene["is_empty"]
    device = is_empty.device
    batch_size = is_empty.shape[0]

    # Count non-empty objects per scene
    object_counts = (~is_empty).sum(dim=1).long()  # (B,) as integers

    # Training distribution (empirical from 3,722 scenes)
    # Counts: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
    training_probs = torch.tensor(
        [
            0.0001,  # 0 objects (impossible, but add smoothing)
            0.0001,  # 1 object (very rare, smoothing)
            0.0001,  # 2 objects (very rare, smoothing)
            0.1405,  # 3 objects: 14.05%
            0.2536,  # 4 objects: 25.36%
            0.3052,  # 5 objects: 30.52% (most common)
            0.1580,  # 6 objects: 15.80%
            0.0930,  # 7 objects: 9.30%
            0.0272,  # 8 objects: ~2.72%
            0.0124,  # 9 objects: ~1.24%
            0.0062,  # 10 objects: ~0.62%
            0.0031,  # 11 objects: ~0.31%
            0.0006,  # 12 objects: ~0.06%
        ],
        device=device,
    )

    # Normalize to ensure sum = 1 (after smoothing)
    training_probs = training_probs / training_probs.sum()

    if mode == "kl":
        # KL divergence mode: Compare batch distribution to training distribution

        # Create histogram of object counts in this batch
        # Count range: [0, 12] TODO: hardcoded for 12 objects scene
        batch_histogram = torch.zeros(13, device=device)

        # Clamp counts to valid range [0, 12]
        clamped_counts = object_counts.clamp(0, 12)

        # Count occurrences of each count value
        for count in range(13):
            batch_histogram[count] = (clamped_counts == count).sum().float()

        # Normalize to get batch probability distribution
        batch_probs = batch_histogram / batch_size

        # Add small epsilon for numerical stability (avoid log(0))
        eps = 1e-10
        batch_probs = batch_probs + eps
        training_probs = training_probs + eps

        # Renormalize after adding epsilon
        batch_probs = batch_probs / batch_probs.sum()
        training_probs = training_probs / training_probs.sum()

        # Compute KL divergence: KL(batch || training)
        # KL(P || Q) = sum(P * log(P / Q))
        kl_div = (
            batch_probs * (torch.log(batch_probs) - torch.log(training_probs))
        ).sum()

        # Return negative KL as reward (lower KL = better = higher reward)
        # All scenes in batch get the same reward (batch-level metric)
        reward_value = -kl_div
        rewards = reward_value.expand(batch_size)

    elif mode == "nll":
        # NLL mode: Per-scene negative log-likelihood
        log_probs = torch.log(training_probs + 1e-10)
        clamped_counts = object_counts.clamp(0, 12)
        rewards = log_probs[clamped_counts]

    else:  # gaussian mode
        # Legacy Gaussian penalty mode
        object_counts_float = object_counts.float()
        deviation = torch.abs(object_counts_float - target_count)
        normalized_deviation = deviation / std_dev
        rewards = -(normalized_deviation**2)

    return rewards


def test_object_count_reward():
    """Test cases for object count reward - demonstrates all three modes."""
    print("\n" + "=" * 70)
    print("Testing Object Count Reward (All Modes)")
    print("=" * 70)

    device = "cpu"
    num_classes = 22
    num_objects = 12

    # Helper to create a scene with specific number of objects
    def create_scene_with_count(count):
        """Create a scene with exactly 'count' non-empty objects."""
        scene = torch.zeros(num_objects, 30, device=device)

        # Fill first 'count' slots with objects (use chair class=4)
        for i in range(count):
            scene[i, 4] = 1.0  # Chair one-hot
            scene[i, num_classes : num_classes + 3] = torch.tensor(
                [0.0, 0.0, 0.0], device=device
            )
            scene[i, num_classes + 3 : num_classes + 6] = torch.tensor(
                [0.0, 0.0, 0.0], device=device
            )
            scene[i, num_classes + 6 : num_classes + 8] = torch.tensor(
                [1.0, 0.0], device=device
            )

        # Fill remaining slots with empty class (index 21)
        for i in range(count, num_objects):
            scene[i, 21] = 1.0  # Empty class
            scene[i, num_classes:] = 0.0

        return scene

    # Import parse function
    from universal_constraint_rewards.commons import parse_and_descale_scenes

    # ========================================================================
    # Test 1: Diverse batch (matches training distribution well)
    # ========================================================================
    print("\n" + "-" * 70)
    print("Test 1: Diverse Batch (good distribution match)")
    print("-" * 70)
    print("Batch counts: [3, 4, 5, 5, 6] - diverse, matches training pattern")

    batch1 = torch.stack(
        [
            create_scene_with_count(3),  # 14.05% in training
            create_scene_with_count(4),  # 25.36% in training
            create_scene_with_count(5),  # 30.52% in training (most common)
            create_scene_with_count(5),  # 30.52% in training
            create_scene_with_count(6),  # 15.80% in training
        ],
        dim=0,
    )

    parsed1 = parse_and_descale_scenes(batch1, num_classes=num_classes)

    # Compute with all modes
    rewards_kl1 = compute_object_count_reward(parsed1, mode="kl")
    rewards_nll1 = compute_object_count_reward(parsed1, mode="nll")
    rewards_gaussian1 = compute_object_count_reward(parsed1, mode="gaussian")

    print(f"\nKL Mode (batch-level): {rewards_kl1[0].item():.4f} (same for all scenes)")
    print("  -> Batch dist: 3:20%, 4:20%, 5:40%, 6:20%")
    print("  -> Train dist: 3:14%, 4:25%, 5:31%, 6:16%")
    print("  -> Low KL divergence = good match!")

    print("\nNLL Mode (per-scene):")
    for i in range(5):
        count = (~parsed1["is_empty"][i]).sum().item()
        print(f"  Scene {i+1} ({count} objects): {rewards_nll1[i].item():.4f}")

    print("\nGaussian Mode (per-scene):")
    for i in range(5):
        count = (~parsed1["is_empty"][i]).sum().item()
        print(f"  Scene {i+1} ({count} objects): {rewards_gaussian1[i].item():.4f}")

    # ========================================================================
    # Test 2: Mode collapse batch (all same count)
    # ========================================================================
    print("\n" + "-" * 70)
    print("Test 2: Mode Collapse Batch (all 5 objects)")
    print("-" * 70)
    print("Batch counts: [5, 5, 5, 5, 5] - no diversity!")

    batch2 = torch.stack(
        [
            create_scene_with_count(5),
            create_scene_with_count(5),
            create_scene_with_count(5),
            create_scene_with_count(5),
            create_scene_with_count(5),
        ],
        dim=0,
    )

    parsed2 = parse_and_descale_scenes(batch2, num_classes=num_classes)

    rewards_kl2 = compute_object_count_reward(parsed2, mode="kl")
    rewards_nll2 = compute_object_count_reward(parsed2, mode="nll")

    print(f"\nKL Mode (batch-level): {rewards_kl2[0].item():.4f}")
    print("  -> Batch dist: 5:100% (everything else:0%)")
    print("  -> Train dist: 5:31% (has 3,4,6,7 too)")
    print("  -> HIGH KL divergence = mode collapse penalty!")

    print(f"\nNLL Mode (per-scene): {rewards_nll2[0].item():.4f}")
    print("  -> All scenes get good reward (5 is common)")
    print("  -> NLL CANNOT detect mode collapse!")

    # ========================================================================
    # Test 3: Rare count batch
    # ========================================================================
    print("\n" + "-" * 70)
    print("Test 3: Rare Count Batch")
    print("-" * 70)
    print("Batch counts: [8, 9, 10, 11, 12] - all rare!")

    batch3 = torch.stack(
        [
            create_scene_with_count(8),  # 2.72%
            create_scene_with_count(9),  # 1.24%
            create_scene_with_count(10),  # 0.62%
            create_scene_with_count(11),  # 0.31%
            create_scene_with_count(12),  # 0.06%
        ],
        dim=0,
    )

    parsed3 = parse_and_descale_scenes(batch3, num_classes=num_classes)

    rewards_kl3 = compute_object_count_reward(parsed3, mode="kl")
    rewards_nll3 = compute_object_count_reward(parsed3, mode="nll")

    print(f"\nKL Mode (batch-level): {rewards_kl3[0].item():.4f}")
    print("  -> Batch focused on rare counts (8-12)")
    print("  -> VERY HIGH KL divergence!")

    print(f"\nNLL Mode (per-scene, average): {rewards_nll3.mean().item():.4f}")
    print("  -> Individual rare scenes penalized")

    # ========================================================================
    # Comparison and assertions
    # ========================================================================
    print("\n" + "=" * 70)
    print("KEY INSIGHT: Why KL Mode is Superior")
    print("=" * 70)

    print("\nDiverse batch (Test 1) vs Mode Collapse (Test 2):")
    print(f"  KL Mode:  {rewards_kl1[0].item():.4f} vs {rewards_kl2[0].item():.4f}")
    print("            -> better      -> worse (penalizes lack of diversity)")
    print(
        f"  NLL Mode: {rewards_nll1.mean().item():.4f} vs {rewards_nll2.mean().item():.4f}"
    )
    print("            -> similar rewards - cannot detect mode collapse!")

    # KL should prefer diverse batch over mode collapse
    assert (
        rewards_kl1[0] > rewards_kl2[0]
    ), "KL should reward diverse batch more than mode collapse!"

    # Both batches should be heavily penalized for rare counts
    assert rewards_kl3[0] < rewards_kl1[0], "Rare count batch should have lower reward"

    print("\nOK: All object count tests passed!")
    print("OK: KL divergence successfully prevents mode collapse!")
    print("=" * 70)


if __name__ == "__main__":
    test_object_count_reward()
