import torch


# Bedroom class ids
_CHAIR_IDX = 4
_DESK_IDX = 7


def compute_desk_chair_for_study_reward(
    parsed_scene,
    both_present_reward=2.0,
    one_missing_penalty=-5.0,
    both_missing_penalty=-10.0,
    extra_item_penalty=5,
    **kwargs,
):
    """
    Reward per scene:
    - +2 if both desk and chair exist
    - -1 if exactly one is missing
    - -2 if both are missing
        - Additional penalty for overplacement: each extra desk/chair beyond 1
            subtracts extra_item_penalty from the scene reward.
    """
    object_indices = parsed_scene["object_indices"]
    is_empty = parsed_scene["is_empty"]
    device = parsed_scene["device"]

    batch_size = object_indices.shape[0]
    rewards = torch.zeros(batch_size, device=device, dtype=torch.float32)

    for b in range(batch_size):
        valid_mask = ~is_empty[b]
        valid_indices = object_indices[b][valid_mask]

        chair_count = int((valid_indices == _CHAIR_IDX).sum().item())
        desk_count = int((valid_indices == _DESK_IDX).sum().item())

        has_chair = chair_count > 0
        has_desk = desk_count > 0

        if has_chair and has_desk:
            rewards[b] = both_present_reward
        elif has_chair or has_desk:
            rewards[b] = one_missing_penalty
        else:
            rewards[b] = both_missing_penalty

        extra_chairs = max(chair_count - 1, 0)
        extra_desks = max(desk_count - 1, 0)
        overplacement_count = extra_chairs + extra_desks
        if overplacement_count > 0:
            rewards[b] -= float(extra_item_penalty) * float(overplacement_count)

    return rewards


def compute_reward(parsed_scene, **kwargs):
    """Generic custom-reward entrypoint used by dynamic loader."""
    with torch.no_grad():
        return compute_desk_chair_for_study_reward(parsed_scene, **kwargs)
