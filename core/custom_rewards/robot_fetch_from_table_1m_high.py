import torch


# Bedroom object ids considered as table-like surfaces.
_TABLE_OBJECT_INDICES = {6, 7, 10, 18}  # coffee_table, desk, dressing_table, table


def compute_robot_fetch_from_table_1m_high_reward(
    parsed_scene,
    max_fetch_height=1.0,
    epsilon=0.05,
    no_table_penalty=-2.0,
    success_reward=2.0,
    **kwargs,
):
    """
    Reward scenes where robot-relevant table surfaces are reachable (<= 1m + epsilon).

    Per-scene logic:
    - If no table-like objects exist: assign a big penalty.
    - If all table tops are <= (max_fetch_height + epsilon): assign +success_reward.
    - If any table top is above (max_fetch_height + epsilon):
            reward is negative sum of excess heights above threshold.

    Height of each table top is computed as:
      top_height = position_y + size_y
    where size_y is half-extent from parsed/descaled scene representation.
    """
    positions = parsed_scene["positions"]
    sizes = parsed_scene["sizes"]
    object_indices = parsed_scene["object_indices"]
    is_empty = parsed_scene["is_empty"]

    device = positions.device
    batch_size = positions.shape[0]

    rewards = torch.zeros(batch_size, dtype=positions.dtype, device=device)
    table_ids = torch.tensor(sorted(_TABLE_OBJECT_INDICES), device=device, dtype=object_indices.dtype)
    max_allowed = max_fetch_height + epsilon

    for b in range(batch_size):
        valid_mask = ~is_empty[b]
        if valid_mask.sum() == 0:
            rewards[b] = no_table_penalty
            continue

        valid_indices = object_indices[b][valid_mask]
        valid_positions = positions[b][valid_mask]
        valid_sizes = sizes[b][valid_mask]

        table_mask = (valid_indices[..., None] == table_ids[None, ...]).any(dim=-1)
        if not table_mask.any():
            rewards[b] = no_table_penalty
            continue

        table_top_heights = valid_positions[table_mask, 1] + valid_sizes[table_mask, 1]
        over_mask = table_top_heights > max_allowed

        if over_mask.any():
            excess_heights = table_top_heights[over_mask] - max_allowed
            rewards[b] = -excess_heights.sum()
        else:
            rewards[b] = success_reward

    return rewards


def compute_reward(parsed_scene, **kwargs):
    """Generic custom-reward entrypoint used by dynamic loader."""
    with torch.no_grad():
        return compute_robot_fetch_from_table_1m_high_reward(parsed_scene, **kwargs)
