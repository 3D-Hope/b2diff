from typing import Dict, List, Sequence

import numpy as np
import torch


def expand_furniture_slots(furniture_objects: Dict[str, int]) -> List[str]:
    """Expand {'chair': 2, 'table': 1} -> ['chair', 'chair', 'table']."""
    slots = []
    for label, count in furniture_objects.items():
        c = int(count)
        if c <= 0:
            continue
        slots.extend([label] * c)
    return slots


def labels_to_indices_in_order(
    ordered_labels: Sequence[str],
    object_type_labels: Sequence[str],
) -> List[int]:
    # print(f"object_type_labels: {object_type_labels}")
    lookup = {str(name): i for i, name in enumerate(object_type_labels)}
    out = []
    for label in ordered_labels:
        if label not in lookup:
            raise ValueError(f"Unknown class label '{label}'. Available labels: {list(object_type_labels)}")
        out.append(int(lookup[label]))
    return out


def apply_fixed_classes_to_scene_tensor(
    x: torch.Tensor,
    fixed_class_indices: Sequence[int],
    class_start: int,
    n_object_types: int,
    active_value: float = 1.0,
    inactive_value: float = -1.0,
) -> torch.Tensor:
    """Force first K object slots to fixed non-empty classes in tensor (B, N, C)."""
    if len(fixed_class_indices) == 0:
        return x

    class_end = class_start + n_object_types + 1
    if class_end > x.shape[-1]:
        raise ValueError("Invalid class slice for scene tensor")

    k = min(len(fixed_class_indices), x.shape[1])
    class_block = x[:, :, class_start:class_end]

    class_block[:, :k, :] = inactive_value
    empty_idx = n_object_types
    class_block[:, :k, empty_idx] = inactive_value

    for slot, cls_idx in enumerate(fixed_class_indices[:k]):
        class_block[:, slot, cls_idx] = active_value

    x[:, :, class_start:class_end] = class_block
    return x


def enforce_fixed_class_order_in_layout(
    layout: Dict[str, np.ndarray],
    fixed_class_indices: Sequence[int],
    n_object_types: int,
) -> Dict[str, np.ndarray]:
    """Overwrite first K rows of class_labels to fixed classes in exact order."""
    if len(fixed_class_indices) == 0:
        return layout

    class_labels = np.asarray(layout["class_labels"], dtype=np.float32)
    if class_labels.ndim != 2 or class_labels.shape[1] != n_object_types:
        raise ValueError(
            f"layout['class_labels'] expected shape (num_obj, {n_object_types}), got {class_labels.shape}"
        )

    k = min(len(fixed_class_indices), class_labels.shape[0])
    forced = class_labels.copy()
    for slot, cls_idx in enumerate(fixed_class_indices[:k]):
        forced[slot, :] = 0.0
        forced[slot, cls_idx] = 1.0

    layout["class_labels"] = forced
    return layout


def furniture_filter_hard_exact(
    layout: Dict[str, np.ndarray],
    required_class_counts: Dict[int, int],
) -> bool:
    """True only if per-class counts exactly match required counts."""
    cls = np.asarray(layout["class_labels"], dtype=np.float32)
    pred = np.argmax(cls, axis=-1)

    for class_idx, target_count in required_class_counts.items():
        found = int(np.sum(pred == int(class_idx)))
        if found != int(target_count):
            return False
    return True


def furniture_filter_soft_weighted(
    layout: Dict[str, np.ndarray],
    required_class_counts: Dict[int, int],
    class_weights: Dict[int, float],
    max_distance: float,
) -> bool:
    """Weighted L1 distance on requested class counts."""
    cls = np.asarray(layout["class_labels"], dtype=np.float32)
    pred = np.argmax(cls, axis=-1)

    distance = 0.0
    for class_idx, target_count in required_class_counts.items():
        found = int(np.sum(pred == int(class_idx)))
        w = float(class_weights.get(int(class_idx), 1.0))
        distance += w * abs(found - int(target_count))

    return distance <= float(max_distance)
