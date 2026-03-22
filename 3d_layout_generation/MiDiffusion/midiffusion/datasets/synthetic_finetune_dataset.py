import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import dataloader


def _scale_to_minus_one_one(x, minimum, maximum):
    """Inverse of descale_to_origin with clipping, matching Scale.scale behavior."""
    x = np.asarray(x, dtype=np.float32)
    minimum = np.asarray(minimum, dtype=np.float32)
    maximum = np.asarray(maximum, dtype=np.float32)
    x = np.clip(x, minimum, maximum)
    x = (x - minimum) / (maximum - minimum)
    x = 2.0 * x - 1.0
    return x


class SyntheticFineTuneDataset(Dataset):
    """Synthetic-only dataset adapter for MiDiffusion finetuning.

    It consumes ThreedFrontResults pickles and emits the same keys used by
    ashok_train.py training loop:
      - translations: (max_length, 3), scaled to [-1, 1]
      - sizes:        (max_length, 3), scaled to [-1, 1]
      - angles:       (max_length, 2), [cos(theta), sin(theta)]
      - class_labels: (max_length, n_object_types+1), where last channel is
                      empty and values are converted to [-1, 1]
      - fpbpn:        (256, 4) floor condition from encoded train+val ordering
      - length:       scalar number of non-empty objects before padding
    """

    def __init__(self, pickle_paths, reference_encoded_dataset, bounds, max_length):
        self._pickle_paths = [str(Path(p)) for p in pickle_paths]
        self._reference_encoded_dataset = reference_encoded_dataset
        self._bounds = bounds
        self._max_length = int(max_length)

        self._object_types = reference_encoded_dataset.object_types
        self._n_object_types = reference_encoded_dataset.n_object_types

        self._records = []
        self._load_records()

    def _load_records(self):
        for pkl_path in self._pickle_paths:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            if not hasattr(data, "_scene_indices") or not hasattr(data, "_predicted_layouts"):
                raise ValueError(
                    f"{pkl_path} does not look like a ThreedFrontResults object."
                )

            scene_indices = list(data._scene_indices)
            layouts = list(data._predicted_layouts)
            if len(scene_indices) != len(layouts):
                raise ValueError(
                    f"{pkl_path} has mismatched scene_indices/layout lengths: "
                    f"{len(scene_indices)} vs {len(layouts)}"
                )

            for scene_idx, layout in zip(scene_indices, layouts):
                scene_idx = int(scene_idx)
                if scene_idx < 0 or scene_idx >= len(self._reference_encoded_dataset):
                    raise IndexError(
                        f"Scene index {scene_idx} from {pkl_path} is out of range for "
                        f"reference encoded dataset of length {len(self._reference_encoded_dataset)}"
                    )
                self._records.append((scene_idx, layout))

    @property
    def bounds(self):
        return self._bounds

    @property
    def n_object_types(self):
        return self._n_object_types

    @property
    def object_types(self):
        return self._object_types

    @property
    def max_length(self):
        return self._max_length

    def __len__(self):
        return len(self._records)

    def __getitem__(self, idx):
        scene_idx, layout = self._records[idx]

        trans_world = np.asarray(layout["translations"], dtype=np.float32)
        sizes_world = np.asarray(layout["sizes"], dtype=np.float32)
        angles_rad = np.asarray(layout["angles"], dtype=np.float32)
        class_non_empty = np.asarray(layout["class_labels"], dtype=np.float32)

        if trans_world.ndim == 1:
            trans_world = trans_world.reshape(1, -1)
        if sizes_world.ndim == 1:
            sizes_world = sizes_world.reshape(1, -1)
        if angles_rad.ndim == 0:
            angles_rad = angles_rad.reshape(1, 1)
        elif angles_rad.ndim == 1:
            angles_rad = angles_rad.reshape(-1, 1)
        if class_non_empty.ndim == 1:
            class_non_empty = class_non_empty.reshape(1, -1)

        n_obj = min(
            trans_world.shape[0],
            sizes_world.shape[0],
            angles_rad.shape[0],
            class_non_empty.shape[0],
            self._max_length,
        )
        assert n_obj == trans_world.shape[0] == sizes_world.shape[0] == angles_rad.shape[0] == class_non_empty.shape[0]

        translations = np.zeros((self._max_length, 3), dtype=np.float32)
        sizes = np.zeros((self._max_length, 3), dtype=np.float32)
        angles = np.zeros((self._max_length, 2), dtype=np.float32)

        class_dim = self._n_object_types + 1
        class_labels = np.zeros((self._max_length, class_dim), dtype=np.float32)
        class_labels[:, -1] = 1.0  # pad with empty class

        if n_obj > 0:
            t_norm = _scale_to_minus_one_one(
                trans_world[:n_obj],
                self._bounds["translations"][0],
                self._bounds["translations"][1],
            )
            s_norm = _scale_to_minus_one_one(
                sizes_world[:n_obj],
                self._bounds["sizes"][0],
                self._bounds["sizes"][1],
            )

            theta = angles_rad[:n_obj, 0]
            angles_cs = np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)

            c_non_empty = class_non_empty[:n_obj]
            if c_non_empty.shape[1] != self._n_object_types:
                raise ValueError(
                    "Unexpected synthetic class dimension. "
                    f"Expected {self._n_object_types}, got {c_non_empty.shape[1]}"
                )
            c_full = np.concatenate(
                [c_non_empty, np.zeros((n_obj, 1), dtype=np.float32)], axis=-1
            )

            translations[:n_obj] = t_norm
            sizes[:n_obj] = s_norm
            angles[:n_obj] = angles_cs
            class_labels[:n_obj] = c_full

        # Real diffusion training uses class labels converted to [-1, 1].
        class_labels = class_labels * 2.0 - 1.0

        ref_sample = self._reference_encoded_dataset[scene_idx]
        fpbpn = np.asarray(ref_sample["fpbpn"], dtype=np.float32)

        return {
            "translations": translations,
            "sizes": sizes,
            "angles": angles,
            "class_labels": class_labels,
            "fpbpn": fpbpn,
            "length": np.int64(n_obj),
        }

    def collate_fn(self, samples):
        samples = list(filter(lambda x: x is not None, samples))
        return dataloader.default_collate(samples)
