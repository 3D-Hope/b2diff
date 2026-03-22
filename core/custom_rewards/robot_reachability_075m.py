import cv2
import numpy as np
import torch


def _map_to_image_coordinate(point_xz, scale, image_size):
    x, z = point_xz
    x_image = int(x / scale * image_size / 2 + image_size / 2)
    z_image = int(z / scale * image_size / 2 + image_size / 2)
    return x_image, z_image


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _compute_scene_walkable_rate(
    floor_plan_vertices,
    floor_plan_faces,
    floor_plan_centroid,
    positions,
    sizes,
    angles,
    image_size=256,
    robot_real_width=0.75,
):
    vertices = floor_plan_vertices - floor_plan_centroid
    vertices_xz = vertices[:, [0, 2]]
    scale = float(np.abs(vertices_xz).max() + 0.2)

    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    robot_width_px = int(robot_real_width / scale * image_size / 2)
    robot_width_px = max(1, robot_width_px)

    # Draw floor polygon(s) in blue channel.
    for face in floor_plan_faces:
        face_vertices = vertices_xz[face]
        face_vertices_image = [
            _map_to_image_coordinate(v, scale, image_size) for v in face_vertices
        ]
        pts = np.array(face_vertices_image, np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(image, [pts], (255, 0, 0))

    # Erode free-space by robot footprint width.
    kernel = np.ones((robot_width_px, robot_width_px), dtype=np.uint8)
    image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)

    # Draw object footprints as obstacles (inflated by robot width).
    # Keep per-object masks for object-wise reachability shaping.
    box_masks = []
    for i in range(positions.shape[0]):
        center = _map_to_image_coordinate(positions[i, [0, 2]], scale, image_size)
        size = (
            int(sizes[i, 0] / scale * image_size / 2) * 2,
            int(sizes[i, 2] / scale * image_size / 2) * 2,
        )
        size = (max(2, abs(size[0])), max(2, abs(size[1])))

        angle = float(angles[i])
        box_points = cv2.boxPoints(((center[0], center[1]), size, -angle / np.pi * 180.0))
        box_points = np.intp(box_points)

        cv2.drawContours(image, [box_points], 0, (0, 255, 0), robot_width_px)
        cv2.fillPoly(image, [box_points], (0, 255, 0))

        box_mask = np.zeros((image_size, image_size), dtype=np.uint8)
        cv2.drawContours(box_mask, [box_points], 0, 255, robot_width_px)
        cv2.fillPoly(box_mask, [box_points], 255)
        box_masks.append(box_mask > 0)

    walkable_map = image[:, :, 0]
    total_walkable = int(np.count_nonzero(walkable_map))
    if total_walkable == 0:
        # No free space: everything is unreachable.
        if len(box_masks) == 0:
            return 0.0, 0.0, 0.0
        return 0.0, 0.0, float(len(box_masks))

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(walkable_map, connectivity=8)

    largest_component_area = 0
    component_masks = []
    for label in range(1, num_labels):
        component_mask = labels == label
        area = int(np.count_nonzero(component_mask))
        if area > largest_component_area:
            largest_component_area = area
        component_masks.append(component_mask)

    # Object-wise reachability ratio in [0, 1].
    # 1.0 means object mask touches walkable space everywhere (best),
    # 0.0 means no touch with any walkable component (unreachable).
    object_reach_ratios = []
    for box_mask in box_masks:
        object_area = int(np.count_nonzero(box_mask))
        if object_area == 0:
            object_reach_ratios.append(0.0)
            continue

        best_overlap = 0
        for component_mask in component_masks:
            overlap = int(np.count_nonzero(np.logical_and(box_mask, component_mask)))
            if overlap > best_overlap:
                best_overlap = overlap

        object_reach_ratios.append(float(best_overlap) / float(object_area))

    if len(object_reach_ratios) == 0:
        mean_reach_ratio = 0.0
        unreachability_sum = 0.0
    else:
        mean_reach_ratio = float(np.mean(object_reach_ratios))
        unreachability_sum = float(np.sum([1.0 - r for r in object_reach_ratios]))

    walkable_rate = float(largest_component_area) / float(total_walkable)
    mean_reach_ratio = float(min(max(mean_reach_ratio, 0.0), 1.0))
    return walkable_rate, mean_reach_ratio, unreachability_sum


def compute_robot_walkability_075m_reward(
    parsed_scene,
    floor_plan_vertices,
    floor_plan_faces,
    floor_plan_centroid,
    robot_real_width=0.1,
    reachable_weight=1.0,
    unreachable_penalty_scale=0.5,
    walkable_weight=0.25,
    empty_scene_penalty=-1.0,
    image_size=256,
    **kwargs,
):
    """
    Object-centric reachability reward for a robot (self-contained).

    Per scene:
     1) Build free-space map with robot width erosion.
     2) For each object, compute reach_ratio_i in [0,1] using overlap with walkable components.
     3) Use a simple linear reward composed of:
         - walkable connectedness
         - mean object reachability
         - linear penalty for mean unreachability

    This gives:
    This avoids large nonlinear jumps and keeps reward scale stable.
    """
    positions = parsed_scene["positions"]
    sizes = parsed_scene["sizes"]
    angles = parsed_scene.get("angles", None)
    orientations = parsed_scene.get("orientations", None)
    is_empty = parsed_scene["is_empty"]


    device = positions.device
    dtype = positions.dtype
    batch_size = positions.shape[0]

    rewards = torch.zeros(batch_size, dtype=dtype, device=device)

    for b in range(batch_size):
        valid_mask = ~is_empty[b]
        if int(valid_mask.sum().item()) == 0:
            rewards[b] = empty_scene_penalty
            continue

        pos_b = _to_numpy(positions[b][valid_mask])
        size_b = _to_numpy(sizes[b][valid_mask])

        if angles is not None:
            angle_b_t = angles[b][valid_mask]
            if angle_b_t.ndim > 1:
                angle_b_t = angle_b_t[..., 0]
            angle_b = _to_numpy(angle_b_t).reshape(-1)
        elif orientations is not None:
            # Fallback path for callers that only pass [cos, sin].
            orient_b = _to_numpy(orientations[b][valid_mask])
            angle_b = np.arctan2(orient_b[:, 1], orient_b[:, 0]).reshape(-1)
        else:
            raise ValueError("parsed_scene must provide either angles or orientations")

        fp_vertices_b = _to_numpy(floor_plan_vertices[b])
        fp_faces_b = _to_numpy(floor_plan_faces[b])
        fp_centroid_b = _to_numpy(floor_plan_centroid[b])

        walkable_rate, mean_reach_ratio, unreachability_sum = _compute_scene_walkable_rate(
            floor_plan_vertices=fp_vertices_b,
            floor_plan_faces=fp_faces_b,
            floor_plan_centroid=fp_centroid_b,
            positions=pos_b,
            sizes=size_b,
            angles=angle_b,
            image_size=image_size,
            robot_real_width=robot_real_width,
        )

        obj_count = float(max(int(valid_mask.sum().item()), 1))
        mean_unreachability = float(unreachability_sum) / obj_count

        rewards[b] = (
            float(walkable_weight) * walkable_rate
            + float(reachable_weight) * mean_reach_ratio
            - float(unreachable_penalty_scale) * mean_unreachability
        )

    return rewards


def compute_reward(parsed_scene, **kwargs):
    """Generic custom-reward entrypoint used by dynamic loader."""
    passthrough_kwargs = dict(kwargs)
    floor_plan_vertices = passthrough_kwargs.pop("floor_plan_vertices", None)
    floor_plan_faces = passthrough_kwargs.pop("floor_plan_faces", None)
    floor_plan_centroid = passthrough_kwargs.pop("floor_plan_centroid", None)

    if floor_plan_vertices is None or floor_plan_faces is None or floor_plan_centroid is None:
        raise ValueError(
            "robot_reachability_075m requires floor_plan_vertices, floor_plan_faces, "
            "and floor_plan_centroid. Set midiffusion.floor_geometry_path and pass "
            "geometry via selection.py."
        )

    with torch.no_grad():
        return compute_robot_walkability_075m_reward(
            parsed_scene,
            floor_plan_vertices=floor_plan_vertices,
            floor_plan_faces=floor_plan_faces,
            floor_plan_centroid=floor_plan_centroid,
            **passthrough_kwargs,
        )
