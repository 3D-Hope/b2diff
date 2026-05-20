import argparse

# from threed_front.evaluation.utils import count_out_of_boundary, compute_bbox_iou
import os
import pickle

import sys

import cv2
import numpy as np
import torch

from physcene_utils import (
    bbox_overlap,
    cal_walkable_metric,
    calc_bbox_masks,
    get_textured_objects,
    map_to_image_coordinate,
)
from threed_front.datasets.threed_future_dataset import ThreedFutureDataset
from threed_front.evaluation import ThreedFrontResults
from utils import PATH_TO_PICKLED_3D_FUTURE_MODEL, PROJ_DIR as THREEDFRONT_PROJ_DIR
from tqdm import tqdm

# MiDiffusion dataset helpers (no steerable_scene_generation required)
_THREEDFRONT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_MIDIFFUSION_ROOT = os.path.normpath(os.path.join(_THREEDFRONT_ROOT, "..", "MiDiffusion"))
if _MIDIFFUSION_ROOT not in sys.path:
    sys.path.insert(0, _MIDIFFUSION_ROOT)
from scripts.utils import update_data_file_paths  # noqa: E402
from midiffusion.datasets.threed_front_encoding import get_dataset_raw_and_encoded  # noqa: E402

"""Script for calculating physcene metrics. """


def calc_wall_overlap(
    threed_front_results,
    raw_dataset,
    encoded_dataset,
    cfg,
    robot_real_width,
    calc_object_area=False,
    classes=None,
):
    box_wall_count = 0
    accessable_count = 0
    box_count = 0
    walkable_metric_list = []
    accessable_rate_list = []
    object_area_ratio = 0.0

    for scene_idx, scene_layout in tqdm(threed_front_results):
        raw_room = raw_dataset[scene_idx]
        floor_plan_vertices, floor_plan_faces = raw_room.floor_plan
        floor_plan_centroid = raw_room.floor_plan_centroid

        # Exclude ceiling-mounted lights from 2D reachability/wall metrics.
        excluded_class_ids = np.array([3, 13], dtype=np.int64)
        scene_class_labels = scene_layout["class_labels"]
        if scene_class_labels.ndim == 2:
            class_ids = np.argmax(scene_class_labels, axis=1)
        else:
            class_ids = scene_class_labels.astype(np.int64)
        valid_idx = ~np.isin(class_ids, excluded_class_ids)
        class_labels = scene_class_labels[valid_idx]
        bbox = np.concatenate(
            [
                scene_layout["translations"][valid_idx],
                scene_layout["sizes"][valid_idx],
                scene_layout["angles"][valid_idx],
            ],
            axis=-1,
        )

        vertices, faces = floor_plan_vertices, floor_plan_faces
        vertices = vertices - floor_plan_centroid
        vertices = vertices[:, 0::2]
        # vertices = vertices[:, :2]
        scale = np.abs(vertices).max() + 0.2

        image_size = 256
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

        robot_width = int(robot_real_width / scale * image_size / 2)

        # draw face
        for face in faces:
            face_vertices = vertices[face]
            face_vertices_image = [
                map_to_image_coordinate(v, scale, image_size) for v in face_vertices
            ]

            pts = np.array(face_vertices_image, np.int32)
            pts = pts.reshape(-1, 1, 2)
            color = (255, 0, 0)  # Blue (BGR)
            cv2.fillPoly(image, [pts], color)

        floor_plan_mask = (image[:, :, 0] == 255) * 255
        # cv2.imwrite(os.path.join(save_path, "debug_floor.png"), floor_plan_mask)
        # 缩小墙边界，机器人行动范围
        kernel = np.ones((robot_width, robot_width))
        image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)

        box_masks, handle_points, box_wall_count, image = calc_bbox_masks(
            bbox,
            class_labels,
            image,
            image_size,
            scale,
            robot_width,
            floor_plan_mask,
            box_wall_count,
        )

        # Empty-scene convention: fully walkable, zero accessibility and zero wall-box penalty.
        if len(box_masks) == 0:
            walkable_metric_list.append(1.0)
            accessable_rate_list.append(0.0)
            if calc_object_area:
                object_area_ratio = 0.0
            continue

        # cv2.imwrite(os.path.join(save_path, "debug.png"), image)
        # breakpoint()
        walkable_map = image[:, :, 0].copy()
        # cv2.imwrite(os.path.join(save_path, "debug2.png"), walkable_map)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            walkable_map, connectivity=8
        )
        # 遍历每个连通域

        accessable_rate = 0
        walkable_pixel_count = (labels != 0).sum()
        for label in range(1, num_labels):
            mask = np.zeros_like(walkable_map)
            mask[labels == label] = 1
            accessable_count = 0
            for box_mask in box_masks:
                if (box_mask * mask).sum() > 0:
                    accessable_count += 1
            if walkable_pixel_count > 0:
                accessable_rate += (
                    accessable_count / len(box_masks)
                    * mask.sum()
                    / walkable_pixel_count
                )
        accessable_rate_list.append(accessable_rate)
        box_count += len(box_masks)

        # walkable map area rate
        if calc_object_area:
            walkable_rate, object_area_ratio = cal_walkable_metric(
                floor_plan_vertices,
                floor_plan_faces,
                floor_plan_centroid,
                bbox,
                robot_width=robot_real_width,
                calc_object_area=True,
            )
        else:
            walkable_rate = cal_walkable_metric(
                floor_plan_vertices,
                floor_plan_faces,
                floor_plan_centroid,
                bbox,
                robot_width=robot_real_width,
            )
        walkable_metric_list.append(walkable_rate)

    walkable_average_rate = sum(walkable_metric_list) / len(walkable_metric_list)
    accessable_rate = sum(accessable_rate_list) / len(accessable_rate_list)
    # accessable_handle_rate = sum(accessable_handle_rate_list)/len(accessable_handle_rate_list)
    box_wall_rate = box_wall_count / box_count if box_count > 0 else 0.0

    # print(the index of the scenes that has less than 50 % accessable_rate)
    print("Scenes with less than 50% accessable_rate:")
    bad_scene_count = 0
    for i, rate in enumerate(accessable_rate_list):
        if rate < 0.5:
            print(f"  Scene {i}: {rate:.2f}", end=", ")
            bad_scene_count += 1

    print(f"Total scenes: {len(accessable_rate_list)}, Scenes with <50% accessable_rate: {bad_scene_count}")
    print("walkable_average_rate:", walkable_average_rate)
    print("accessable_rate:", accessable_rate)
    # print('accessable_handle_rate:', accessable_handle_rate)
    print("box_wall_rate:", box_wall_rate)
    if calc_object_area:
        print("object_area_ratio:", object_area_ratio)
        return walkable_average_rate, accessable_rate, box_wall_rate, object_area_ratio
    else:
        return walkable_average_rate, accessable_rate, box_wall_rate


def calc_overlap(
    threed_front_results,
    raw_dataset,
    encoded_dataset,
    cfg,
    visualize_overlap=False,
    path_to_3d_future_template=PATH_TO_PICKLED_3D_FUTURE_MODEL,
):
    # Get classes here
    classes = raw_dataset.class_labels[:-1]  # Removing the empty class (end)
    print("classes:", classes)
    print("classes shape:", len(list(classes)))
    device = "cuda"
    # class_num = classes.shape[0]-1
    a = 0
    overlap_cnt_total = 0
    obj_cnt_total = 0
    overlap_scene = 0
    scene_cnt = 0

    overlap_area = 0
    overlap_area_max = 0
    obj_overlap_cnt = 0

    room_type = cfg["data"].get("room_type")
    if room_type is None:
        for name in ("diningroom", "livingroom", "bedroom", "library"):
            if name in cfg["data"]["dataset_directory"]:
                room_type = name
                break
    if room_type is None:
        raise ValueError("Could not infer room_type from config['data']['dataset_directory']")
    path_to_pickled_3d_future_models = path_to_3d_future_template.format(room_type)
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        path_to_pickled_3d_future_models
    )
    print("Loaded {} 3D-FUTURE models".format(len(objects_dataset)))

    cnt = min(100, len(threed_front_results))
    for scene_idx, scene_layout in tqdm(threed_front_results):
        # raw_item = encoded_dataset[scene_idx]
        a += 1
        print("visualizing scene ", a, "/", cnt)
        boxes = scene_layout
        bbox_params = np.concatenate(
            [
                boxes["class_labels"],
                boxes["translations"],
                boxes["sizes"],
                boxes["angles"],
            ],
            axis=-1,
        )[None, :, :]

        renderables, _, _, renderables_remesh, _ = get_textured_objects(
            bbox_params, objects_dataset, classes, cfg
        )
        obj_cnt = len(renderables)

        overlap_flag = np.zeros(obj_cnt)
        overlap_depths = np.zeros(obj_cnt)
        for i in range(obj_cnt):
            try:
                points, faces = renderables_remesh[i].to_points_and_faces()
            except:
                mesh_cnt = len(renderables_remesh[i].renderables)
                points = []
                faces = []
                point_cnt = 0

                for s in range(mesh_cnt):
                    p, f = renderables_remesh[i].renderables[s].to_points_and_faces()
                    points.append(p)
                    faces.append(f + point_cnt)
                    point_cnt += p.shape[0]
                points = np.concatenate(points, axis=0)
                faces = np.concatenate(faces, axis=0)

            if visualize_overlap:
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(points)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

                pcd0 = o3d.geometry.PointCloud()
                pcd0.points = o3d.utility.Vector3dVector(points)
                pcd0.colors = o3d.utility.Vector3dVector([[0, 1, 0]])

            verts = torch.tensor(points, device=device).unsqueeze(0)
            faces = torch.tensor(faces, device=device).long()
            for j in range(i + 1, obj_cnt):
                if overlap_flag[i] and overlap_flag[j]:
                    continue
                # obj2
                if not bbox_overlap(renderables_remesh[i], renderables[j]):
                    continue
                try:
                    points = renderables[j].to_points_and_faces()[0][None, :, :]
                except:
                    mesh_cnt = len(renderables[j].renderables)
                    points = [
                        renderables[j].renderables[s].to_points_and_faces()[0]
                        for s in range(mesh_cnt)
                    ]
                    points = np.concatenate(points, axis=0)[None, :, :]
                pointscuda = torch.tensor(points, device=device)
                from kaolin.ops.mesh import check_sign
                occupancy = check_sign(verts, faces, pointscuda)

                if occupancy.max() > 0:
                    overlap_flag[i] = 1
                    overlap_flag[j] = 1

        print(overlap_flag)
        print(overlap_depths)
        overlap_cnt_total += overlap_flag.sum()
        obj_cnt_total += obj_cnt

        overlap_scene += overlap_flag.sum() > 0
        scene_cnt += 1

        overlap_area += overlap_depths.sum()
        overlap_area_max += overlap_depths.max()

        obj_overlap_cnt += sum(overlap_depths > 0)

        # if visualize_overlap:
        #     show_renderables(renderables)
    overlap_ratio = overlap_cnt_total / obj_cnt_total
    print(
        "overlap object: ", overlap_ratio, "cnt ", overlap_cnt_total, "/", obj_cnt_total
    )
    overlap_scene_rate = overlap_scene / scene_cnt
    print("overlap scene rate: ", overlap_scene_rate)
    print("overlap_area_mean: ", overlap_area / obj_cnt_total)
    print("overlap_area_max : ", overlap_area_max / scene_cnt)
    print("overlap_area_mean_only_overlaped : ", overlap_area / obj_overlap_cnt)

    return overlap_ratio, overlap_scene_rate


def main(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Compute the FID scores between the real and the " "synthetic images"
        )
    )
    parser.add_argument(
        "result_file", help="Path to a pickled result file (ThreedFrontResults object)"
    )

    parser.add_argument(
        "--robot_real_width",
        type=float,
        default=0.3,
        help="The real width of the robot in meters (default: 0.3m)",
    )
    parser.add_argument(
        "--filtered_gt",
        action="store_true",
        help=(
            "Whether the input result file is from filtered GT export. If set, "
            "the script will use the embedded dataset in the results object for "
            "scene_idx lookup, ensuring consistency with the filtered GT scenes."
        ),
    )
    parser.add_argument(
        "--overlap",
        action="store_true",
        help="Also compute Col_obj / Col_scene (requires Kaolin + remesh 3D-FUTURE pickles)",
    )
    parser.add_argument(
        "--remesh",
        action="store_true",
        help="Use threed_future_model_<room>_remesh.pkl instead of standard pickle",
    )
    args = parser.parse_args(argv)

    # Load saved results
    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)
    assert threed_front_results.floor_condition

    config = threed_front_results.config
    config_data = update_data_file_paths(dict(config["data"]))
    max_length = config["network"].get("sample_num_points", 21)

    if args.filtered_gt:
        splits = ["train", "val", "test"]
    else:
        splits = config.get("validation", {}).get("splits", ["test"])

    raw_dataset, encoded_dataset = get_dataset_raw_and_encoded(
        config_data,
        split=splits,
        max_length=max_length,
        include_room_mask=config["network"].get("room_mask_condition", True),
    )
    print(f"Loaded {len(encoded_dataset)} scenes (splits={splits})")



    walkable_average_rate, accessable_rate, box_wall_rate = calc_wall_overlap(
        threed_front_results,
        raw_dataset,
        encoded_dataset,
        config,
        robot_real_width=args.robot_real_width,
        calc_object_area=False,
        classes=None,
    )
    print("walkable_average_rate, Rwalkable:", walkable_average_rate)
    print("accessable_rate, Rreach:", accessable_rate)
    print("box_wall_rate, Rout:", box_wall_rate)

    if args.overlap:
        future_pkl_template = PATH_TO_PICKLED_3D_FUTURE_MODEL
        if args.remesh:
            future_pkl_template = os.path.join(
                THREEDFRONT_PROJ_DIR, "output/threed_future_model_{}_remesh.pkl"
            )
        overlap_ratio, overlap_scene_rate = calc_overlap(
            threed_front_results,
            raw_dataset,
            encoded_dataset,
            cfg=config,
            visualize_overlap=False,
            path_to_3d_future_template=future_pkl_template,
        )
        print("overlap_ratio, Col_obj:", overlap_ratio)
        print("overlap_scene_rate, Col_scene:", overlap_scene_rate)


# Walkable Average Rate(walkable_average_rate): Path exists in the scene for the robot to walk through
# Accessable Rate(accessable_rate): The robot can access every object in the scene
# Box Wall Rate(box_wall_rate): Objects in scene are going out of bounds or not
# Overlap Ratio(overlap_ratio) (Col Object): The ratio of the number of overlapping objects to the total number of objects
# Overlap Scene Rate(overlap_scene_rate) (Col Scene): The ratio of the number of scenes with overlapping objects to the total number of scenes
if __name__ == "__main__":
    main(None)
