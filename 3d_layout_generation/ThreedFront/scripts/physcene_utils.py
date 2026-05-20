#
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
#

import math

import cv2
import numpy as np
import torch
import trimesh

from PIL import Image
from scipy.ndimage import binary_dilation
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh


def map_to_image_coordinate(point, scale, image_size):
    x, y = point
    x_image = int(x / scale * image_size / 2) + image_size / 2
    y_image = int(y / scale * image_size / 2) + image_size / 2
    return x_image, y_image


def calc_bbox_masks(
    bbox,
    class_labels,
    image,
    image_size,
    scale,
    robot_width,
    floor_plan_mask,
    box_wall_count,
):
    """
    type: ["bbox","front_line", "front_center"]
    """
    box_masks = []
    handle_points = []
    for box, class_label in zip(bbox, class_labels):
        box_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        box_wall_mask = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        center = map_to_image_coordinate(box[:3][0::2], scale, image_size)
        size = (
            int(box[3] / scale * image_size / 2) * 2,
            int(box[5] / scale * image_size / 2) * 2,
        )
        angle = box[-1]
        w, h = size
        open_size = 0
        rot_center = center
        handle = np.array([0, h / 2 + open_size + robot_width / 2 + 1])
        bbox = np.array(
            [
                [-w / 2, -h / 2],
                [-w / 2, h / 2 + open_size],
                [w / 2, h / 2 + open_size],
                [w / 2, -h / 2],
            ]
        )

        R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

        box_points = bbox.dot(R) + rot_center
        handle_point = handle.dot(R) + rot_center
        handle_points.append(handle_point)
        box_points = np.intp(box_points)
        # box wall collision
        cv2.fillPoly(box_wall_mask, [box_points], (0, 255, 0))
        # cv2.imwrite(os.path.join(save_path, "debug3.png"), box_wall_mask)
        box_wall_mask = box_wall_mask[:, :, 1] == 255

        if (box_wall_mask * (255 - floor_plan_mask)).sum() > 0:
            box_wall_count += 1
        # cv2.imwrite(os.path.join(save_path, "debug4.png"), floor_plan_mask)

        # image connected region
        cv2.drawContours(
            image, [box_points], 0, (0, 255, 0), robot_width
        )  # Green (BGR)
        cv2.fillPoly(image, [box_points], (0, 255, 0))

        # per box mask
        cv2.drawContours(
            box_mask, [box_points], 0, (0, 255, 0), robot_width
        )  # Green (BGR)
        cv2.fillPoly(box_mask, [box_points], (0, 255, 0))
        st_element = np.ones((3, 3), dtype=bool)
        box_mask = binary_dilation(
            (box_mask[:, :, 1].copy() == 255).astype(image.dtype), st_element
        )
        box_masks.append(box_mask)
    return box_masks, handle_points, box_wall_count, image


def cal_walkable_metric(
    floor_plan_vertices,
    floor_plan_faces,
    floor_plan_centroid,
    bboxes,
    robot_width=0.01,
    calc_object_area=False,
):
    vertices, faces = floor_plan_vertices, floor_plan_faces
    vertices = vertices - floor_plan_centroid
    vertices = vertices[:, 0::2]
    scale = np.abs(vertices).max() + 0.2
    bboxes = bboxes[bboxes[:, 1] < 1.5]

    image_size = 256
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)

    robot_width = int(robot_width / scale * image_size / 2)

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

    kernel = np.ones((robot_width, robot_width))
    image[:, :, 0] = cv2.erode(image[:, :, 0], kernel, iterations=1)
    # draw bboxes
    # cv2.imwrite(os.path.join(save_path, "image1.png"), image)
    for box in bboxes:
        center = map_to_image_coordinate(box[:3][0::2], scale, image_size)
        size = (
            int(box[3] / scale * image_size / 2) * 2,
            int(box[5] / scale * image_size / 2) * 2,
        )
        angle = box[-1]

        # calculate box vertices
        box_points = cv2.boxPoints(((center[0], center[1]), size, -angle / np.pi * 180))
        box_points = np.intp(box_points)

        cv2.drawContours(
            image, [box_points], 0, (0, 255, 0), robot_width
        )  # Green (BGR)
        cv2.fillPoly(image, [box_points], (0, 255, 0))

    # cv2.imwrite(os.path.join(save_path, "image2.png"), image)

    if calc_object_area:
        green_cnt = 0
        blue_cnt = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if list(image[i][j]) == [0, 255, 0]:
                    green_cnt += 1
                elif list(image[i][j]) == [255, 0, 0]:
                    blue_cnt += 1
        object_area_ratio = green_cnt / (blue_cnt + green_cnt)

    walkable_map = image[:, :, 0].copy()
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        walkable_map, connectivity=8
    )

    walkable_map_max = np.zeros_like(walkable_map)
    for label in range(1, num_labels):
        mask = np.zeros_like(walkable_map)
        mask[labels == label] = 255

        if mask.sum() > walkable_map_max.sum():
            # room connected component with door
            walkable_map_max = mask.copy()

        # print("walkable_rate:", walkable_map_max.sum()/walkable_map.sum())
        if calc_object_area:
            return walkable_map_max.sum() / walkable_map.sum(), object_area_ratio
        else:
            return walkable_map_max.sum() / walkable_map.sum()
    if calc_object_area:
        return 0.0, object_area_ratio
    else:
        return 0.0


def voxel_grid_from_mesh(mesh, device, gridsize=0.01):
    # mesh
    try:
        points, faces = mesh.to_points_and_faces()
    except:
        mesh_cnt = len(mesh.renderables)
        points = []
        faces = []
        point_cnt = 0

        for s in range(mesh_cnt):
            p, f = mesh.renderables[s].to_points_and_faces()
            points.append(p)
            faces.append(f + point_cnt)
            point_cnt += p.shape[0]
        points = np.concatenate(points, axis=0)
        faces = np.concatenate(faces, axis=0)

    verts = torch.tensor(points, device=device).unsqueeze(0)
    faces = torch.tensor(faces, device=device).long()

    # grid
    # [xmin,ymin,zmin]
    # [xmax,ymax,zmax]
    box = mesh.bbox
    xmin, ymin, zmin = box[0]
    xmax, ymax, zmax = box[1]
    points = []

    xcnt = round((xmax - xmin) / gridsize + 1)
    ycnt = round((ymax - ymin) / gridsize + 1)
    zcnt = round((zmax - zmin) / gridsize + 1)
    for i in range(xcnt):
        x = xmin + (xmax - xmin) * i / (xcnt - 1)
        for j in range(ycnt):
            y = ymin + (ymax - ymin) * j / (ycnt - 1)
            for k in range(zcnt):
                z = zmin + (zmax - zmin) * k / (zcnt - 1)
                points.append(np.array([[x, y, z]]))

    points = np.concatenate(points, axis=0)[None, :, :]
    pointscuda = torch.tensor(points, device=device)
    try:
        from kaolin.ops.mesh import check_sign
    except ImportError as e:
        raise ImportError(
            "Col_obj / Col_scene overlap metrics need a real NVIDIA Kaolin install "
            "(not the PyPI placeholder). See https://kaolin.readthedocs.io/en/latest/notes/installation.html"
        ) from e
    occupancy = check_sign(verts, faces, pointscuda)

    points_inside = points[0][occupancy.cpu().numpy()[0] == 1]

    # #vis
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points_inside)
    # app = gui.Application.instance
    # app.initialize()
    # vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
    # vis.show_settings = True
    # vis.add_geometry("pcd0",pcd)

    # vis.reset_camera_to_default()
    # app.add_window(vis)
    # app.run()
    return points_inside


def bbox_overlap(render1, render2):
    box1 = np.array(render1.bbox)
    box2 = np.array(render2.bbox)
    if box1[0, 0] >= box2[1, 0] or box2[0, 0] >= box1[1, 0]:  # xmin>xmax
        return False
    if box1[0, 1] >= box2[1, 1] or box2[0, 1] >= box1[1, 1]:  # y
        return False
    if box1[0, 2] >= box2[1, 2] or box2[0, 2] >= box1[1, 2]:  # z
        return False

    return True


def get_textured_objects(bbox_params_t, objects_dataset, classes, cfg=None):
    # For each one of the boxes replace them with an object
    renderables = []
    renderables_remesh = []

    trimesh_meshes = []
    class_num = len(classes)
    furniture_names = []
    scene_info = {"ThreedFront": dict()}
    for data in ["3dfront"]:
        for j in range(bbox_params_t.shape[1]):
            query_label = classes[bbox_params_t[0, j, :class_num].argmax(-1)]
            translation = bbox_params_t[0, j, class_num : class_num + 3]
            query_size = bbox_params_t[0, j, class_num + 3 : class_num + 6]

            theta = bbox_params_t[0, j, class_num + 6]
            # query_objfeat = bbox_params_t[0, j, class_num+7:]

            if query_label == "empty":
                continue
            if data == "3dfront":
                # if use_feature:
                #     furniture = objects_dataset.get_closest_furniture_to_objfeats_and_size(
                #         query_label, query_objfeat, query_size
                #     )
                # else:
                furniture = objects_dataset.get_closest_furniture_to_box(
                    query_label, query_size
                )
                rot = (theta / math.pi * 180) % 360
                print("3D FUTURE ", furniture.label, furniture.model_jid, rot)
                furniture_names.append(furniture.label)

                # Load the furniture and scale it as it is given in the dataset
                # raw mesh
                raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                raw_mesh.scale(furniture.scale)

                # Compute the centroid of the vertices in order to match the
                # bbox (because the prediction only considers bboxes)
                bbox = raw_mesh.bbox
                centroid = (bbox[0] + bbox[1]) / 2

                # Extract the predicted affine transformation to position the
                # mesh
                R = np.zeros((3, 3))
                R[0, 0] = np.cos(theta)
                R[0, 2] = -np.sin(theta)
                R[2, 0] = np.sin(theta)
                R[2, 2] = np.cos(theta)
                R[1, 1] = 1.0

                # Apply the transformations in order to correctly position the mesh
                raw_mesh.affine_transform(t=-centroid)
                raw_mesh.affine_transform(R=R, t=translation)
                renderables.append(raw_mesh)

                # load remesh model for collision rate
                # raw_mesh = TexturedMesh.from_file(furniture.remesh_model_path)
                # raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)

                try:
                    raw_mesh = TexturedMesh.from_file(furniture.remesh_model_path)
                except (ValueError, Exception):
                    # If simple_3dviz fails (e.g., missing UV coords), load with trimesh and convert
                    tr_remesh = trimesh.load(furniture.remesh_model_path, force="mesh")
                    vertices = tr_remesh.vertices
                    faces = tr_remesh.faces

                    # Create UV coordinates from vertex positions (simple cylindrical mapping)
                    uv = np.copy(vertices[:, [0, 2]])  # Use X and Z coordinates
                    uv -= uv.min(axis=0)
                    uv /= uv.max(axis=0) + 1e-6  # Normalize to [0,1]

                    # Create a simple material
                    from simple_3dviz.renderables.textured_mesh import Material

                    material = Material(
                        ambient=(0.5, 0.5, 0.5),
                        diffuse=(0.5, 0.5, 0.5),
                        specular=(0.2, 0.2, 0.2),
                    )

                    # Create TexturedMesh from vertices and faces
                    raw_mesh = TexturedMesh.from_faces(
                        vertices=vertices, faces=faces, uv=uv, material=material
                    )
                raw_mesh.scale(furniture.scale)

                # Apply the transformations in order to correctly position the mesh
                raw_mesh.affine_transform(t=-centroid)
                raw_mesh.affine_transform(R=R, t=translation)
                renderables_remesh.append(raw_mesh)

                # Create a trimesh object for the same mesh in order to save
                # everything as a single scene
                tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
                tr_mesh.visual.material.image = Image.open(furniture.texture_image_path)
                tr_mesh.vertices *= furniture.scale
                tr_mesh.vertices -= centroid
                tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
                trimesh_meshes.append(tr_mesh)

                if furniture.model_jid not in scene_info["ThreedFront"]:
                    scene_info["ThreedFront"][furniture.model_jid] = []
                scene_info["ThreedFront"][furniture.model_jid].append(
                    {
                        "path": furniture.raw_model_path,
                        "scale": furniture.scale,
                        "size": furniture.size.tolist(),
                        "label": furniture.label,
                        "query_label": query_label,
                        "query_size": [float(s / 2) for s in query_size.tolist()],
                        "theta": float(theta),
                        "position": (translation - centroid).tolist(),
                        "centroid": centroid.tolist(),
                    }
                )

    return renderables, trimesh_meshes, furniture_names, renderables_remesh, scene_info


def get_textured_objects_rescale(bbox_params_t, objects_dataset, classes, cfg=None):
    # For each one of the boxes replace them with an object
    renderables = []
    renderables_remesh = []

    trimesh_meshes = []
    use_feature = False  # cfg.task.dataset.use_feature
    class_num = cfg.task.dataset.class_num
    furniture_names = []
    matrial_save = None
    texture_color_lst = []
    scene_info = {"ThreedFront": dict()}
    for data in ["3dfront"]:
        for j in range(bbox_params_t.shape[1]):
            query_label = classes[bbox_params_t[0, j, :class_num].argmax(-1)]  # TODO
            translation = bbox_params_t[0, j, class_num : class_num + 3]
            query_size = bbox_params_t[0, j, class_num + 3 : class_num + 6]

            theta = bbox_params_t[0, j, class_num + 6]
            query_objfeat = bbox_params_t[0, j, class_num + 7 :]

            # MapThreedfuture2gparnet = dict()

            if query_label == "empty":
                continue
            if data == "3dfront":
                if use_feature:
                    furniture = (
                        objects_dataset.get_closest_furniture_to_objfeats_and_size(
                            query_label, query_objfeat, query_size
                        )
                    )
                else:
                    furniture = objects_dataset.get_closest_furniture_to_box(
                        query_label, query_size
                    )
                # scale1 = furniture.scale
                scale = query_size / furniture.size_recal() * furniture.scale
                rot = (theta / math.pi * 180) % 360
                print("3D FUTURE ", furniture.label, furniture.model_jid, rot)
                furniture_names.append(furniture.label)

                # Load the furniture and scale it as it is given in the dataset
                # raw mesh
                raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                raw_mesh.scale(scale)

                # Compute the centroid of the vertices in order to match the
                # bbox (because the prediction only considers bboxes)
                bbox = raw_mesh.bbox
                # centroid = (bbox[0] + bbox[1])/2
                centroid = (bbox[0] + bbox[1]) / 2 * scale
                centroid4TV = [
                    centroid[0] * np.cos(theta) - centroid[2] * np.sin(theta),
                    centroid[1],
                    centroid[0] * np.sin(theta) + centroid[2] * np.cos(theta),
                ]

                # Extract the predicted affine transformation to position the
                # mesh
                R = np.zeros((3, 3))
                R[0, 0] = np.cos(theta)
                R[0, 2] = -np.sin(theta)
                R[2, 0] = np.sin(theta)
                R[2, 2] = np.cos(theta)
                R[1, 1] = 1.0

                # Apply the transformations in order to correctly position the mesh
                raw_mesh.affine_transform(t=-centroid)
                raw_mesh.affine_transform(R=R, t=translation)
                renderables.append(raw_mesh)

                # load remesh model for collision rate
                raw_mesh = TexturedMesh.from_file(furniture.remesh_model_path)
                raw_mesh.scale(scale)

                # Apply the transformations in order to correctly position the mesh
                raw_mesh.affine_transform(t=-centroid)
                raw_mesh.affine_transform(R=R, t=translation)
                renderables_remesh.append(raw_mesh)

                # Create a trimesh object for the same mesh in order to save
                # everything as a single scene
                tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
                tr_mesh.visual.material.image = Image.open(furniture.texture_image_path)
                tr_mesh.vertices *= scale
                tr_mesh.vertices -= centroid
                tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
                trimesh_meshes.append(tr_mesh)

                if furniture.model_jid not in scene_info["ThreedFront"]:
                    scene_info["ThreedFront"][furniture.model_jid] = []
                scene_info["ThreedFront"][furniture.model_jid].append(
                    {
                        "path": furniture.raw_model_path,  #    "scale":scale,\
                        "scale": [float(s) for s in scale.tolist()],
                        "size": furniture.size.tolist(),
                        "label": furniture.label,
                        "query_label": query_label,
                        "query_size": [float(s / 2) for s in query_size.tolist()],
                        "theta": float(theta),
                        "position": (translation - np.array(centroid4TV)).tolist(),
                        "centroid": centroid4TV,
                    }
                )
    return renderables, trimesh_meshes, furniture_names, renderables_remesh, scene_info


def get_bbox_objects(bbox_params_t, classes, cfg=None):
    # For each one of the boxes replace them with an object
    class_num = cfg.task.dataset.class_num

    scene_info = {"objects": []}

    for j in range(bbox_params_t.shape[1]):
        query_label = classes[bbox_params_t[0, j, :class_num].argmax(-1)]  # TODO
        translation = bbox_params_t[0, j, class_num : class_num + 3]
        query_size = bbox_params_t[0, j, class_num + 3 : class_num + 6]
        theta = bbox_params_t[0, j, class_num + 6]
        rot = (theta / math.pi * 180) % 360

        if query_label == "empty":
            continue

        scene_info["objects"].append(
            {
                "label": query_label,
                "position": (translation).tolist(),
                "scale": query_size.tolist(),
                "theta": float(theta),
                "rot_degree": float(rot),
            }
        )

    return scene_info


def get_textured_objects_from_bbox(bboxes, objects_dataset, classes, cfg=None):
    # For each one of the boxes replace them with an object
    renderables = []
    renderables_remesh = []

    trimesh_meshes = []
    use_feature = False
    furniture_names = []
    scene_info = {"ThreedFront": dict()}
    for data in ["3dfront", "gpn"]:
        for bbox in bboxes:
            query_label = bbox["label"]
            translation = np.array(bbox["position"])
            query_size = np.array(bbox["scale"])
            theta = bbox["theta"]
            # query_objfeat = bbox_params_t[0, j, class_num+7:]

            if data == "3dfront":
                furniture = objects_dataset.get_closest_furniture_to_box(
                    query_label, query_size
                )
                rot = (theta / math.pi * 180) % 360
                print("3D FUTURE ", furniture.label, furniture.model_jid, rot)
                furniture_names.append(furniture.label)

                # Load the furniture and scale it as it is given in the dataset
                # raw mesh
                raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                raw_mesh.scale(furniture.scale)

                # Compute the centroid of the vertices in order to match the
                # bbox (because the prediction only considers bboxes)
                bbox = raw_mesh.bbox
                centroid = (bbox[0] + bbox[1]) / 2

                # Extract the predicted affine transformation to position the
                # mesh
                R = np.zeros((3, 3))
                R[0, 0] = np.cos(theta)
                R[0, 2] = -np.sin(theta)
                R[2, 0] = np.sin(theta)
                R[2, 2] = np.cos(theta)
                R[1, 1] = 1.0

                # Apply the transformations in order to correctly position the mesh
                raw_mesh.affine_transform(t=-centroid)
                raw_mesh.affine_transform(R=R, t=translation)
                renderables.append(raw_mesh)

                # load remesh model for collision rate
                raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
                raw_mesh.scale(furniture.scale)

                # Apply the transformations in order to correctly position the mesh
                raw_mesh.affine_transform(t=-centroid)
                raw_mesh.affine_transform(R=R, t=translation)
                renderables_remesh.append(raw_mesh)

                # Create a trimesh object for the same mesh in order to save
                # everything as a single scene
                tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
                tr_mesh.visual.material.image = Image.open(furniture.texture_image_path)
                tr_mesh.vertices *= furniture.scale
                tr_mesh.vertices -= centroid
                tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + translation
                trimesh_meshes.append(tr_mesh)

                if furniture.model_jid not in scene_info["ThreedFront"]:
                    scene_info["ThreedFront"][furniture.model_jid] = []
                scene_info["ThreedFront"][furniture.model_jid].append(
                    {
                        "path": furniture.raw_model_path,
                        "scale": furniture.scale,
                        "label": furniture.label,
                        "theta": float(theta),
                        "position": (translation - centroid).tolist(),
                    }
                )

    return renderables, trimesh_meshes, furniture_names, renderables_remesh, scene_info


def get_floor_plan(scene, floor_textures):
    """Return the floor plan of the scene as a trimesh mesh and a simple-3dviz
    TexturedMesh."""
    vertices, faces = scene.floor_plan
    vertices = vertices - scene.floor_plan_centroid
    uv = np.copy(vertices[:, [0, 2]])
    uv -= uv.min(axis=0)
    uv /= 0.3  # repeat every 30cm
    texture = np.random.choice(floor_textures)

    floor = TexturedMesh.from_faces(
        vertices=vertices,
        uv=uv,
        faces=faces,
        material=Material.with_texture_image(texture),
    )

    tr_floor = trimesh.Trimesh(np.copy(vertices), np.copy(faces), process=False)
    tr_floor.visual = trimesh.visual.TextureVisuals(
        uv=np.copy(uv),
        material=trimesh.visual.material.SimpleMaterial(image=Image.open(texture)),
    )

    return floor, tr_floor
