# 
# Licensed under the NVIDIA Source Code License.
# Modified from https://github.com/nv-tlabs/ATISS.
# 

import os
from typing import Union, Tuple, List
import numpy as np
import torch
from PIL import Image
import trimesh
from pyrr import Matrix44

from simple_3dviz import Mesh, Scene
from simple_3dviz.renderables.textured_mesh import Material, TexturedMesh
from simple_3dviz.renderables import Renderable, Lines, Spherecloud
from simple_3dviz.utils import save_frame

from threed_front.datasets.threed_front_scene import Room
from threed_front.datasets.threed_front import CachedRoom
from threed_front.datasets.threed_front_scene import rotation_matrix_around_y


""" Helper functions """

def scene_from_args(args):
    # Create the scene and the behaviour list for simple-3dviz
    scene = Scene(
        size=args.get("window_size", (256, 256)), 
        background=args.get("background", (1, 1, 1, 1))
    )
    scene.up_vector = args.get("up_vector")
    scene.camera_target = args.get("camera_target")
    scene.camera_position = args.get("camera_position")
    scene.light = args.get("camera_position")
    if "room_side" in args: 
        scene.camera_matrix = Matrix44.orthogonal_projection(
            left=-args.get("room_side"), right=args.get("room_side"),
            bottom=args.get("room_side"), top=-args.get("room_side"),
            near=0.1, far=6)
    return scene


""" Floor plan rednerable """

def get_floor_plan(scene: Union[Room, CachedRoom], texture=None, 
                   color=(0.87, 0.72, 0.53), with_trimesh=False,
                   with_room_mask=False) \
    -> Tuple[TexturedMesh, trimesh.Trimesh, np.ndarray]:
    """Return the floor plan of the scene as a simple-3dviz TexturedMesh, a trimesh mesh, 
    and an optional binary numpy array."""
    vertices, faces = scene.floor_plan
    vertices -= scene.floor_plan_centroid

    if texture is not None:
        uv = np.copy(vertices[:, [0, 2]])
        uv -= uv.min(axis=0)
        uv /= 0.3  # repeat every 30cm
        floor = TexturedMesh.from_faces(
            vertices=np.copy(vertices), uv=np.copy(uv), faces=np.copy(faces),
            material=Material.with_texture_image(texture)
        )
    else:
        floor = Mesh.from_faces(
            vertices=np.copy(vertices), faces=np.copy(faces), colors=color
        )

    if with_trimesh:
        tr_floor = trimesh.Trimesh(
            vertices=np.copy(vertices), faces=np.copy(faces), process=False
        )
        if texture is not None:
            tr_floor.visual = trimesh.visual.TextureVisuals(
                uv=np.copy(uv), image=Image.open(texture)
            )
        else:
            tr_floor.visual.face_colors = np.tile(color, (faces.shape[0], 1))
    else:
        tr_floor = None
    
    if with_room_mask:
        room_mask = torch.from_numpy(
            np.transpose(scene.room_mask[None, :, :, 0:1], (0, 3, 1, 2)))
    else:
        room_mask = None
    
    return floor, tr_floor, room_mask


""" Wall renderables """

def build_wall_renderables(floor_plan_ordered_corners, wall_height,
                            color=(1.0, 1.0, 1.0), with_trimesh=False):
    """Build double-sided wall meshes from ordered floor plan corners (XZ, centroid-centered).

    Args:
        floor_plan_ordered_corners: (N, 2) float array of XZ wall corners,
            already centered on floor_plan_centroid (same frame as floor vertices).
        wall_height: float, height of walls in metres (Y axis = up).
        color: RGB tuple for flat wall color.
        with_trimesh: if True, also return a trimesh.Trimesh object.

    Returns:
        wall_mesh: simple_3dviz Mesh
        tr_walls:  trimesh.Trimesh or None
    """
    fpoc = np.asarray(floor_plan_ordered_corners, dtype=np.float32)
    n = fpoc.shape[0]

    all_verts = []
    all_faces = []
    vc = 0  # running vertex count

    for i in range(n):
        x0, z0 = fpoc[i]
        x1, z1 = fpoc[(i + 1) % n]

        # Four corners of this wall quad
        BL = [x0, 0.0,         z0]
        BR = [x1, 0.0,         z1]
        TR = [x1, wall_height, z1]
        TL = [x0, wall_height, z0]

        all_verts.extend([BL, BR, TR, TL])

        # Front face (CCW from outside)
        all_faces.append([vc,   vc+1, vc+2])
        all_faces.append([vc,   vc+2, vc+3])
        # Back face (reverse winding, so wall is visible from inside too)
        all_faces.append([vc+2, vc+1, vc  ])
        all_faces.append([vc+3, vc+2, vc  ])

        vc += 4

    vertices = np.array(all_verts, dtype=np.float32)
    faces    = np.array(all_faces, dtype=np.int32)

    wall_mesh = Mesh.from_faces(vertices=np.copy(vertices), faces=np.copy(faces), colors=color)

    if with_trimesh:
        tr_walls = trimesh.Trimesh(vertices=np.copy(vertices), faces=np.copy(faces), process=False)
        tr_walls.visual.face_colors = np.tile(
            [int(c * 255) for c in color] + [255], (faces.shape[0], 1)
        )
    else:
        tr_walls = None

    return wall_mesh, tr_walls


def _build_furn_polys(translations, sizes, angles):
    """Build 2D shapely Polygons for each furniture piece in the XZ plane."""
    from shapely.geometry import Polygon
    polys = []
    if translations is None or len(translations) == 0:
        return polys
    for j in range(len(translations)):
        tx, tz = float(translations[j, 0]), float(translations[j, 2])
        sx = float(sizes[j, 0])   # half-extent X
        sz = float(sizes[j, 2])   # half-extent Z
        theta = float(angles[j]) if angles is not None else 0.0
        corners = np.array([[-sx, -sz], [-sx, sz], [sx, sz], [sx, -sz]])
        ca, sa = np.cos(theta), np.sin(theta)
        R2 = np.array([[ca, -sa], [sa, ca]])
        world = corners @ R2.T + np.array([tx, tz])
        try:
            poly = Polygon(world)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_valid and poly.area > 1e-6:
                polys.append(poly)
        except Exception:
            pass
    return polys


def _door_swing_poly(hinge_xz, d_along, n_inward, door_width, door_open_angle, n_arc=20):
    """Return a shapely Polygon for the door swing sector (arc fan in XZ)."""
    from shapely.geometry import Polygon
    pts = [tuple(hinge_xz)]
    for k in range(n_arc + 1):
        angle = np.deg2rad(door_open_angle * k / n_arc)
        ca, sa = np.cos(angle), np.sin(angle)
        swing_dir = ca * d_along + sa * n_inward
        pts.append(tuple(hinge_xz + door_width * swing_dir))
    try:
        poly = Polygon(pts)
        return poly.buffer(0) if not poly.is_valid else poly
    except Exception:
        return None


def find_door_placement(fpoc, furn_polys, door_width=0.9, door_open_angle=60.0,
                        margin=0.15, step=0.05):
    """Slide a door opening along every wall edge and find the collision-free spot.

    For each candidate hinge position, the door swing sector (fan of radius=door_width,
    from wall-flush to door_open_angle degrees inward) is tested against every furniture
    footprint polygon using shapely.  The position with minimum overlap area wins.

    Args:
        fpoc: (N, 2) XZ centroid-centered ordered polygon corners.
        furn_polys: list of shapely Polygons for furniture footprints.
        door_width: door opening width (metres).
        door_open_angle: degrees door swings open into room.
        margin: minimum distance (metres) between door jamb and wall corner.
        step: scanning resolution along each edge (metres).

    Returns:
        (edge_idx, u_hinge, hinge_xz, d_along, n_inward)  — all numpy arrays/floats
        or None if no feasible edge found.
    """
    n = fpoc.shape[0]
    best_result = None
    best_score = float('inf')

    for i in range(n):
        p0 = fpoc[i]
        p1 = fpoc[(i + 1) % n]
        edge_vec = p1 - p0
        edge_len = float(np.linalg.norm(edge_vec))

        if edge_len < door_width + 2 * margin:
            continue

        d = edge_vec / edge_len                    # unit vector along edge
        n_in = np.array([-d[1], d[0]])             # perpendicular (one of two directions)
        # Ensure n_in points toward room interior (toward centroid ≈ origin)
        wall_mid = (p0 + p1) * 0.5
        if np.dot(n_in, -wall_mid) < 0:
            n_in = -n_in

        u = margin
        u_end = edge_len - door_width - margin
        while u <= u_end + 1e-9:
            hinge_xz = p0 + d * u
            swing = _door_swing_poly(hinge_xz, d, n_in, door_width, door_open_angle)
            if swing is None:
                u += step
                continue

            score = 0.0
            for fp in furn_polys:
                try:
                    if swing.intersects(fp):
                        score += swing.intersection(fp).area
                except Exception:
                    pass

            if score < best_score:
                best_score = score
                best_result = (i, u, hinge_xz.copy(), d.copy(), n_in.copy())

            if score == 0.0:
                break   # perfect spot on this edge — no need to scan further

            u += step

    return best_result, best_score


def build_walls_with_door(
        floor_plan_ordered_corners,
        wall_height,
        translations=None,
        sizes=None,
        angles=None,
        wall_color=(0.94, 0.92, 0.88),
        door_width=0.9,
        door_height=2.0,
        door_color=(0.55, 0.35, 0.15),
        door_open_angle=60.0,
        with_trimesh=False):
    """Build double-sided walls with a collision-aware half-open door.

    Args:
        floor_plan_ordered_corners: (N, 2) XZ centroid-centered corners.
        wall_height: float metres.
        translations: (J, 3) ground-furniture world translations.
        sizes: (J, 3) furniture half-extents.
        angles: (J,) or (J,1) furniture rotation around Y.
        wall_color, door_width, door_height, door_color, door_open_angle: see above.
        with_trimesh: if True also return trimesh.Trimesh objects.

    Returns:
        wall_mesh, door_mesh, tr_walls, tr_door
    """
    fpoc = np.asarray(floor_plan_ordered_corners, dtype=np.float64)
    n = fpoc.shape[0]

    # Normalise angles to 1D
    ang = None
    if angles is not None:
        ang = np.asarray(angles, dtype=np.float64).ravel()

    # Build furniture 2D footprint polygons
    furn_polys = _build_furn_polys(translations, sizes, ang)

    # Find best door placement
    placement, score = find_door_placement(
        fpoc, furn_polys, door_width=door_width, door_open_angle=door_open_angle
    )

    # If no placement found (all edges too short), fall back to longest edge, center
    if placement is None:
        lengths = [np.linalg.norm(fpoc[(i + 1) % n] - fpoc[i]) for i in range(n)]
        best_i = int(np.argmax(lengths))
        p0 = fpoc[best_i]
        p1 = fpoc[(best_i + 1) % n]
        edge_vec = p1 - p0
        edge_len = float(np.linalg.norm(edge_vec))
        d = edge_vec / (edge_len + 1e-9)
        n_in = np.array([-d[1], d[0]])
        wall_mid = (p0 + p1) * 0.5
        if np.dot(n_in, -wall_mid) < 0:
            n_in = -n_in
        u_hinge = (edge_len - door_width) * 0.5
        hinge_xz = p0 + d * u_hinge
        placement = (best_i, u_hinge, hinge_xz, d, n_in)

    door_edge, u_hinge, hinge_xz, d_door, n_in_door = placement
    dw = min(door_width, np.linalg.norm(fpoc[(door_edge + 1) % n] - fpoc[door_edge]) * 0.8)
    dh = min(door_height, wall_height)

    # ── Helpers ────────────────────────────────────────────────────────
    all_verts, all_faces = [], []
    vc = 0

    def add_quad(bl, br, tr_, tl):
        nonlocal vc
        all_verts.extend([list(bl), list(br), list(tr_), list(tl)])
        all_faces.extend([
            [vc, vc+1, vc+2], [vc, vc+2, vc+3],
            [vc+2, vc+1, vc], [vc+3, vc+2, vc],   # back face (double-sided)
        ])
        vc += 4

    # ── Build per-edge wall geometry ───────────────────────────────────
    for i in range(n):
        x0, z0 = fpoc[i]
        x1, z1 = fpoc[(i + 1) % n]
        edge_vec = np.array([x1 - x0, z1 - z0])
        edge_len = float(np.linalg.norm(edge_vec))

        if i != door_edge:
            add_quad([x0, 0.0, z0], [x1, 0.0, z1],
                     [x1, wall_height, z1], [x0, wall_height, z0])
            continue

        # Wall with door cutout on this edge
        d = edge_vec / (edge_len + 1e-9)
        u0 = u_hinge                # left jamb position along edge
        u1 = u_hinge + dw           # right jamb position along edge

        def at(u, y):
            return [x0 + d[0] * u, y, z0 + d[1] * u]

        if u0 > 0.01:                       # left section
            add_quad(at(0, 0), at(u0, 0), at(u0, wall_height), at(0, wall_height))
        if wall_height > dh + 0.01:         # header above door opening
            add_quad(at(u0, dh), at(u1, dh), at(u1, wall_height), at(u0, wall_height))
        if u1 < edge_len - 0.01:            # right section
            add_quad(at(u1, 0), at(edge_len, 0), at(edge_len, wall_height), at(u1, wall_height))

    # ── Assemble wall mesh ─────────────────────────────────────────────
    wv = np.array(all_verts, dtype=np.float32)
    wf = np.array(all_faces,  dtype=np.int32)
    wall_mesh = Mesh.from_faces(vertices=np.copy(wv), faces=np.copy(wf), colors=wall_color)

    tr_walls = None
    if with_trimesh:
        tr_walls = trimesh.Trimesh(vertices=np.copy(wv), faces=np.copy(wf), process=False)
        tr_walls.visual.face_colors = np.tile(
            [int(c * 255) for c in wall_color] + [255], (wf.shape[0], 1))

    # ── Door slab (half-open panel) ────────────────────────────────────
    hx, hz = float(hinge_xz[0]), float(hinge_xz[1])
    theta_open = np.deg2rad(door_open_angle)
    ca, sa = np.cos(theta_open), np.sin(theta_open)
    swing_dir = ca * d_door + sa * n_in_door
    fx, fz = float(hinge_xz[0] + dw * swing_dir[0]), float(hinge_xz[1] + dw * swing_dir[1])

    dv = np.array([[hx, 0.0, hz], [fx, 0.0, fz],
                   [fx, dh,  fz], [hx, dh,  hz]], dtype=np.float32)
    df = np.array([[0, 1, 2], [0, 2, 3],
                   [2, 1, 0], [3, 2, 0]], dtype=np.int32)
    door_mesh = Mesh.from_faces(vertices=dv, faces=df, colors=door_color)

    tr_door = None
    if with_trimesh:
        tr_door = trimesh.Trimesh(vertices=np.copy(dv), faces=np.copy(df), process=False)
        tr_door.visual.face_colors = np.tile(
            [int(c * 255) for c in door_color] + [255], (df.shape[0], 1))

    return wall_mesh, door_mesh, tr_walls, tr_door


""" Furniture rednerable """

def get_bbox_points(centroid, size, angle) -> np.ndarray:
    """Return a set of bounding box segments as a 24 by 3 numpy array."""
    R = rotation_matrix_around_y(angle)
    l_x, l_y, l_z = -size / 2
    u_x, u_y, u_z = size / 2
    bbox_points = np.array([
        (l_x, l_y, l_z), (u_x, l_y, l_z), (u_x, l_y, l_z), (u_x, u_y, l_z),
        (u_x, u_y, l_z), (l_x, u_y, l_z), (l_x, u_y, l_z), (l_x, l_y, l_z),
        (l_x, l_y, u_z), (u_x, l_y, u_z), (u_x, l_y, u_z), (u_x, u_y, u_z),
        (u_x, u_y, u_z), (l_x, u_y, u_z), (l_x, u_y, u_z), (l_x, l_y, u_z),
        (l_x, l_y, l_z), (l_x, l_y, u_z), (u_x, l_y, l_z), (u_x, l_y, u_z),
        (u_x, u_y, l_z), (u_x, u_y, u_z), (l_x, u_y, l_z), (l_x, u_y, u_z)
    ])
    return bbox_points @ R.T + centroid


def get_textured_objects_in_scene(scene: Room, colors=None, with_bbox=False, 
                                  box_color=(0.0, 1, 0.4, 1.0), width=0.05) \
    -> List[Renderable]:
    """Return the objects in a scene as a list of simple-3dviz TexturedMesh with an 
    option to add bounding box lines. 
    If "colors" is given, furniture texture will be replaced by the input list of colors."""
    if colors is not None:
        assert len(colors) == len(scene.bboxes)

    renderables = []
    bbox_renderables = []
    for i, furniture in enumerate(scene.bboxes):
        # Load the furniture and scale it as it is given in the dataset
        if colors is None:
            raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
        else:
            raw_mesh = Mesh.from_file(furniture.raw_model_path, color=colors[i])
        raw_mesh.scale(furniture.scale)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1])/2

        # Extract the predicted affine transformation to position the
        # mesh
        translation = furniture.centroid(offset=-scene.centroid)
        theta = furniture.z_angle
        R = rotation_matrix_around_y(theta)

        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R.T, t=translation)
        renderables.append(raw_mesh)

        if with_bbox:
            # Get bounding box segments
            bbox_points = get_bbox_points(translation, bbox[1] - bbox[0], theta)
            bbox_renderables.append(Lines(bbox_points, colors=box_color, width=width))
    
    return renderables + bbox_renderables


def get_textured_objects(
        bbox_params, objects_dataset, classes, retrieve_mode="size", 
        color_palette=None, with_bbox=False, box_color=(0.0, 1.0, 0.4, 1.0), 
        width=0.05, with_trimesh=True
    ) -> Tuple[List[TexturedMesh], List[trimesh.Trimesh]]:
    """Return the predicted objects as a list of simple-3dviz TexturedMesh, 
    and a list of trimesh mesh."""
    # For each one of the boxes replace them with an object
    renderables = []
    trimesh_meshes = []
    bbox_renderables = []
    for j in range(bbox_params["class_labels"].shape[0]):
        # Extract prediction of object j
        class_index = bbox_params["class_labels"][j].argmax(-1)
        query_label = classes[class_index]
        translation = bbox_params["translations"][j]
        query_size = bbox_params["sizes"][j]
        theta = bbox_params["angles"][j, 0]
        R = rotation_matrix_around_y(theta)
        
        if query_label in ["start", "end"]:
            continue

        # Retrieve 3D-FUTURE model
        if retrieve_mode == "size":
            # print(f"query_label: {query_label}, query_size: {query_size}")
            furniture = objects_dataset.get_closest_furniture_to_box(
                query_label, query_size)
        elif retrieve_mode == "objfeat":
            query_objfeat = bbox_params["objfeats"][j]
            furniture = objects_dataset.get_closest_furniture_to_objfeats(
                query_label, query_objfeat
            )
        else:
            return NotImplemented

        # Load the furniture
        if color_palette is None:
            raw_mesh = TexturedMesh.from_file(furniture.raw_model_path)
        else:
            raw_mesh = Mesh.from_file(
                furniture.raw_model_path, color=color_palette[class_index]
            )
        # Scale it 
        # as it is given in the dataset
        if retrieve_mode == "size":
            size_scale = furniture.scale
        # by predicted size
        elif retrieve_mode == "objfeat":
            raw_bbox_vertices = \
                np.load(furniture.path_to_bbox_vertices, mmap_mode="r")
            raw_sizes = np.array([  # Note: ThreedFutureModel implements size as half distance between vertices
                np.linalg.norm(raw_bbox_vertices[4] - raw_bbox_vertices[0]) / 2,
                np.linalg.norm(raw_bbox_vertices[2] - raw_bbox_vertices[0]) / 2,
                np.linalg.norm(raw_bbox_vertices[1] - raw_bbox_vertices[0]) / 2
            ])
            size_scale = query_size / raw_sizes
        else:
            return NotImplemented
        raw_mesh.scale(size_scale)

        # Compute the centroid of the vertices in order to match the
        # bbox (because the prediction only considers bboxes)
        bbox = raw_mesh.bbox
        centroid = (bbox[0] + bbox[1]) / 2

        # Apply the transformations in order to correctly position the mesh
        raw_mesh.affine_transform(t=-centroid)
        raw_mesh.affine_transform(R=R.T, t=translation)
        renderables.append(raw_mesh)

        if with_bbox:
            # Get bounding box segments
            bbox_points = get_bbox_points(translation, bbox[1] - bbox[0], theta)
            bbox_renderables.append(Lines(bbox_points, colors=box_color, width=width))
        
        if with_trimesh:
            # Create a trimesh object for the same mesh in order to save
            # everything as a single scene
            tr_mesh = trimesh.load(furniture.raw_model_path, force="mesh")
            if color_palette is None:
                tr_mesh.visual.material.image = Image.open(
                    furniture.texture_image_path
                )
            else:
                color = color_palette[class_index]
                tr_mesh.visual.face_colors = \
                    (color[None, :].repeat(tr_mesh.faces.shape[0], axis=0) \
                     .reshape(-1, 3) * 255).astype(np.uint8)
                tr_mesh.visual.vertex_colors = \
                    (color[None, :].repeat(tr_mesh.vertices.shape[0], axis=0) \
                     .reshape(-1, 3) * 255).astype(np.uint8)
            tr_mesh.vertices *= size_scale
            tr_mesh.vertices -= centroid
            tr_mesh.vertices[...] = tr_mesh.vertices.dot(R.T) + translation
            trimesh_meshes.append(tr_mesh)

    if with_trimesh:
        return renderables + bbox_renderables, trimesh_meshes
    else:
        return renderables + bbox_renderables, None


def get_edge_renderables(centroids: Union[torch.Tensor, np.ndarray], 
                         edge_index_list: List[np.ndarray], 
                         line_colors=(0.0, 1, 0.4, 1.0), line_widths=0.05, 
                         marker_color=(0.0, 1, 0.4, 1.0), marker_size=0.2):
    
    if isinstance(line_colors[0], float) or isinstance(line_colors[0], int):
        line_colors = [line_colors] * len(edge_index_list)

    if isinstance(line_widths, float) or isinstance(line_widths, int):
        line_widths = [line_widths] * len(edge_index_list)
    
    edge_lines = [
        Lines(
            centroids[edge_index.transpose().reshape(-1)], 
            colors=color, width=width
        ) for edge_index, color, width in 
        zip(edge_index_list, line_colors, line_widths) if edge_index.size != 0
    ]
    centers = Spherecloud(centroids, colors=marker_color, sizes=marker_size)

    return edge_lines + [centers]


""" Rendering """

def render_projection(scene, renderables, color, mode, frame_path=None):
    if color is not None:
        try:
            color[0][0]
        except TypeError:
            color = [color]*len(renderables)
    else:
        color = [None]*len(renderables)

    scene.clear()
    for r, c in zip(renderables, color):
        if isinstance(r, Mesh) and c is not None:
            r.mode = mode
            r.colors = c
        if isinstance(r, TexturedMesh) and r.material.ambient.ndim == 2:
            # take average if the .mtl file provides more than one vector
            r.material.ambient = r.material.ambient.mean(0)
            r.material.diffuse = r.material.diffuse.mean(0)
            r.material.specular = r.material.specular.mean(0)
        scene.add(r)
    scene.render()
    if frame_path is not None:
        save_frame(frame_path, scene.frame)

    return np.copy(scene.frame)


def export_scene(output_directory, trimesh_meshes, names=None):
    # Export each object
    if names is None:
        names = ["object_{:03d}.obj".format(i) for i in range(len(trimesh_meshes))]
    mtl_names = ["material_{:03d}".format(i) for i in range(len(trimesh_meshes))]

    for i, m in enumerate(trimesh_meshes):
        obj_out, tex_out = trimesh.exchange.obj.export_obj(m, return_texture=True)

        with open(os.path.join(output_directory, names[i]), "w") as f:
            f.write(obj_out.replace("material0", mtl_names[i]))

        # No material and texture to rename
        if tex_out is None:
            continue

        mtl_key = next(k for k in tex_out.keys() if k.endswith(".mtl"))
        path_to_mtl_file = os.path.join(output_directory, mtl_names[i]+".mtl")
        with open(path_to_mtl_file, "wb") as f:
            f.write(tex_out[mtl_key].replace(b"material0", mtl_names[i].encode("ascii")))
        tex_key = next(k for k in tex_out.keys() if not k.endswith(".mtl"))
        tex_ext = os.path.splitext(tex_key)[1]
        path_to_tex_file = os.path.join(output_directory, mtl_names[i]+tex_ext)
        with open(path_to_tex_file, "wb") as f:
            f.write(tex_out[tex_key])

    # Export scene (scene.obj, material.mtl, material_0.png)
    trimesh_combined = trimesh.util.concatenate(trimesh_meshes)
    trimesh_combined.export(os.path.join(output_directory, "scene.obj"))
