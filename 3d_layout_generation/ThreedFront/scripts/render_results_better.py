"""Script to generate synthesized orthographic projection images from result pickle file.
These images are needed to evaluate the model in terms of FID scores and real/fake classification accuracy.
"""
import argparse
import numpy as np
import os
import sys
import seaborn as sns
import pickle
from tqdm import tqdm
import pyrr
import trimesh

from threed_front.datasets import ThreedFutureDataset
# Remove scene_from_args import, add direct Scene import
from simple_3dviz import Scene
from simple_3dviz.utils import render, save_frame
from simple_3dviz.behaviours.io import SaveFrames
from threed_front.evaluation import ThreedFrontResults
# Remove ORTHOGRAPHIC_PROJECTION_SCENE import
from threed_front.rendering import get_textured_objects, get_floor_plan, render_projection, build_walls_with_door
from utils import PATH_TO_PICKLED_3D_FUTURE_MODEL, PATH_TO_FLOOR_PLAN_TEXTURES


def main(argv):
    parser = argparse.ArgumentParser(
        description="Generate synthetic layout images from predicted results"
    )        
    parser.add_argument(
        "result_file",
        help="Path to a pickled result (ThreedFrontResults object)"
    )
    parser.add_argument(
        "--output_directory",
        help="Path to the output directory (default: result_file directory)"
    )
    parser.add_argument(
        "--path_to_pickled_3d_future_model",
        default=PATH_TO_PICKLED_3D_FUTURE_MODEL,
        help="Path to pickled 3d future model"
        "(default: output/threed_future_model_<room_type>.pkl)"
    )
    parser.add_argument(
        "--retrieve_by_size",
        action="store_true",
        help="Ignore objfeat and use size to retrieve most similar 3D-FUTURE models "
        "(default: use objfeat instead of sizes if available)"
    )
    parser.add_argument(
        "--no_texture",
        action="store_true",
        help="Color objects by semantic label, and set floor plan to white"
    )
    parser.add_argument(
        "--without_floor",
        action="store_true",
        help="Remove the floor plane (will be set to True if the model is not trained with floor plans)"
    )
    parser.add_argument(
        "--floor_color",
        type=lambda x: tuple(map(float, x.split(","))) if x!= None else None,
        help="Set floor color of generated images (and override path_to_floor_plan_textures)"
    )
    parser.add_argument(
        "--path_to_floor_plan_textures",
        default=PATH_TO_FLOOR_PLAN_TEXTURES,
        help="Path to floor texture image directory or a single image file "
        "(default: demo/floor_plan_texture_images)"
    )
    parser.add_argument(
        "--room_side",
        type=float,
        default=None,
        help="The size of the room along a side "
        "(default:3.1 for bedroom and library, 6.1 for diningroom and livingroom)"
    )
    parser.add_argument(
        "--window_size",
        type=lambda x: tuple(map(int, x.split(","))),
        default="512,512",
        help="Define the size of the scene and the window (default: 512,512)"
    )
    # parser.add_argument(
    #     "--camera_position",
    #     type=lambda x: tuple(map(float, x.split(","))),
    #     default="-1.0,3.0,2.0",
    #     help="Camera position in the scene (default: 2.0,0.2,2.0)"
    # ) # TODO: USE THIS default
    parser.add_argument(
        "--camera_position",
        type=lambda x: tuple(map(float, x.split(","))),
        default="-12.0,5.0,-10.0",
        help="Camera position in the scene. X=right(+)/left(-), Y=up, Z=depth. (default: -12.0,5.0,-10.0 = left side view)"
    )
    parser.add_argument(
        "--camera_target",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,0,0",
        help="Set the target for the camera (default: 0,0,0)"
    )
    parser.add_argument(
        "--up_vector",
        type=lambda x: tuple(map(float, x.split(","))),
        default="0,-1,0",
        help="Up vector of the scene (default: 0,-1,0)"
    )
    parser.add_argument(
        "--background",
        type=lambda x: list(map(float, x.split(","))),
        default="1,1,1,1",
        help="Set the background of the scene (default: 1,1,1,1)"
    )
    parser.add_argument(
        "--with_orthographic_projection",
        action="store_true",
        default=True,
        help="Use orthographic projection (default: True)"
    )
    parser.add_argument(
        "--ortho_zoom",
        type=float,
        default=1.0,
        help="Zoom factor for orthographic projection (default: 1.0, >1.0 zooms in, <1.0 zooms out)"
    )
    parser.add_argument(
        "--without_walls",
        action="store_true",
        help="Do not render walls"
    )
    parser.add_argument(
        "--without_door",
        action="store_true",
        help="Do not render the door"
    )
    parser.add_argument(
        "--export_glb",
        action="store_true",
        help="Also export each scene as a GLB file alongside the PNG"
    )

    args = parser.parse_args(argv)

    # Load results
    with open(args.result_file, "rb") as f:
        threed_front_results = pickle.load(f)
    assert isinstance(threed_front_results, ThreedFrontResults)
    room_type = next((
        type for type in ["diningroom", "livingroom", "bedroom", "library"] \
        if type in os.path.basename(threed_front_results.config["data"]["dataset_directory"])
        ), None)
    assert room_type is not None
    print("Room type:", room_type)
    if not threed_front_results.config["network"].get("room_mask_condition", True):
        args.without_floor = True

    # Default output directory
    if args.output_directory is None:
        args.output_directory = os.path.dirname(args.result_file)
    print("Saving rendered results to: {}.".format(args.output_directory))

    # Output paths
    path_to_image = os.path.join(args.output_directory, "{:04d}_{}.png")

    # Check if output directory exists and if it doesn't create it
    if os.path.exists(args.output_directory) and \
        len([fi for fi in os.listdir(args.output_directory) if fi.endswith(".png")]) > 0:
        input("{} contain png files. Press any key to remove all png files..." \
              .format(args.output_directory))
        for fi in os.listdir(args.output_directory):
            if fi.endswith(".png"):
                os.remove(os.path.join(args.output_directory, fi))
    else:
        os.makedirs(args.output_directory, exist_ok=True)

    # Build the dataset of 3D models
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(
        args.path_to_pickled_3d_future_model.format(room_type)
    )
    print("Loaded {} 3D-FUTURE models from: {}.".format(
        len(objects_dataset), args.path_to_pickled_3d_future_model.format(room_type)
    ))
    
    # Set floor texture or color
    if args.without_floor:
        args.floor_color = None
        floor_textures = [None]
    elif args.no_texture:
        # set floor to specified color if given, or white
        if args.floor_color is None:
            args.floor_color = (1, 1, 1)
        floor_textures = [None]
    else:
        # set floor to specified color if given, or sampled textures
        if args.floor_color is None:
            floor_textures = \
                [os.path.join(args.path_to_floor_plan_textures, fi)
                    for fi in os.listdir(args.path_to_floor_plan_textures)]
        else:
            floor_textures = [None]
    
    # Set color palette if args.no_texture
    if args.no_texture:
        color_palette = \
            sns.color_palette('hls', threed_front_results.test_dataset.n_object_types)
    else:
        color_palette = None
    
    # Determine room side
    if args.room_side is None:
        room_side = 3.1 if room_type in ["bedroom", "library"] else 6.1
    else:
        room_side = args.room_side
    print("Room side:", room_side)
    
    scene = Scene(size=args.window_size)
    scene.light = args.camera_position
    scene.camera_position = args.camera_position
    scene.camera_target = args.camera_target
    scene.up_vector = args.up_vector
    scene.background = args.background
    
    if args.with_orthographic_projection:
        zoomed_side = room_side / args.ortho_zoom
        scene.camera_matrix = pyrr.Matrix44.orthogonal_projection(
            left=-zoomed_side, right=zoomed_side, 
            bottom=-zoomed_side, top=zoomed_side, 
            near=0.1, far=1000
        )
    
    
    # Render projection images
    for i in tqdm(range(len(threed_front_results))):
        if i not in [298, 0, 2, 13, 46, 55, 56, 62, 124, 147, 184, 224, 334, 413, 439, 729, 727]: continue # TODO: rm 
        scene_idx = threed_front_results[i][0]
        image_path = path_to_image.format(
            i, threed_front_results.test_dataset[scene_idx].scene_id
        )
        
        # Get the predicted layout
        predicted_layout = threed_front_results._predicted_layouts[i]
        room = threed_front_results.test_dataset[scene_idx]
        
        # Determine retrieve mode
        retrieve_mode = "size" if args.retrieve_by_size else None
        if retrieve_mode is None:
            retrieve_mode = "object" if \
                "objfeats" in predicted_layout.keys() or "objfeats_32" in predicted_layout.keys() else "size"
        
        # Get object renderables
        renderables, trimesh_meshes = get_textured_objects(
            predicted_layout, objects_dataset, 
            threed_front_results.test_dataset.object_types,
            retrieve_mode=retrieve_mode, 
            color_palette=color_palette, 
            with_trimesh=args.export_glb
        )
        
        from simple_3dviz.renderables.textured_mesh import TexturedMesh
        from simple_3dviz import Mesh
        for r in renderables:
            if isinstance(r, TexturedMesh) and hasattr(r, 'material'):
                if hasattr(r.material, 'ambient') and r.material.ambient.ndim == 2:
                    # take average if the .mtl file provides more than one vector
                    r.material.ambient = r.material.ambient.mean(0)
                if hasattr(r.material, 'diffuse') and r.material.diffuse.ndim == 2:
                    r.material.diffuse = r.material.diffuse.mean(0)
                if hasattr(r.material, 'specular') and r.material.specular.ndim == 2:
                    r.material.specular = r.material.specular.mean(0)
        
        # Add floor plan if needed
        if not (args.without_floor or (np.random.choice(floor_textures) is None and args.floor_color is None)):
            floor_texture = np.random.choice(floor_textures)
            floor_plan, tr_floor, _ = get_floor_plan(
                room, 
                floor_texture, 
                args.floor_color, 
                with_trimesh=args.export_glb
            )
            renderables.append(floor_plan)
            if args.export_glb and tr_floor is not None:
                trimesh_meshes.append(tr_floor)

        # ── Walls + Door ────────────────────────────────────────────────────
        # Load ordered floor-plan corners (XZ, centroid-centered) from boxes.npz
        boxes_path = threed_front_results.test_dataset._path_to_room(scene_idx)
        D_boxes = np.load(boxes_path)
        if "floor_plan_ordered_corners" in D_boxes:
            fpoc = D_boxes["floor_plan_ordered_corners"]  # (N, 2)
        else:
            # Fallback: convex hull of unique XZ floor vertices
            from scipy.spatial import ConvexHull
            verts_xz = np.unique(
                (room.floor_plan_vertices - room.floor_plan_centroid)[:, [0, 2]], axis=0
            )
            hull = ConvexHull(verts_xz)
            fpoc = verts_xz[hull.vertices]

        # Determine wall height: top of ceiling/pendant lamp objects, else 2.5 m
        ceiling_labels = {"ceiling_lamp", "pendant_lamp"}
        object_types = threed_front_results.test_dataset.object_types
        ceiling_indices = {
            idx for idx, lbl in enumerate(object_types) if lbl in ceiling_labels
        }
        special_labels = {"start", "end"}
        skip_indices = ceiling_indices | {
            idx for idx, lbl in enumerate(object_types) if lbl in special_labels
        }
        wall_height = 2.5
        ground_translations = []
        ground_sizes = []
        ground_angles = []
        for j in range(predicted_layout["class_labels"].shape[0]):
            cidx = int(predicted_layout["class_labels"][j].argmax(-1))
            if cidx in ceiling_indices:
                top_y = float(
                    predicted_layout["translations"][j, 1]
                    + predicted_layout["sizes"][j, 1]
                )
                if top_y > wall_height:
                    wall_height = top_y
            if cidx not in skip_indices:
                ground_translations.append(predicted_layout["translations"][j])
                ground_sizes.append(predicted_layout["sizes"][j])
                ground_angles.append(predicted_layout["angles"][j, 0])
        furn_trans_for_door  = np.array(ground_translations) if ground_translations else None
        furn_sizes_for_door  = np.array(ground_sizes)        if ground_sizes        else None
        furn_angles_for_door = np.array(ground_angles)       if ground_angles       else None

        if not args.without_walls:
            wall_mesh, door_mesh, tr_walls, tr_door = build_walls_with_door(
                fpoc, wall_height,
                translations=furn_trans_for_door,
                sizes=furn_sizes_for_door,
                angles=furn_angles_for_door,
                wall_color=(0.94, 0.92, 0.88),
                with_trimesh=args.export_glb
            )
            renderables.append(wall_mesh)
            if door_mesh is not None and not args.without_door:
                renderables.append(door_mesh)
            if args.export_glb:
                if tr_walls is not None:
                    trimesh_meshes.append(tr_walls)
                if tr_door is not None and not args.without_door:
                    trimesh_meshes.append(tr_door)
        # ───────────────────────────────────────────────────────────────────

        scene.clear()
        for r in renderables:
            scene.add(r)
        
        # Render using simple_3dviz.utils.render similar to render_threedfront_scene.py
        behaviours = [SaveFrames(image_path, 1)]
        render(
            renderables,
            size=args.window_size,
            camera_position=args.camera_position,
            camera_target=args.camera_target,
            up_vector=args.up_vector,
            background=args.background,
            behaviours=behaviours,
            n_frames=1,
            scene=scene
        )

        # Export GLB if requested
        if args.export_glb and trimesh_meshes:
            glb_path = image_path.replace(".png", ".glb")
            tr_scene = trimesh.scene.scene.Scene()
            for j, mesh in enumerate(trimesh_meshes):
                tr_scene.add_geometry(mesh, node_name="object_{:03d}".format(j))
            tr_scene.export(glb_path)
            print("Saved GLB to {}".format(glb_path))


if __name__ == "__main__":
    main(sys.argv[1:])