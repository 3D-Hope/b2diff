from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pyrr
import trimesh
from simple_3dviz import Scene
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz.renderables import Mesh
from simple_3dviz.renderables.textured_mesh import TexturedMesh
from simple_3dviz.utils import render
from threed_front.rendering import get_textured_objects

from .floor_condition_utils import build_floor_tri_mesh_inputs


def _sanitize_material_vectors(renderables: Sequence[object]) -> None:
    for r in renderables:
        if isinstance(r, TexturedMesh) and hasattr(r, "material"):
            for name in ["ambient", "diffuse", "specular"]:
                if hasattr(r.material, name):
                    v = getattr(r.material, name)
                    if hasattr(v, "ndim") and v.ndim == 2:
                        setattr(r.material, name, v.mean(0))


def render_layouts_to_png_and_glb(
    layouts: List[Dict[str, np.ndarray]],
    floor_polygon_centered_xz: np.ndarray,
    objects_dataset,
    object_types: Sequence[str],
    output_dir: str,
    export_glb: bool = True,
    add_floor: bool = True,
    floor_color: Tuple[float, float, float] = (0.95, 0.95, 0.95),
    window_size: Tuple[int, int] = (512, 512),
    camera_position: Tuple[float, float, float] = (-12.0, 5.0, -10.0),
    camera_target: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    up_vector: Tuple[float, float, float] = (0.0, -1.0, 0.0),
    room_side: float = 3.1,
    ortho_zoom: float = 1.0,
) -> Dict[str, List[str]]:
    """Render predicted layouts to PNG and optional GLB without walls/doors."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scene = Scene(size=window_size)
    scene.light = camera_position
    scene.camera_position = camera_position
    scene.camera_target = camera_target
    scene.up_vector = up_vector
    scene.background = (1.0, 1.0, 1.0, 1.0)

    zoomed_side = room_side / max(ortho_zoom, 1e-6)
    scene.camera_matrix = pyrr.Matrix44.orthogonal_projection(
        left=-zoomed_side,
        right=zoomed_side,
        bottom=-zoomed_side,
        top=zoomed_side,
        near=0.1,
        far=1000.0,
    )

    png_paths: List[str] = []
    glb_paths: List[str] = []

    floor_renderable = None
    floor_trimesh = None
    if add_floor:
        floor_mesh = build_floor_tri_mesh_inputs(floor_polygon_centered_xz)
        floor_renderable = Mesh.from_faces(
            vertices=np.copy(floor_mesh["vertices"]),
            faces=np.copy(floor_mesh["faces"]),
            colors=floor_color,
        )
        floor_trimesh = trimesh.Trimesh(
            vertices=np.copy(floor_mesh["vertices"]),
            faces=np.copy(floor_mesh["faces"]),
            process=False,
        )

    for i, layout in enumerate(layouts):
        renderables, tr_meshes = get_textured_objects(
            layout,
            objects_dataset,
            object_types,
            retrieve_mode="object" if ("objfeats" in layout or "objfeats_32" in layout) else "size",
            color_palette=None,
            with_trimesh=export_glb,
        )
        _sanitize_material_vectors(renderables)

        if floor_renderable is not None:
            renderables.append(floor_renderable)
            if export_glb and floor_trimesh is not None:
                tr_meshes.append(floor_trimesh)

        scene.clear()
        for r in renderables:
            scene.add(r)

        png_path = out / f"{i:04d}.png"
        render(
            renderables,
            size=window_size,
            camera_position=camera_position,
            camera_target=camera_target,
            up_vector=up_vector,
            background=(1.0, 1.0, 1.0, 1.0),
            behaviours=[SaveFrames(str(png_path), 1)],
            n_frames=1,
            scene=scene,
        )
        png_paths.append(str(png_path))

        if export_glb and len(tr_meshes) > 0:
            glb_path = out / f"{i:04d}.glb"
            tr_scene = trimesh.scene.scene.Scene()
            for m_idx, mesh in enumerate(tr_meshes):
                tr_scene.add_geometry(mesh, node_name=f"object_{m_idx:03d}")
            tr_scene.export(str(glb_path))
            glb_paths.append(str(glb_path))

    return {
        "png_paths": png_paths,
        "glb_paths": glb_paths,
    }
