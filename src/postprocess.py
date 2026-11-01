"""Heuristic cavity carving for open containers. Assumes Y-up (see reconstruct._orient)."""
import logging, re
import numpy as np
import trimesh
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)
_CONTAINER = re.compile(r"\b(mug|cup|bowl|vase|bucket|basket|jar|pot)s?\b", re.I)

def is_container(label, prompt=None) -> bool:
    return bool(_CONTAINER.search(f"{label or ''} {prompt or ''}"))

def carve_cavity(mesh, wall=0.12, floor=0.15, probe=0.10):
    try:
        m = mesh.copy()
        m.merge_vertices(); m.fill_holes(); trimesh.repair.fix_normals(m)
        cols = m.visual.to_color().vertex_colors.copy()
        lo, hi = m.bounds
        height, top = hi[1] - lo[1], hi[1]

        sec = m.section(plane_origin=[0, top - probe * height, 0], plane_normal=[0, 1, 0])
        if sec is None:
            return mesh
        path2d, to_3d = sec.to_planar()
        if not path2d.polygons_full:
            return mesh
        poly = max(path2d.polygons_full, key=lambda p: p.area)
        x0, y0, x1, y1 = poly.bounds
        inset = poly.buffer(-wall * min(x1 - x0, y1 - y0))
        if inset.is_empty:
            return mesh
        if inset.geom_type == "MultiPolygon":
            inset = max(inset.geoms, key=lambda g: g.area)

        bottom = lo[1] + floor * height
        cav_h = (top - bottom) + 0.05 * height
        cav = trimesh.creation.extrude_polygon(inset, cav_h)
        cav.apply_transform(to_3d)
        if abs((cav.bounds[1][1] - cav.bounds[0][1]) - cav_h) > 1e-3 * height:
            return mesh                      # cutter is not vertical: up axis is wrong
        cav.apply_translation([0, bottom - cav.bounds[0][1], 0])

        out = m.difference(cav, engine="manifold")
        if out is None or len(out.faces) == 0:
            return mesh

        dist, idx = cKDTree(m.vertices).query(out.vertices)
        new_cols = cols[idx]
        inner = dist > 0.5 * m.edges_unique_length.mean()     # new cut surface
        if inner.any():
            new_cols[inner, :3] = (cols[:, :3].mean(axis=0) * 0.7).astype(np.uint8)
        out.visual = trimesh.visual.ColorVisuals(mesh=out, vertex_colors=new_cols)
        return out
    except Exception as e:
        log.warning("cavity carve skipped: %s", e)
        return mesh