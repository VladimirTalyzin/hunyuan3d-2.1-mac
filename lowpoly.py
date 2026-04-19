"""Low-poly optimization algorithms for Hunyuan3D output.

Three strategies:
  * quadric  — universal quadric edge collapse (good for organic shapes)
  * planar   — planar-region remesh (good for CAD / hard-surface: doors,
               boxes, furniture). Detects large coplanar regions, simplifies
               their boundary, re-triangulates.
  * hybrid   — planar first, then quadric on the result (recovers organic
               details that survived planar)
  * auto     — dispatches to planar+quadric for hard-surface meshes and to
               plain quadric for organic meshes, based on the fraction of
               surface area covered by large planar regions.

Shared levels (weak / medium / strong) control the aggressiveness.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import trimesh


# ==== Level presets =========================================================
# Mirror the same names the Gradio UI uses so the two stay aligned.

@dataclass(frozen=True)
class LevelPreset:
    # Quadric
    quadric_ratio: float         # target_faces = max(min, round(n_faces * ratio))
    quadric_min_faces: int
    # Planar — initial BFS clustering (strict, edge-local)
    coplanar_deg: float          # max face-adjacency angle to stay in one cluster
    coplanar_dist_frac: float    # max vertex distance to region plane, in units of bbox_diag
    # Planar — cluster merge pass (loose, global plane fit)
    merge_deg: float             # allowed angle between two cluster planes to merge them
    merge_dist_frac: float       # allowed distance (as fraction of bbox_diag) between merged cluster planes
    # Planar — boundary simplification & filtering
    boundary_simplify_frac: float  # Douglas-Peucker tolerance in units of bbox_diag
    min_region_faces: int        # regions smaller than this are left untouched
    min_region_area_frac: float  # regions covering less than this fraction of total area are skipped

LEVELS: dict[str, LevelPreset] = {
    # BFS tol is tight enough to preserve ≥45° feature edges cleanly;
    # the merge pass absorbs MC stair-step noise within each logical plane.
    "weak":   LevelPreset(0.30, 3000, 3.0, 0.0015,  8.0, 0.0030, 0.0015, 40, 0.0050),
    "medium": LevelPreset(0.08, 1500, 5.0, 0.0030, 12.0, 0.0050, 0.0050, 20, 0.0020),
    "strong": LevelPreset(0.02,  400, 8.0, 0.0060, 18.0, 0.0120, 0.0120, 10, 0.0010),
}

# Target edge length (as fraction of bbox diagonal) for isotropic remeshing per level.
ISO_TARGET_LEN_FRAC = {"weak": 0.010, "medium": 0.025, "strong": 0.050}
ISO_FEATURE_DEG = 30.0  # corners sharper than this are preserved as feature edges


# ==== Shared pymeshlab helpers (mirror gradio_app.py) =======================

def _preclean_pymeshlab(ms, aggressive: bool = True) -> None:
    """Basic cleanup.

    `aggressive=True` also runs non-manifold repair — needed on raw marching
    cubes output but destructive on meshes we just assembled from planar
    clusters (the seam edges look "non-manifold" to pymeshlab and get split,
    creating visible holes).
    """
    safe_steps = (
        "meshing_remove_duplicate_vertices",
        "meshing_remove_duplicate_faces",
        "meshing_remove_unreferenced_vertices",
        "meshing_remove_null_faces",
        "meshing_remove_t_vertices",
    )
    aggressive_steps = (
        "meshing_repair_non_manifold_edges",
        "meshing_repair_non_manifold_vertices",
    )
    steps = safe_steps + (aggressive_steps if aggressive else ())
    for step in steps:
        if hasattr(ms, step):
            try:
                getattr(ms, step)()
            except Exception:
                pass


def _taubin_smooth(ms, steps: int = 2) -> None:
    if hasattr(ms, "apply_coord_taubin_smoothing"):
        try:
            ms.apply_coord_taubin_smoothing(stepsmoothnum=int(steps))
            return
        except Exception:
            pass
    try:
        ms.apply_filter("apply_coord_taubin_smoothing", stepsmoothnum=int(steps))
    except Exception:
        pass


# ==== Quadric decimation ====================================================

def decimate_quadric(
    mesh: trimesh.Trimesh,
    target_faces: int,
    preserve_boundary: bool = True,
    feature_smooth: bool = True,
) -> trimesh.Trimesh:
    """Quadric edge collapse via pymeshlab, trimesh fallback. Same logic as
    the current gradio_app.decimate_mesh (kept in one place)."""
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        pm = pymeshlab.Mesh(
            vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
            face_matrix=np.asarray(mesh.faces, dtype=np.int32),
        )
        ms.add_mesh(pm, "orig")
        # Only run the destructive non-manifold repairs when the input
        # actually needs them — raw MC output does, planar/iso output doesn't.
        _preclean_pymeshlab(ms, aggressive=not mesh.is_watertight)

        kw = dict(
            targetfacenum=int(target_faces),
            preserveboundary=preserve_boundary,
            preservenormal=True,
            preservetopology=True,
            optimalplacement=True,
            planarquadric=True,
            planarweight=0.002,
            qualitythr=0.3,
            qualityweight=False,
            autoclean=True,
        )
        if preserve_boundary:
            kw["boundaryweight"] = 1.5

        tried = False
        for name in (
            "meshing_decimation_quadric_edge_collapse",
            "simplification_quadric_edge_collapse_decimation",
        ):
            if hasattr(ms, name):
                try:
                    getattr(ms, name)(**kw)
                    tried = True
                    break
                except TypeError:
                    try:
                        getattr(ms, name)(
                            targetfacenum=int(target_faces),
                            preserveboundary=preserve_boundary,
                            preservenormal=True,
                            preservetopology=True,
                            optimalplacement=True,
                            planarquadric=True,
                            autoclean=True,
                        )
                        tried = True
                        break
                    except Exception:
                        continue
        if not tried:
            ms.apply_filter("meshing_decimation_quadric_edge_collapse", **kw)

        if feature_smooth:
            _taubin_smooth(ms, steps=2)
        # Skip the post-decimation preclean: `autoclean=True` inside the
        # quadric filter already handles duplicates/null faces, and running
        # `meshing_remove_duplicate_faces` after it sometimes removes faces
        # that quadric intentionally keeps at cluster boundaries (when the
        # input came from planar remesh), which opens new holes.

        out = ms.current_mesh()
        new_mesh = trimesh.Trimesh(
            vertices=out.vertex_matrix(),
            faces=out.face_matrix(),
            process=False,
        )
        if len(new_mesh.faces) < max(50, int(target_faces * 0.1)):
            raise RuntimeError("too few faces after decimation")
        return new_mesh
    except Exception as e:
        print(f"[quadric] pymeshlab failed ({e}); trimesh fallback", flush=True)
        try:
            return mesh.simplify_quadric_decimation(int(target_faces))
        except Exception as e2:
            print(f"[quadric] trimesh fallback failed: {e2}", flush=True)
            return mesh


# ==== Planar region detection ==============================================

def _cluster_coplanar_faces(
    mesh: trimesh.Trimesh,
    tol_deg: float,
    dist_tol: float,
) -> list[np.ndarray]:
    """Region-growing clustering of connected coplanar faces.

    For every candidate face we check BOTH the angle to the *current region
    plane* (re-fit on the fly) and the distance of its vertices to that
    plane. That catches plane drift on long faces which a pure local-normal
    BFS would not.
    """
    n = len(mesh.faces)
    if n == 0:
        return []

    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_centroids = np.asarray(mesh.triangles_center, dtype=np.float64)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)

    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)  # (M, 2)
    adj_angles = np.asarray(mesh.face_adjacency_angles, dtype=np.float64)

    tol_rad = math.radians(tol_deg)
    neighbors: list[list[int]] = [[] for _ in range(n)]
    for i, (a, b) in enumerate(adj):
        if adj_angles[i] <= tol_rad:
            neighbors[a].append(b)
            neighbors[b].append(a)

    visited = np.zeros(n, dtype=bool)
    clusters: list[np.ndarray] = []

    # Seed in descending area order — makes large flat faces absorb noise first
    seed_order = np.argsort(-face_areas)

    for seed in seed_order:
        if visited[seed]:
            continue
        plane_normal = face_normals[seed].copy()
        plane_point = face_centroids[seed].copy()
        cluster: list[int] = [int(seed)]
        visited[seed] = True
        stack = [int(seed)]
        refit_every = 25
        since_refit = 0

        while stack:
            f = stack.pop()
            for g in neighbors[f]:
                if visited[g]:
                    continue
                # Angle from *region* normal
                cos = float(np.dot(face_normals[g], plane_normal))
                cos = max(-1.0, min(1.0, cos))
                if math.acos(cos) > tol_rad:
                    continue
                # Max vertex distance to region plane
                v = vertices[faces[g]]  # (3,3)
                d = np.abs((v - plane_point) @ plane_normal)
                if d.max() > dist_tol:
                    continue
                visited[g] = True
                cluster.append(g)
                stack.append(g)
                since_refit += 1
                if since_refit >= refit_every:
                    # Re-fit plane via area-weighted PCA on cluster centroids
                    c_idx = np.asarray(cluster)
                    w = face_areas[c_idx]
                    pts = face_centroids[c_idx]
                    plane_point = (pts * w[:, None]).sum(0) / w.sum()
                    # Covariance for plane normal
                    diffs = pts - plane_point
                    cov = (diffs[:, :, None] * diffs[:, None, :] * w[:, None, None]).sum(0)
                    _, _, vh = np.linalg.svd(cov)
                    new_n = vh[-1]
                    # Keep consistent orientation with seed
                    if np.dot(new_n, plane_normal) < 0.0:
                        new_n = -new_n
                    plane_normal = new_n / (np.linalg.norm(new_n) + 1e-20)
                    since_refit = 0

        clusters.append(np.asarray(cluster, dtype=np.int64))

    return clusters


def _merge_adjacent_coplanar_clusters(
    mesh: trimesh.Trimesh,
    clusters: list[np.ndarray],
    merge_tol_deg: float,
    merge_dist_tol: float,
    min_merge_face_count: int = 50,
) -> list[np.ndarray]:
    """Greedy merge pass: two adjacent clusters unite if their *combined*
    vertex set still fits a single plane within tolerance.

    Needed because marching-cubes meshes have local ridges of 5-15° that
    split one topological face into several BFS clusters, even though the
    union is globally planar.

    We only consider clusters with ≥ min_merge_face_count — small clusters
    are noise we'd rather leave for the size filter to discard.
    """
    n_clusters = len(clusters)
    if n_clusters < 2:
        return clusters

    face_to_cluster = np.full(len(mesh.faces), -1, dtype=np.int64)
    for i, c in enumerate(clusters):
        face_to_cluster[c] = i

    # Build cluster-adjacency graph from ALL face adjacencies (ignore angles).
    # Two clusters are connected if any shared edge exists between their faces.
    adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
    cpairs = np.stack([face_to_cluster[adj[:, 0]], face_to_cluster[adj[:, 1]]], 1)
    mask = cpairs[:, 0] != cpairs[:, 1]
    cpairs = cpairs[mask]
    cluster_adj: dict[int, set[int]] = {i: set() for i in range(n_clusters)}
    for a, b in cpairs:
        cluster_adj[int(a)].add(int(b))
        cluster_adj[int(b)].add(int(a))

    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
    face_centroids = np.asarray(mesh.triangles_center, dtype=np.float64)

    # Per-cluster plane representation: (area-weighted centroid, area-weighted normal).
    def _plane_of(idx: int) -> tuple[np.ndarray, np.ndarray, float]:
        c = clusters[idx]
        w = face_areas[c]
        total_w = float(w.sum())
        centroid = (face_centroids[c] * w[:, None]).sum(0) / max(total_w, 1e-20)
        # PCA on centroids for the normal (robust to noise), fallback to mean normal
        try:
            diffs = face_centroids[c] - centroid
            cov = (diffs[:, :, None] * diffs[:, None, :] * w[:, None, None]).sum(0)
            _, s, vh = np.linalg.svd(cov)
            normal = vh[-1]
            # Keep consistent with mean normal direction
            mean_n = (face_normals[c] * w[:, None]).sum(0)
            if np.dot(normal, mean_n) < 0:
                normal = -normal
            normal /= (np.linalg.norm(normal) + 1e-20)
        except Exception:
            mean_n = (face_normals[c] * w[:, None]).sum(0)
            normal = mean_n / (np.linalg.norm(mean_n) + 1e-20)
        return centroid, normal, total_w

    planes: list[Optional[tuple[np.ndarray, np.ndarray, float]]] = [None] * n_clusters
    for i in range(n_clusters):
        if len(clusters[i]) >= min_merge_face_count:
            planes[i] = _plane_of(i)

    # Union-find
    parent = list(range(n_clusters))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merge_tol_rad = math.radians(merge_tol_deg)

    # Queue of candidate pairs: all cluster-adjacent pairs where both sides
    # pass the min-size filter. Sort by combined size descending so big
    # regions absorb neighbors first.
    candidates: list[tuple[int, int]] = []
    for a, neigh in cluster_adj.items():
        if planes[a] is None:
            continue
        for b in neigh:
            if b > a and planes[b] is not None:
                candidates.append((a, b))
    candidates.sort(key=lambda t: -(len(clusters[t[0]]) + len(clusters[t[1]])))

    merged_any = True
    passes = 0
    while merged_any and passes < 5:
        merged_any = False
        passes += 1
        for a, b in candidates:
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            pa, pb = planes[ra], planes[rb]
            if pa is None or pb is None:
                continue
            ca, na, wa = pa
            cb, nb, wb = pb
            # Angle
            cos = float(np.dot(na, nb))
            cos = max(-1.0, min(1.0, cos))
            if math.acos(cos) > merge_tol_rad:
                continue
            # Distance between planes (project each centroid onto the other)
            d_ab = abs(float(np.dot(cb - ca, na)))
            d_ba = abs(float(np.dot(ca - cb, nb)))
            if max(d_ab, d_ba) > merge_dist_tol:
                continue
            # Merge ra <- rb (pick the heavier as the winner)
            if wa >= wb:
                keep, drop = ra, rb
            else:
                keep, drop = rb, ra
            parent[drop] = keep
            # Recompute plane for the merged cluster.
            merged_faces = np.concatenate([clusters[keep], clusters[drop]])
            clusters[keep] = merged_faces
            clusters[drop] = np.empty(0, dtype=np.int64)
            planes[keep] = _plane_of(keep)
            planes[drop] = None
            merged_any = True

    # Collect distinct surviving clusters
    out: list[np.ndarray] = []
    for i in range(n_clusters):
        if find(i) == i and len(clusters[i]) > 0:
            out.append(clusters[i])
    return out


# ==== Boundary extraction & 2D projection ==================================

def _extract_boundary_edges(faces: np.ndarray) -> np.ndarray:
    """Given (F, 3) faces of one region, return edges appearing exactly
    once (= region boundary). Each edge is returned as a sorted pair."""
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    return uniq[counts == 1]


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis in the plane perpendicular to `normal`."""
    n = normal / (np.linalg.norm(normal) + 1e-20)
    # Pick the world axis least aligned with n to avoid near-parallel cross product
    ref = np.eye(3)[int(np.argmin(np.abs(n)))]
    u = np.cross(n, ref)
    u /= (np.linalg.norm(u) + 1e-20)
    v = np.cross(n, u)
    return u, v


def _trace_polygons_2d(
    edges: np.ndarray,
    points_2d: np.ndarray,
) -> list[np.ndarray]:
    """Walk boundary edges into closed loops. Returns list of vertex-index
    sequences (each is a closed ring, start == end implied)."""
    # Build adjacency: vertex -> list of neighbor vertex indices via boundary edges
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    used = set()  # undirected edges, as frozenset pairs
    rings: list[np.ndarray] = []

    for start in list(adj.keys()):
        while True:
            # Find an unused edge from this vertex
            next_v = None
            for nb in adj[start]:
                key = frozenset((start, nb))
                if key not in used:
                    next_v = nb
                    break
            if next_v is None:
                break

            ring = [start]
            cur, prev = next_v, start
            used.add(frozenset((prev, cur)))
            ring.append(cur)

            safety = len(edges) + 10
            while cur != start and safety > 0:
                candidates = [nb for nb in adj[cur]
                              if nb != prev and frozenset((cur, nb)) not in used]
                if not candidates:
                    # Dead end — non-manifold boundary. Abort this ring.
                    ring = None
                    break
                if len(candidates) == 1:
                    nxt = candidates[0]
                else:
                    # Pick the sharpest left turn in 2D — keeps the traversal on
                    # the outside of the region for outer rings and on the
                    # outside of holes for hole rings, consistent with how
                    # shapely expects orientation.
                    p_prev = points_2d[prev]
                    p_cur = points_2d[cur]
                    in_dir = p_cur - p_prev
                    in_ang = math.atan2(in_dir[1], in_dir[0])
                    best = None
                    best_turn = -1e9  # most positive (left) turn wins
                    for nb in candidates:
                        out = points_2d[nb] - p_cur
                        ang = math.atan2(out[1], out[0])
                        turn = ang - in_ang
                        # Normalize to (-pi, pi]
                        while turn <= -math.pi: turn += 2 * math.pi
                        while turn > math.pi:   turn -= 2 * math.pi
                        if turn > best_turn:
                            best_turn = turn
                            best = nb
                    nxt = best
                used.add(frozenset((cur, nxt)))
                prev, cur = cur, nxt
                ring.append(cur)
                safety -= 1

            if ring is None or cur != start:
                continue  # unclosed ring — skip
            rings.append(np.asarray(ring[:-1], dtype=np.int64))

    return rings


def _dp_simplify_chain(
    coords: np.ndarray,
    tol: float,
) -> list[int]:
    """Douglas-Peucker on a polyline (any dim), returning indices of KEPT
    points. Endpoints (first and last) are always kept.
    coords: (N, D) float — D can be 2 or 3 (or any)."""
    n = coords.shape[0]
    if n <= 2:
        return list(range(n))
    keep = [False] * n
    keep[0] = True
    keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 < 2:
            continue
        a = coords[i0]
        b = coords[i1]
        ab = b - a
        ab_len = float(np.linalg.norm(ab))
        seg = coords[i0 + 1:i1] - a
        if ab_len < 1e-20:
            d = np.linalg.norm(seg, axis=1)
        else:
            ab_hat = ab / ab_len
            proj = seg @ ab_hat
            perp_sq = (seg ** 2).sum(1) - proj ** 2
            perp_sq = np.maximum(perp_sq, 0.0)
            d = np.sqrt(perp_sq)
        if d.size == 0:
            continue
        j_local = int(np.argmax(d))
        if d[j_local] > tol:
            j = i0 + 1 + j_local
            keep[j] = True
            stack.append((i0, j))
            stack.append((j, i1))
    return [i for i, k in enumerate(keep) if k]


def _compute_boundary_chain_must_keep(
    mesh: trimesh.Trimesh,
    face_group: np.ndarray,
    initial_must_keep: set[int],
    simplify_tol: float,
) -> set[int]:
    """For every chain of boundary edges between two accepted groups, run
    Douglas-Peucker in 3D once and return the intermediate vertices that
    must be preserved in both clusters' 2D simplifications.

    Without this step the two sides of a shared boundary make independent
    decisions on which intermediate vertices to drop — and when they
    disagree, the resulting mesh has T-junctions (visible holes).

    `face_group[i] ≥ 0` → accepted cluster id; `face_group[i] == -1` →
    rejected / tiny-region face. Chains involving -1 on one side ARE still
    boundaries but are not simplified here (we preserve them verbatim on
    the accepted-cluster side via `initial_must_keep`, which already
    includes every vertex used by rejected faces).
    """
    from collections import defaultdict

    faces = np.asarray(mesh.faces, dtype=np.int64)

    # Each edge -> set of accepted-group ids touching it.
    edge_groups: dict[tuple[int, int], set[int]] = defaultdict(set)
    for f_idx in range(len(faces)):
        g = int(face_group[f_idx])
        if g < 0:
            continue  # rejected faces stay as-is; their edges are handled
                      # by initial_must_keep which already includes their vertices
        tri = faces[f_idx]
        for a_, b_ in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (int(min(a_, b_)), int(max(a_, b_)))
            edge_groups[key].add(g)

    # Boundary edges between two accepted clusters: exactly 2 distinct groups.
    boundary_edges: dict[tuple[int, int], frozenset[int]] = {
        e: frozenset(gs) for e, gs in edge_groups.items() if len(gs) >= 2
    }
    if not boundary_edges:
        return set()

    # Per-vertex adjacency inside each group-pair: vertex -> list of
    # (other_endpoint, edge_tuple) for each group-pair.
    vertex_nbrs: dict[int, dict[frozenset, list[tuple[int, tuple[int, int]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for e, gp in boundary_edges.items():
        vertex_nbrs[e[0]][gp].append((e[1], e))
        vertex_nbrs[e[1]][gp].append((e[0], e))

    # A vertex is an "anchor" if (a) it's already forced-keep, or
    # (b) it participates in edges of more than one group-pair, or
    # (c) within a single group-pair it has degree ≠ 2 (branch point / dead-end).
    anchors: set[int] = set(initial_must_keep)
    for v, gp_map in vertex_nbrs.items():
        if len(gp_map) > 1:
            anchors.add(v)
            continue
        # Single group-pair → must have degree exactly 2 to be a through-point
        (only_gp, nbrs), = gp_map.items()
        if len(nbrs) != 2:
            anchors.add(v)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    visited_edges: set[tuple[int, int]] = set()
    extra_must_keep: set[int] = set()

    def _walk_chain(start: int, first_edge: tuple[int, int],
                    gp: frozenset) -> list[int]:
        """Walk a chain from anchor `start` along boundary edges of the
        same group-pair until hitting another anchor or dead end."""
        visited_edges.add(first_edge)
        chain = [start]
        nxt = first_edge[0] if first_edge[1] == start else first_edge[1]
        prev_edge = first_edge
        safety = len(boundary_edges) + 10
        while safety > 0:
            chain.append(nxt)
            if nxt in anchors:
                return chain
            # Find the other edge of the same group-pair at `nxt`
            nbrs = vertex_nbrs[nxt][gp]
            next_edge = None
            for (_other, e) in nbrs:
                if e == prev_edge or e in visited_edges:
                    continue
                next_edge = e
                break
            if next_edge is None:
                return chain  # dead end
            visited_edges.add(next_edge)
            prev_edge = next_edge
            nxt = next_edge[0] if next_edge[1] == nxt else next_edge[1]
            safety -= 1
        return chain

    # Walk every chain starting from each anchor along every unused edge.
    for v_anchor in list(anchors):
        gp_map = vertex_nbrs.get(v_anchor, {})
        for gp, nbrs in list(gp_map.items()):
            for (_other, e) in nbrs:
                if e in visited_edges:
                    continue
                chain = _walk_chain(v_anchor, e, gp)
                if len(chain) < 3:
                    continue
                coords = verts[chain]
                kept_local = _dp_simplify_chain(coords, simplify_tol)
                for li in kept_local:
                    if 0 < li < len(chain) - 1:
                        extra_must_keep.add(int(chain[li]))

    # Handle loops with no anchors: unvisited boundary edges form rings.
    for e, gp in boundary_edges.items():
        if e in visited_edges:
            continue
        # Walk a ring starting arbitrarily. Treat the first vertex as anchor.
        start = e[0]
        chain = _walk_chain(start, e, gp)
        if len(chain) < 4:
            continue
        # For a closed ring DP needs an open polyline; repeat first vertex
        coords = verts[chain]
        kept_local = _dp_simplify_chain(coords, simplify_tol)
        for li in kept_local:
            if 0 < li < len(chain) - 1:
                extra_must_keep.add(int(chain[li]))

    return extra_must_keep


def _simplify_ring_constrained(
    ring_coords_2d: np.ndarray,       # (N, 2) — closed ring, first != last
    keep_mask: np.ndarray,             # (N,) bool — True means MUST NOT drop
    tol: float,
) -> np.ndarray:
    """Simplify a closed ring, always preserving vertices where keep_mask is True.

    Douglas-Peucker is applied to each sub-chain between consecutive must-keep
    vertices. If no vertices are forced to stay, we DP the ring as a whole
    (rotated so the longest gap becomes the single chain).
    """
    n = len(ring_coords_2d)
    if n < 3:
        return ring_coords_2d
    if tol <= 0:
        return ring_coords_2d

    must_keep_idx = [i for i in range(n) if keep_mask[i]]
    if len(must_keep_idx) == 0:
        # No constraints — fall back to standard DP. Start/end at any vertex.
        # We can't just call DP on the closed ring without losing the closure,
        # so pick i=0 as a virtual anchor.
        idx = _dp_simplify_chain(
            np.vstack([ring_coords_2d, ring_coords_2d[:1]]), tol
        )
        # Drop the duplicated closure index
        idx = [i for i in idx if i < n]
        return ring_coords_2d[idx]

    kept_indices: list[int] = []
    m = len(must_keep_idx)
    for k in range(m):
        a = must_keep_idx[k]
        b = must_keep_idx[(k + 1) % m]
        # Extract inclusive sub-chain from a to b (wrapping around the ring)
        if b > a:
            chain_idx = list(range(a, b + 1))
        else:
            chain_idx = list(range(a, n)) + list(range(0, b + 1))
        chain_coords = ring_coords_2d[chain_idx]
        sub_keep = _dp_simplify_chain(chain_coords, tol)
        # Append all kept EXCEPT the last (it's the start of the next chain)
        kept_indices.extend(chain_idx[i] for i in sub_keep[:-1])
    # Deduplicate while preserving order
    seen = set()
    out_idx = []
    for i in kept_indices:
        if i not in seen:
            seen.add(i)
            out_idx.append(i)
    return ring_coords_2d[out_idx]


def _retriangulate_planar_region(
    mesh: trimesh.Trimesh,
    face_indices: np.ndarray,
    simplify_tol: float,
    must_keep_vertex_ids: Optional[set[int]] = None,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """For one planar region, return (vertices_3d, faces) of a re-triangulated
    version with simplified boundary. Returns None on failure.

    Approach: project every triangle of the region to the fitted plane,
    build a shapely Polygon per triangle, union them with `shapely.unary_union`,
    simplify the resulting polygon, and re-triangulate. This is robust to
    non-manifold points and holes (they come out of the union automatically)
    — much better than manual ring tracing.
    """
    try:
        from shapely.geometry import Polygon, MultiPolygon
        from shapely.ops import unary_union
        from shapely.validation import make_valid
        from trimesh.creation import triangulate_polygon
    except Exception as e:
        print(f"[planar] shapely/trimesh import failed: {e}", flush=True)
        return None

    faces = np.asarray(mesh.faces[face_indices], dtype=np.int64)
    areas = np.asarray(mesh.area_faces[face_indices], dtype=np.float64)
    centroids = np.asarray(mesh.triangles_center[face_indices], dtype=np.float64)

    # Area-weighted plane via PCA
    plane_point = (centroids * areas[:, None]).sum(0) / max(areas.sum(), 1e-20)
    diffs = centroids - plane_point
    cov = (diffs[:, :, None] * diffs[:, None, :] * areas[:, None, None]).sum(0)
    try:
        _, _, vh = np.linalg.svd(cov)
        plane_normal = vh[-1]
    except Exception:
        plane_normal = (np.asarray(mesh.face_normals[face_indices]) *
                        areas[:, None]).sum(0)
    avg_n = (np.asarray(mesh.face_normals[face_indices]) * areas[:, None]).sum(0)
    if np.dot(plane_normal, avg_n) < 0:
        plane_normal = -plane_normal
    nn = np.linalg.norm(plane_normal)
    if nn < 1e-20:
        return None
    plane_normal = plane_normal / nn

    u, v = _plane_basis(plane_normal)

    # Project every vertex of every face in the region to 2D.
    tri_verts_3d = mesh.vertices[faces]  # (F, 3, 3)
    rel = tri_verts_3d - plane_point[None, None, :]
    x = np.einsum("fvk,k->fv", rel, u)
    y = np.einsum("fvk,k->fv", rel, v)
    tri_2d = np.stack([x, y], axis=-1)  # (F, 3, 2)

    # Build 2D→vertex_id LUT once. Used both by constrained simplification
    # and by the final un-projection (to snap back to exact original coords).
    q = 1.0 / max(simplify_tol * 1e-3, 1e-9) if simplify_tol > 0 else 1e6
    coord_to_vid: dict[tuple[int, int], int] = {}
    face_verts = np.asarray(mesh.faces[face_indices], dtype=np.int64)
    for fi_local, vids in enumerate(face_verts):
        for corner in range(3):
            vid = int(vids[corner])
            x_, y_ = tri_2d[fi_local, corner]
            key = (int(round(x_ * q)), int(round(y_ * q)))
            coord_to_vid[key] = vid

    try:
        # Build polygons per triangle; skip degenerate ones.
        polys = []
        for tri in tri_2d:
            p = Polygon(tri)
            if p.is_valid and not p.is_empty and p.area > 0:
                polys.append(p)
            else:
                p2 = p.buffer(0)
                if p2.is_valid and not p2.is_empty and p2.area > 0:
                    polys.append(p2)
        if not polys:
            return None

        merged = unary_union(polys)
        if merged.is_empty:
            return None
        if merged.geom_type not in ("Polygon", "MultiPolygon"):
            return None
        if merged.geom_type == "MultiPolygon":
            poly = max(merged.geoms, key=lambda g: g.area)
        else:
            poly = merged
        if not poly.is_valid:
            poly = make_valid(poly)
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda g: g.area)
            elif poly.geom_type != "Polygon":
                return None

        # Constrained simplification: drop boundary vertices ONLY if they are
        # not shared with any face outside the cluster, otherwise the
        # neighboring (non-retriangulated) faces keep a vertex the simplified
        # polygon no longer has — a T-junction, which shows up as an open
        # edge / visible hole in the final mesh.
        if simplify_tol > 0 and must_keep_vertex_ids is not None:

            def _vid_of(xy: np.ndarray) -> int:
                key = (int(round(xy[0] * q)), int(round(xy[1] * q)))
                return coord_to_vid.get(key, -1)

            def _simplify_ring(coords_closed):
                # coords_closed: list-like of (x,y) pairs where first == last.
                arr = np.asarray(coords_closed[:-1], dtype=np.float64)
                if len(arr) < 3:
                    return coords_closed
                keep = np.zeros(len(arr), dtype=bool)
                for i, xy in enumerate(arr):
                    vid = _vid_of(xy)
                    if vid in must_keep_vertex_ids:
                        keep[i] = True
                simp = _simplify_ring_constrained(arr, keep, simplify_tol)
                # Close it
                return np.vstack([simp, simp[:1]])

            new_exterior = _simplify_ring(list(poly.exterior.coords))
            new_interiors = [_simplify_ring(list(h.coords)) for h in poly.interiors]
            try:
                simp_poly = Polygon(new_exterior, new_interiors)
                if simp_poly.is_valid and not simp_poly.is_empty:
                    poly = simp_poly
            except Exception:
                pass  # fall back to unsimplified
        elif simplify_tol > 0:
            # No constraints supplied — plain shapely simplify (may drop shared
            # boundary vertices). Kept for backwards compatibility.
            simp = poly.simplify(simplify_tol, preserve_topology=True)
            if (simp.is_valid and not simp.is_empty
                    and simp.geom_type == "Polygon" and simp.area > 0):
                poly = simp

        verts2d, tris = triangulate_polygon(poly, engine="earcut")
    except Exception as e:
        print(f"[planar] triangulate failed for region "
              f"({len(face_indices)} faces): {e}", flush=True)
        return None

    if len(tris) == 0 or len(verts2d) == 0:
        return None

    # Un-project 2D -> 3D. For vertices that correspond to an original mesh
    # vertex (by 2D coord match), use the ORIGINAL 3D coordinates — otherwise
    # the same vertex will have tiny numerical differences between adjacent
    # clusters (different (u, v, plane_point) frames) and merge_vertices
    # won't weld them, creating invisible cracks in the mesh.
    verts3d = (plane_point[None, :]
               + verts2d[:, 0:1] * u[None, :]
               + verts2d[:, 1:2] * v[None, :])

    mesh_verts = np.asarray(mesh.vertices, dtype=np.float64)
    for i, xy in enumerate(verts2d):
        key = (int(round(float(xy[0]) * q)),
               int(round(float(xy[1]) * q)))
        vid = coord_to_vid.get(key, -1)
        if vid >= 0:
            verts3d[i] = mesh_verts[vid]
    return verts3d, tris


# ==== Planar-remesh decimation =============================================

@dataclass
class PlanarStats:
    n_regions_total: int = 0
    n_regions_kept: int = 0
    n_regions_retriangulated: int = 0
    area_retriangulated: float = 0.0
    area_total: float = 0.0
    fallback_regions: int = 0


def decimate_planar(
    mesh: trimesh.Trimesh,
    preset: LevelPreset,
    post_quadric: bool = False,
    stats: Optional[PlanarStats] = None,
    quadric_target_basis: Optional[int] = None,
) -> trimesh.Trimesh:
    """Planar-region remesh.

    1. Cluster connected coplanar faces.
    2. For each cluster big enough (by face count AND area), re-triangulate
       its 2D boundary with Douglas-Peucker simplification.
    3. Small/curved regions keep their original triangles.
    4. Merge, weld duplicate vertices, remove T-vertices.
    5. Optionally run quadric decimation on the result (`post_quadric=True`).
    """
    if stats is None:
        stats = PlanarStats()

    original_face_count = len(mesh.faces)
    scale = float(mesh.scale)  # bbox diagonal
    dist_tol = preset.coplanar_dist_frac * scale
    simplify_tol = preset.boundary_simplify_frac * scale

    clusters = _cluster_coplanar_faces(mesh, preset.coplanar_deg, dist_tol)
    # Greedy merge pass: unify clusters split by MC stair-step noise.
    clusters = _merge_adjacent_coplanar_clusters(
        mesh, clusters,
        merge_tol_deg=preset.merge_deg,
        merge_dist_tol=preset.merge_dist_frac * scale,
    )
    stats.n_regions_total = len(clusters)
    stats.area_total = float(mesh.area)

    total_area = stats.area_total
    min_area = preset.min_region_area_frac * total_area

    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    used = np.zeros(len(mesh.faces), dtype=bool)

    # Pre-decide which clusters are "big enough" to retriangulate. Build a
    # per-face "group id": accepted clusters get their own id, all rejected
    # faces share id -1. A vertex is must-keep if it is referenced by more
    # than one group — that's exactly where T-junctions would otherwise
    # appear (vertex dropped on one side, kept on the other).
    face_group = np.full(len(mesh.faces), -1, dtype=np.int64)
    group_count = 0
    for c in clusters:
        a = float(face_areas[c].sum())
        if len(c) >= preset.min_region_faces and a >= min_area:
            face_group[c] = group_count
            group_count += 1

    must_keep: set[int] = set()
    if group_count > 0:
        # For each vertex, accumulate distinct group ids that use it.
        faces_arr = np.asarray(mesh.faces, dtype=np.int64)
        n_verts = len(mesh.vertices)
        # Pack (vertex_id, group_id) and take unique rows → O(F) memory.
        vg = np.stack([faces_arr.ravel(),
                       np.repeat(face_group, 3)], axis=1)
        vg = np.unique(vg, axis=0)
        v_ids = vg[:, 0]
        counts = np.bincount(v_ids, minlength=n_verts)
        multi_group_verts = np.nonzero(counts > 1)[0]
        must_keep = set(int(v) for v in multi_group_verts)

        # 3D-consistent chain simplification: for each boundary chain
        # between two accepted clusters, DP once in 3D and add the kept
        # intermediate vertices to must_keep — so both clusters agree on
        # which points to preserve and the seam stays closed.
        chain_extras = _compute_boundary_chain_must_keep(
            mesh, face_group, must_keep, simplify_tol,
        )
        must_keep |= chain_extras

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vert_offset = 0

    for c in clusters:
        a = float(face_areas[c].sum())
        if len(c) < preset.min_region_faces or a < min_area:
            continue  # too small — leave original triangles
        stats.n_regions_kept += 1
        # Also force-keep vertices shared with OTHER big clusters, so that
        # planar↔planar boundaries stay consistent from both sides.
        result = _retriangulate_planar_region(
            mesh, c, simplify_tol,
            must_keep_vertex_ids=must_keep,
        )
        if result is None:
            stats.fallback_regions += 1
            continue
        v3d, tris = result
        all_verts.append(v3d)
        all_faces.append(tris + vert_offset)
        vert_offset += v3d.shape[0]
        used[c] = True
        stats.n_regions_retriangulated += 1
        stats.area_retriangulated += a

    # Unchanged (small/curved) faces: copy originals, remapping vertices.
    keep_face_idx = np.where(~used)[0]
    if len(keep_face_idx) > 0:
        kept_faces = mesh.faces[keep_face_idx]
        used_vert_ids, inverse = np.unique(kept_faces.ravel(), return_inverse=True)
        kept_verts = mesh.vertices[used_vert_ids]
        all_verts.append(np.asarray(kept_verts, dtype=np.float64))
        all_faces.append(inverse.reshape(-1, 3) + vert_offset)
        vert_offset += kept_verts.shape[0]

    if not all_faces:
        return mesh  # nothing to do

    V = np.vstack(all_verts)
    F = np.vstack(all_faces).astype(np.int64)
    result = trimesh.Trimesh(vertices=V, faces=F, process=False)

    # Weld duplicate vertices. Choice of digits: scale-dependent, a bit tighter
    # than simplify_tol so we don't collapse genuine topology.
    weld_tol = 0.1 * simplify_tol
    if weld_tol > 0:
        # merge_vertices uses digits in base units
        digits = max(3, int(-math.log10(max(weld_tol, 1e-9))))
        try:
            result.merge_vertices(digits_vertex=digits)
        except TypeError:
            result.merge_vertices()

    # Final cleanup: ONLY the non-destructive subset of filters. In particular
    # we skip `meshing_repair_non_manifold_edges` / `_vertices` here — on a
    # freshly-assembled planar mesh those repair filters mis-classify
    # cluster-boundary edges as "non-manifold" and SPLIT them, creating
    # exactly the holes we worked so hard to avoid.
    try:
        import pymeshlab
        ms = pymeshlab.MeshSet()
        pm = pymeshlab.Mesh(
            vertex_matrix=np.asarray(result.vertices, dtype=np.float64),
            face_matrix=np.asarray(result.faces, dtype=np.int32),
        )
        ms.add_mesh(pm, "planar")
        for filt in (
            "meshing_remove_duplicate_vertices",
            "meshing_remove_duplicate_faces",
            "meshing_remove_unreferenced_vertices",
            "meshing_remove_null_faces",
        ):
            if hasattr(ms, filt):
                try:
                    getattr(ms, filt)()
                except Exception:
                    pass
        out = ms.current_mesh()
        result = trimesh.Trimesh(
            vertices=out.vertex_matrix(),
            faces=out.face_matrix(),
            process=False,
        )
    except Exception as e:
        print(f"[planar] pymeshlab cleanup skipped: {e}", flush=True)

    if post_quadric:
        # Target is computed against the ORIGINAL face count so `hybrid`
        # is never more aggressive than plain `quadric` at the same level.
        # If planar already dropped below target, skip quadric entirely —
        # its output is always at-or-better than the planar result.
        basis = quadric_target_basis if quadric_target_basis is not None \
            else original_face_count
        target = max(preset.quadric_min_faces,
                     int(round(basis * preset.quadric_ratio)))
        if target < len(result.faces):
            result = decimate_quadric(
                result, target_faces=target,
                preserve_boundary=True, feature_smooth=False,
            )

    return result


# ==== Isotropic remesh (CAD-friendly, feature-preserving) ===================

def decimate_isotropic(
    mesh: trimesh.Trimesh,
    target_edge_len_frac: float,
    feature_deg: float = ISO_FEATURE_DEG,
    iterations: int = 5,
) -> trimesh.Trimesh:
    """Uniform-edge remeshing via pymeshlab. Good for hard-surface / CAD.

    - Detects feature edges (corners ≥ `feature_deg`) and keeps them as
      sharp creases during remeshing.
    - Targets a uniform edge length, producing well-shaped triangles.
    - Because the algorithm splits and collapses edges, flat regions end up
      covered by the minimum number of similarly-sized triangles.
    """
    try:
        import pymeshlab
    except Exception as e:
        print(f"[iso] pymeshlab unavailable ({e}); returning original", flush=True)
        return mesh

    scale = float(mesh.scale)
    target_len = max(1e-6, target_edge_len_frac * scale)

    ms = pymeshlab.MeshSet()
    pm = pymeshlab.Mesh(
        vertex_matrix=np.asarray(mesh.vertices, dtype=np.float64),
        face_matrix=np.asarray(mesh.faces, dtype=np.int32),
    )
    ms.add_mesh(pm, "orig")
    _preclean_pymeshlab(ms, aggressive=not mesh.is_watertight)

    # Try the modern API with a PureValue targetlen, fall back to plain float.
    tl_kwargs = {}
    try:
        pv = pymeshlab.PureValue(target_len)
        tl_kwargs["targetlen"] = pv
    except Exception:
        tl_kwargs["targetlen"] = float(target_len)

    kw = dict(
        iterations=int(iterations),
        adaptive=False,
        selectedonly=False,
        featuredeg=float(feature_deg),
        checksurfdist=True,
        maxsurfdist=pymeshlab.PureValue(target_len * 0.5)
            if hasattr(pymeshlab, "PureValue") else float(target_len * 0.5),
        **tl_kwargs,
    )
    # Some pymeshlab versions don't support maxsurfdist as PureValue; retry
    # without it on TypeError.
    try:
        ms.meshing_isotropic_explicit_remeshing(**kw)
    except TypeError:
        kw.pop("maxsurfdist", None)
        kw.pop("checksurfdist", None)
        try:
            ms.meshing_isotropic_explicit_remeshing(**kw)
        except Exception as e:
            print(f"[iso] isotropic remesh failed: {e}", flush=True)
            return mesh
    except Exception as e:
        print(f"[iso] isotropic remesh failed: {e}", flush=True)
        return mesh

    # Post-remesh: isotropic remeshing leaves a clean mesh — only basic tidying.
    _preclean_pymeshlab(ms, aggressive=False)
    out = ms.current_mesh()
    return trimesh.Trimesh(
        vertices=out.vertex_matrix(),
        faces=out.face_matrix(),
        process=False,
    )


# ==== Auto detector =========================================================

def classify_mesh(mesh: trimesh.Trimesh, preset: Optional[LevelPreset] = None,
                  threshold: float = 0.30) -> tuple[str, float]:
    """Return ('hard_surface'|'organic', ratio). ratio = area in large planar
    regions / total area."""
    if preset is None:
        preset = LEVELS["medium"]
    scale = float(mesh.scale)
    dist_tol = preset.coplanar_dist_frac * scale
    clusters = _cluster_coplanar_faces(mesh, preset.coplanar_deg, dist_tol)
    clusters = _merge_adjacent_coplanar_clusters(
        mesh, clusters,
        merge_tol_deg=preset.merge_deg,
        merge_dist_tol=preset.merge_dist_frac * scale,
    )
    total = float(mesh.area)
    face_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    big_area_cutoff = 0.01 * total
    big_area = 0.0
    for c in clusters:
        a = float(face_areas[c].sum())
        if a >= big_area_cutoff:
            big_area += a
    ratio = big_area / max(total, 1e-20)
    return ("hard_surface" if ratio >= threshold else "organic"), ratio


# ==== Unified entry point ===================================================

@dataclass
class RunResult:
    mesh: trimesh.Trimesh
    mode_used: str
    level: str
    elapsed: float
    planar_stats: Optional[PlanarStats] = None
    classifier_ratio: Optional[float] = None


def optimize(
    mesh: trimesh.Trimesh,
    mode: str,          # 'quadric' | 'planar' | 'hybrid' | 'auto'
    level: str,         # 'weak' | 'medium' | 'strong'
) -> RunResult:
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}")
    preset = LEVELS[level]

    t0 = time.time()
    chosen = mode
    ratio = None
    planar_stats: Optional[PlanarStats] = None

    if mode == "auto":
        kind, ratio = classify_mesh(mesh, preset)
        # For hard-surface we use `planar` (preserves flat-face look perfectly).
        # `hybrid` is still experimental — quadric tends to undo the clean planar
        # work and introduces bigger deviations than planar alone.
        chosen = "planar" if kind == "hard_surface" else "quadric"

    if chosen == "quadric":
        target = max(preset.quadric_min_faces,
                     int(round(len(mesh.faces) * preset.quadric_ratio)))
        out = decimate_quadric(mesh, target_faces=target,
                               preserve_boundary=True, feature_smooth=True)
    elif chosen == "planar":
        planar_stats = PlanarStats()
        out = decimate_planar(mesh, preset, post_quadric=False, stats=planar_stats)
    elif chosen == "hybrid":
        planar_stats = PlanarStats()
        out = decimate_planar(mesh, preset, post_quadric=True, stats=planar_stats)
    elif chosen == "iso":
        out = decimate_isotropic(
            mesh,
            target_edge_len_frac=ISO_TARGET_LEN_FRAC[level],
            feature_deg=ISO_FEATURE_DEG,
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return RunResult(
        mesh=out,
        mode_used=chosen,
        level=level,
        elapsed=time.time() - t0,
        planar_stats=planar_stats,
        classifier_ratio=ratio,
    )


# ==== Metrics: symmetric Hausdorff-like deviation ===========================

def measure_deviation(
    original: trimesh.Trimesh,
    simplified: trimesh.Trimesh,
    samples: int = 20000,
    rng: Optional[np.random.Generator] = None,
) -> dict:
    """Symmetric sample-based Hausdorff approximation.

    We sample points on the surface of each mesh and measure the distance to
    the *other* mesh. Returned values are in mesh units plus normalized to
    the bbox diagonal (`..._rel`)."""
    if rng is None:
        rng = np.random.default_rng(0)

    scale = float(original.scale)
    if len(simplified.faces) == 0 or len(original.faces) == 0:
        return {"error": "empty mesh"}

    # Sample points
    try:
        pts_o, _ = trimesh.sample.sample_surface(original, samples, seed=0)
        pts_s, _ = trimesh.sample.sample_surface(simplified, samples, seed=0)
    except TypeError:
        # older trimesh: no seed kwarg
        pts_o, _ = trimesh.sample.sample_surface(original, samples)
        pts_s, _ = trimesh.sample.sample_surface(simplified, samples)

    # orig sample -> simplified mesh
    _, d_os, _ = trimesh.proximity.closest_point(simplified, pts_o)
    # simplified sample -> orig mesh
    _, d_so, _ = trimesh.proximity.closest_point(original, pts_s)

    all_d = np.concatenate([d_os, d_so])
    return {
        "hausdorff":     float(max(d_os.max(), d_so.max())),
        "hausdorff_rel": float(max(d_os.max(), d_so.max()) / scale),
        "mean":          float(all_d.mean()),
        "mean_rel":      float(all_d.mean() / scale),
        "p95":           float(np.quantile(all_d, 0.95)),
        "p95_rel":       float(np.quantile(all_d, 0.95) / scale),
    }
