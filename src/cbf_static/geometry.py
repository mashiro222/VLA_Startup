"""Geometry in metres. Ellipsoids use (x-c).T A (x-c) <= 1."""
from itertools import product
import numpy as np
from scipy.spatial import ConvexHull


def mvee(points, tolerance=1e-3, max_iterations=20000):
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite 3D collision vertices')
    if np.linalg.matrix_rank(points - points.mean(0)) != 3:
        raise ValueError('Degenerate collision geometry; no silent bounding fallback')
    points = points[ConvexHull(points).vertices]
    q = np.vstack([points.T, np.ones(len(points))])
    u = np.full(len(points), 1 / len(points))
    converged = False
    for _ in range(max_iterations):
        v = (q * u) @ q.T
        scores = np.sum(q * np.linalg.solve(v, q), axis=0)
        j = int(np.argmax(scores)); maximum = scores[j]
        if maximum <= 4 * (1 + tolerance):
            converged = True
            break
        step = (maximum - 4) / (4 * (maximum - 1))
        u *= 1 - step; u[j] += step
    if not converged:
        raise RuntimeError('MVEE failed to converge')
    center = points.T @ u
    a = np.linalg.inv((points.T * u) @ points - np.outer(center, center)) / 3
    # Numerical termination must never leave a mesh vertex outside the enclosure.
    d = points - center
    maximum = np.max(np.einsum('ni,ij,nj->n', d, a, d))
    a /= max(1., maximum) * (1 + 1e-10)
    return center, a


def inflate(a, radius):
    """Conservative outer ellipsoid of E + radius*unit_ball, not axis addition."""
    if radius < 0:
        raise ValueError('Negative radius')
    if radius == 0:
        return a.copy()
    q = np.linalg.inv(a)
    beta = radius * np.sqrt(3 / np.trace(q))
    return np.linalg.inv((1 + beta) * q + (1 + 1 / beta) * radius**2 * np.eye(3))


def barrier(p, centers, matrices):
    d = np.asarray(p) - np.asarray(centers).reshape(-1, 3)
    a = np.asarray(matrices).reshape(-1, 3, 3)
    return np.einsum('ni,nij,nj->n', d, a, d) - 1, 2 * np.einsum('nij,nj->ni', a, d)


def nearest_segment(centers, start, end):
    centers = np.asarray(centers)
    d = np.asarray(end) - start
    t = np.clip((centers - start) @ d / max(float(d @ d), 1e-20), 0, 1)
    return int(np.argmin(np.linalg.norm(centers - (start + t[:, None] * d), axis=1)))


def geom_vertices(sim, gid):
    """World collision mesh vertices; primitive boxes conservatively enclose solids.

    MuJoCo's compiled mesh vertices already contain asset scaling/centering.
    Planes/hfields/SDFs are deliberately unsupported for movable obstacles.
    """
    import mujoco
    m, d = sim.model, sim.data
    kind = int(m.geom_type[gid]); size = np.asarray(m.geom_size[gid])
    if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh = m.geom_dataid[gid]; offset = m.mesh_vertadr[mesh]; count = m.mesh_vertnum[mesh]
        pts = np.asarray(m.mesh_vert[offset:offset + count], dtype=float)
    else:
        if kind == int(mujoco.mjtGeom.mjGEOM_BOX) or kind == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            half = size
        elif kind == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            half = np.repeat(size[0], 3)
        elif kind in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            half = np.array([size[0], size[0], size[1] + (size[0] if kind == int(mujoco.mjtGeom.mjGEOM_CAPSULE) else 0)])
        else:
            raise ValueError(f'Unsupported obstacle collision geom type {kind}')
        pts = np.array(list(product([-1, 1], repeat=3))) * half
    return pts @ np.asarray(d.geom_xmat[gid]).reshape(3, 3).T + d.geom_xpos[gid]
