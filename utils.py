import taichi as ti
import trimesh
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from icecream import ic

T = ti.f32
mat3 = ti.types.matrix(3, 3, dtype=T)

vecs = ti.Vector.field
mats = ti.Matrix.field
scalars = ti.field


@ti.func
def deg2rad(x):
    return x * 0.017453292519943295


@ti.func
def rad2deg(x):
    return x * 57.29577951308232


def get_reduce_min_inplace(keys, values):
    n = keys.shape[0]

    # return reduce_min_func


def trimesh_show(geoms):
    scene = trimesh.Scene()
    for g in geoms:
        scene.add_geometry(g)
    scene.show()


def transform_verts(verts: np.ndarray, transform: np.ndarray):
    assert (len(verts.shape) == 2)
    assert (verts.shape[1] == 3)
    assert (transform.shape == (4, 4))
    return verts @ transform[:3, :3].T + transform[:3, 3]


left_tf = np.empty((4, 4), float)
right_tf = np.empty((4, 4), float)
left_tf[:3, :3] = R.from_euler('z', -np.pi/2).as_matrix()
right_tf[:3, :3] = R.from_euler('z', np.pi/2).as_matrix()
left_tf[:3, 3] = right_tf[:3, 3] = left_tf[3, :3] = right_tf[3, :3] = 0
left_tf[3, 3] = right_tf[3, 3] = 1.


def fix_unity_urdf_tf(tf):
    return left_tf @ tf @ right_tf

def get_mouse_ray(win: ti.ui.Window, cam: ti.ui.Camera):
    # calculate ray from mouse
    ndc_pos = np.array(win.get_cursor_pos()) * 2 - 1
    res = win.get_window_shape()
    inv_cam_mat = np.linalg.inv(
        cam.get_view_matrix() @ cam.get_projection_matrix(res[0] / res[1]))
    ray_ndc = np.array([ndc_pos[0], ndc_pos[1], 1, 1]) # z = -1 or 1?
    ray_world = ray_ndc @ inv_cam_mat
    ray_world /= ray_world[-1]
    ray_origin = cam.curr_position.to_numpy()
    ray_dir = ray_world[:-1] - ray_origin
    ray_dir /= np.linalg.norm(ray_dir)
    ray_origin = ti.Vector(ray_origin)
    ray_dir = ti.Vector(ray_dir)
    return ray_origin, ray_dir

class TetMesh:
    def __init__(self, verts: np.ndarray,
                 surf_tris: np.ndarray,
                 surf_vert_inds: np.ndarray,
                 tets: np.ndarray) -> None:
        self.verts = verts
        self.surf_tris = surf_tris
        self.surf_vert_inds = surf_vert_inds
        self.tets = tets
        self.surf_mesh = trimesh.Trimesh(self.verts, self.surf_tris)

    @property
    def n_verts(self):
        return self.verts.shape[0]

    @property
    def n_tets(self):
        return self.tets.shape[0]

    @property
    def show(self):
        trimesh_show([self.surf_mesh])


def read_tet(path):
    tet_mesh = pv.read(path)
    points = np.asarray(tet_mesh.points)
    tets = np.asarray(tet_mesh.cells_dict[10])

    n_points = tet_mesh.points.shape[0]
    tet_mesh['orig_inds'] = np.arange(n_points, dtype=np.int32)
    surf_mesh = tet_mesh.extract_surface()
    surf_tris = np.asarray(surf_mesh.faces).reshape(-1, 4)[:, 1:]
    surf_tris = np.asarray(surf_mesh['orig_inds'][surf_tris])
    surf_vert_inds = np.asarray(surf_mesh['orig_inds'])

    return TetMesh(points, surf_tris, surf_vert_inds, tets)


def interpolate_from_mesh(query_points: np.ndarray,
                          mesh: trimesh.Trimesh,
                          mesh_value: np.ndarray,
                          dist_thresh=0.005,
                          default_value=0):
    assert (len(query_points.shape) == 2)
    assert (query_points.shape[1] == 3)
    if len(mesh_value.shape) == 1:
        mesh_value = mesh_value[:, None]
    closest, distance, tri_inds = trimesh.proximity.closest_point(
        mesh, query_points)
    tri_vert_inds = mesh.faces[tri_inds]
    tri_verts = mesh.vertices[tri_vert_inds]
    bc_weights = trimesh.triangles.points_to_barycentric(
        tri_verts, closest)
    query_value = (mesh_value[tri_vert_inds] *
                   bc_weights[:, :, None]).sum(axis=1)
    query_value[distance > dist_thresh] = default_value
    return query_value
