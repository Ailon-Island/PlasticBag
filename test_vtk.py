import pyvista as pv
from icecream import ic
import numpy as np
import trimesh
from vedo import show

if __name__ == '__main__':
    tet_mesh = pv.read('./vtk/plasticbag2.vtk')
    points = np.asarray(tet_mesh.points)
    tets = np.asarray(tet_mesh.cells_dict[10])
    n_points = tet_mesh.points.shape[0]
    tet_mesh['orig_inds'] = np.arange(n_points, dtype=np.int32)

    surf_mesh = tet_mesh.extract_surface()
    surf_tris = np.asarray(surf_mesh.faces).reshape(-1, 4)[:, 1:]
    surf_tris = np.asarray(surf_mesh['orig_inds'][surf_tris])

    # ic(points.shape)
    # surf_mesh = pv.PolyData(points, surf_tris.astype(np.int32))
    # surf_mesh.plot()
    tri_mesh = trimesh.Trimesh(points, surf_tris)
    show([tri_mesh])
