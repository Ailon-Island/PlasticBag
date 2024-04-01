import taichi as ti
import numpy as np
import trimesh
import pyvista as pv
from sim2A import MpmLagSim
from utils import read_tet


def test_lag_mpm():
    ti.init(arch=ti.cuda, debug=False)
    sim = MpmLagSim()
    plastic_mesh = trimesh.load_mesh('./data/bag-5e3.obj')
    # sim = MpmTetLagSim(origin=np.asarray([-0.5,-0.4,-0.5]), dt=dt)
    # plastic_mesh = read_tet('./data/object_meshes/dumpling1_.vtk')
    sim.set_soft(plastic_mesh)
    # sim.set_air(100000, 1e-4, 1e-5, 1e-1, 0.1)
    # sim.set_air(100000, 1e-4, 1e-5, 1e-1, 0.1)
    sim.init_sim()

    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        # if sim.window.is_pressed("w"):
        #     sim.camera.position
        # 老规矩 y 轴正方向朝上
        if not sim.object_selected[None] and not sim.soft_dragging and not sim.soft_selecting:
            sim.camera.track_user_inputs(sim.window, 0.01 / 2, hold_key=ti.ui.LMB)
        
        sim.substep()
        if sim.window.is_pressed("r"):
            sim.camera.position(-0.7, 0.7, 0.3)
            sim.camera.lookat(0.5, 0.5, 0.5)
        sim.update_scene(True)

        if sim.window.is_pressed(ti.GUI.SPACE):
            pl = pv.Plotter()
            reader_1 = pv.get_reader('./vtk/plasticbag1_n.vtk')
            mesh_1 = reader_1.read()
            x_position = sim.x_soft.to_numpy()
            mesh_1.points = x_position
            pl.add_mesh(mesh_1, show_edges=True, color='white',
                        opacity=0.5, lighting=False)
            pl.show()
        sim.show()


if __name__ == '__main__':
    test_lag_mpm()
