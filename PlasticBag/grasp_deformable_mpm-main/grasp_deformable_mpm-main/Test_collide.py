import taichi as ti
import numpy as np
import trimesh
import pyvista as pv
from sim import MpmLagSim, MpmTetLagSim
from utils import read_tet

def test_lag_mpm():
    ti.init(arch=ti.gpu)
    dt = 1e-4
    sim = MpmLagSim(origin=np.asarray([-0.5,-0.4,-0.5]), dt=dt)
    plastic_mesh = trimesh.load_mesh('./data/plasticbag1_n.obj')
    # sim = MpmTetLagSim(origin=np.asarray([-0.5,-0.4,-0.5]), dt=dt)
    # plastic_mesh = read_tet('./data/object_meshes/dumpling1_.vtk')
    sim.set_soft(plastic_mesh)
    sim.init_sim()
    sim.bending_p = 0.5

    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        if sim.window.is_pressed(ti.GUI.UP):
            # print("hehe")
            sim.substep_1()
        else:
            sim.substep()
        sim.update_scene()
        
        if sim.window.is_pressed(ti.GUI.SPACE):
            pl = pv.Plotter()
            reader_1 = pv.get_reader('./vtk/plasticbag1_n.vtk')
            mesh_1 = reader_1.read()
            x_position = sim.x_soft.to_numpy()
            mesh_1.points = x_position
            pl.add_mesh(mesh_1, show_edges=True, color='white',opacity = 0.5,lighting = False)
            pl.show()
        sim.show()

if __name__ == '__main__':
    test_lag_mpm()