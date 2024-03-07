import os
import taichi as ti
from config import mano_urdf_path, points_data_path
import numpy as np
import trimesh
from os.path import join as pjoin
from sim import RigidBody, MpmLagSim, MpmTetLagSim
from icecream import ic
from models import ManoHand
import json
from vedo import show
import pyvista as pv
from scipy.spatial.transform import Rotation as R
from utils import trimesh_show, fix_unity_urdf_tf, read_tet

if __name__ == '__main__':
    force_view_factor = 0.01
    frame = 40
    data_folder = './out'

    obj_mesh = trimesh.load_mesh(pjoin(data_folder, f'{frame}.obj'))
    hand_info = json.load(open(pjoin(data_folder, f'hand_{frame}.json'), 'r'))
    mano_hand = ManoHand(mano_urdf_path)
    mano_hand.close(hand_info['closeness'])
    hand_tf = np.asarray(hand_info['tf'])
    meshes = mano_hand.urdf.scene.dump(False).tolist()
    for mesh in meshes:
        mesh.apply_transform(hand_tf)

    vtk_mesh = pv.read('./data/touch/touch0004/haimian1_.vtk')
    vtk_mesh.points = np.load(pjoin(data_folder, f'tet_verts_{frame}.npy'))
    vtk_mesh['collision_force'] = np.load(
        pjoin(data_folder, f'collision_{frame}.npy'))
    vtk_mesh.plot()

    sensor_inds = np.load(f'./out/sensor_inds_{frame}.npz')
    sensor_pos = np.load(f'./out/sensor_pos_{frame}.npz')
    sensor_forces = np.load(f'./out/sensor_forces_{frame}.npz')
    pcds = []
    paths = []
    for link_name in sensor_forces.keys():
        pos = sensor_pos[link_name]
        forces = -sensor_forces[link_name]
        ic(forces[:3])
        colors = ((forces * 3 + 0.5) * 255).astype(np.uint8)
        pcds.append(trimesh.PointCloud(pos, colors))
        force_segs = np.stack([pos, pos + forces * force_view_factor], axis=1)
        paths.append(trimesh.load_path(force_segs))

    trimesh_show(meshes + pcds + paths + [obj_mesh])
    trimesh_show(meshes + pcds + paths)
