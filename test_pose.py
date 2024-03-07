import os
import taichi as ti
from config import mano_urdf_path, points_data_path
import numpy as np
import trimesh
from os.path import join as pjoin
from sim import RigidBody, MpmLagSim, MpmTetLagSim
from icecream import ic
from models import ManoHand, ManoHandSensors
import json
from vedo import show
from scipy.spatial.transform import Rotation as R
from utils import trimesh_show, fix_unity_urdf_tf, read_tet, \
    interpolate_from_mesh


def test_trimesh_pose(touch_folder):
    hand_tf_path = pjoin(touch_folder, 'HandPose.json')
    with open(hand_tf_path, 'r') as f:
        hand_tf = np.asarray(json.load(f))
    hand_tf = fix_unity_urdf_tf(hand_tf)

    mano_hand = ManoHand(mano_urdf_path, np.eye(4))
    dumpling_mesh = trimesh.load_mesh(pjoin(touch_folder, 'dumpling1.obj'))
    meshes = mano_hand.urdf.scene.dump(False).tolist()
    for mesh in meshes:
        mesh.apply_transform(hand_tf)
    meshes.append(dumpling_mesh)
    trimesh_show(meshes)


def test_sim():
    ti.init(arch=ti.cuda)
    substeps = 5

    touch_folder = './data/touch/touch0001/'
    hand_tf_path = pjoin(touch_folder, 'HandPose.json')
    with open(hand_tf_path, 'r') as f:
        hand_tf = np.asarray(json.load(f))
    hand_tf = fix_unity_urdf_tf(hand_tf)

    mano_hand = ManoHand(mano_urdf_path, hand_tf)
    sim = MpmLagSim(origin=np.asarray([-0.5,] * 3))
    sim.set_camera_pos(0.75, 1, 0.3)
    for rigid in mano_hand.link_rigids:
        sim.add_kinematic_rigid(rigid)
    dumpling_mesh = trimesh.load_mesh(pjoin(touch_folder, 'dumpling1.obj'))
    sim.set_soft(dumpling_mesh)

    sim.init_sim()
    frame = 0
    closeness = 0
    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        if closeness < 0.7:
            closeness += 0.01

        for i in range(substeps):
            sim.substep()
        sim.update_scene()
        sim.show()
        mano_hand.close(closeness)
        sim.toward_kinematic_target(substeps=substeps)
        export_mesh = trimesh.Trimesh(
            sim.x_soft.to_numpy(), np.asarray(dumpling_mesh.faces))
        export_mesh.export(f'./out/{frame}.obj')
        frame += 1


def test_tet_sim():
    ti.init(arch=ti.cuda)
    substeps = 5

    touch_folder = './data/'
    hand_tf_path = pjoin(touch_folder, 'HandPose.json')
    with open(hand_tf_path, 'r') as f:
        hand_tf = np.asarray(json.load(f))
    hand_tf = fix_unity_urdf_tf(hand_tf)

    mano_hand = ManoHand(mano_urdf_path, hand_tf)
    sensors = ManoHandSensors(mano_hand, points_data_path)
    sim = MpmTetLagSim(origin=np.asarray([-0.5,] * 3))
    sim.set_camera_pos(0.75, 1, 0.3)
    for rigid in mano_hand.link_rigids:
        sim.add_kinematic_rigid(rigid)
    obj_tet_mesh = read_tet(pjoin(touch_folder, 'haimian1_.vtk'))
    sim.set_soft(obj_tet_mesh)
    tet_surf_tris = np.asarray(obj_tet_mesh.surf_tris)

    sim.init_sim()
    frame = 0
    closeness = 0
    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        for i in range(substeps):
            sim.substep()
        sim.update_scene()
        sim.show()

        tet_verts = sim.x_soft.to_numpy() + sim.origin
        tet_vert_collision_forces = sim.collision_force_soft.to_numpy()
        tet_vert_elastic_forces = sim.elastic_force_soft.to_numpy()
        surf_mesh = trimesh.Trimesh(tet_verts, tet_surf_tris)
        surf_vert_collision_forces = tet_vert_collision_forces[obj_tet_mesh.surf_vert_inds]
        surf_mesh.export(f'./out/{frame}.obj')
        np.save(f'./out/tet_verts_{frame}.npy', tet_verts)
        np.save(f'./out/collision_{frame}.npy', tet_vert_collision_forces)
        np.save(f'./out/elastic_{frame}.npy', tet_vert_elastic_forces)
        hand_info = {
            'tf': hand_tf.tolist(),
            'closeness': closeness,
            'joints': mano_hand.cfg
        }
        with open(f'./out/hand_{frame}.json', 'w') as f:
            json.dump(hand_info, f)
        link_sensor_inds = {}
        link_sensor_pos = {}
        link_sensor_forces = {}

        link_sensors_offsets = [0]
        n_sensors = 0
        for p in sensors.link_sensor_pos:
            n_sensors += p.shape[0]
            link_sensors_offsets.append(n_sensors)
        query_pos = np.concatenate(sensors.link_sensor_pos, axis=0)
        queried_forces = sim.query_collision_force(query_pos)

        for link_ind, link_name in enumerate(sensors.link_names):
            sensor_inds = sensors.link_sensor_inds[link_ind]
            sensor_pos = sensors.link_sensor_pos[link_ind]
            link_sensor_pos[link_name] = sensor_pos
            link_sensor_inds[link_name] = sensor_inds
            link_sensor_forces[link_name] = queried_forces[link_sensors_offsets[link_ind]
                :link_sensors_offsets[link_ind+1]]

        np.savez(f'./out/sensor_inds_{frame}.npz', **link_sensor_inds)
        np.savez(f'./out/sensor_pos_{frame}.npz', **link_sensor_pos)
        np.savez(f'./out/sensor_forces_{frame}.npz', **link_sensor_forces)

        if closeness < 0.7:
            closeness += 0.01
        else:
            break
        mano_hand.close(closeness)
        sim.toward_kinematic_target(substeps=substeps)
        frame += 1


if __name__ == '__main__':
    test_tet_sim()
