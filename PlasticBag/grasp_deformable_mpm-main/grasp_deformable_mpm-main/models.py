import re
import os 
import json 
import taichi as ti 
import yourdfpy
from yourdfpy import URDF, Joint
from config import mano_urdf_path, points_data_path
from typing import List, Dict
import numpy as np
import trimesh
from os.path import join as pjoin
from sim import RigidBody, MpmLagSim, MpmTetLagSim
import time 
from utils import transform_verts
from icecream import ic 


class ManoHand:
    def __init__(self, urdf_path, root_transform=np.eye(4)) -> None:
        self.urdf: URDF = yourdfpy.URDF.load(urdf_path)
        self.joint_names: List[str] = self.urdf.actuated_joint_names
        self.joints: List[Joint] = [self.urdf.joint_map[jname]
                                    for jname in self.joint_names]
        self.lower_cfg: np.ndarray = np.asarray(
            [j.limit.lower for j in self.joints])
        self.upper_cfg: np.ndarray = np.asarray(
            [j.limit.upper for j in self.joints])
        self.link_names: List[str] = []
        self.link_rigids: List[RigidBody] = []
        self.link_rest_verts: List[np.ndarray] = []
        urdf_folder = os.path.dirname(urdf_path)
        self.cfg = {k : v for k, v in zip(self.joint_names, self.lower_cfg)}
        self.urdf.update_cfg(self.cfg)
        for link_name, link in self.urdf.link_map.items():
            link_mesh = trimesh.load_mesh(
                pjoin(urdf_folder, link.visuals[0].geometry.mesh.filename))
            self.link_names.append(link_name)
            self.link_rest_verts.append(np.asarray(link_mesh.vertices))
            local_tf = self.urdf.scene.graph.get(link_name)[0]
            link_mesh.apply_transform(root_transform @ local_tf)
            self.link_rigids.append(RigidBody(link_mesh))
        self.root_transform: np.ndarray = root_transform

    def close(self, closeness: float):
        cfg_vals = self.lower_cfg * (1 - closeness) + self.upper_cfg * closeness
        self.cfg = {k : v for k, v in zip(self.joint_names, cfg_vals)}
        self.urdf.update_cfg(self.cfg)
        for link_name, link_rigid, rest_link_verts in zip(self.link_names, self.link_rigids, self.link_rest_verts):
            transform = self.urdf.scene.graph.get(link_name)[0]
            target_verts = transform_verts(rest_link_verts, transform) 
            target_verts = transform_verts(target_verts, self.root_transform)
            link_rigid.set_target(target_verts)

    def set_root_transform(self, transform: np.ndarray):
        self.root_transform = transform

    def get_link_local_transform(self, link_name: str):
        return self.urdf.scene.graph.get(link_name)[0]

    def show(self):
        self.urdf.show()


class ManoHandSensors:
    point_index_reg = re.compile(r'\((\d+),\s(\d+)\)')
    def __init__(self, mano_hand: ManoHand, sensor_points_path: str) -> None:
        self.mano_hand = mano_hand
        with open(sensor_points_path, 'r') as f:
            sensors_info = json.load(f)
        
        link_sensors_inds_dict = {k: [] for k in mano_hand.urdf.link_map.keys()}
        link_sensors_dict = {k: [] for k in mano_hand.urdf.link_map.keys()}
        for k, v in sensors_info.items():
            if k.startswith('$'):
                continue
            pind = ManoHandSensors.point_index_reg.match(k).groups()
            pind = [int(pind[0]), int(pind[1])]
            link_name = v['Item2']
            pos = v['Item3']
            pos = np.asarray([pos['z'], -pos['x'], pos['y']])
            link_sensors_dict[link_name].append(pos)
            link_sensors_inds_dict[link_name].append(pind)
        
        self.link_names = []
        self.link_sensor_rest_pos = []
        self.link_sensor_inds = []
        for k, v in link_sensors_dict.items():
            if not len(v):
                link_sensors_dict[k] = None 
            else: 
                self.link_names.append(k)
                self.link_sensor_rest_pos.append(np.stack(v, axis=0))
                self.link_sensor_inds.append(np.stack(link_sensors_inds_dict[k], axis=0))
    
    @property
    def link_sensor_pos(self):
        ret = []
        for link_name, sensor_rest_pos in zip(self.link_names, self.link_sensor_rest_pos):
            link_transform = self.mano_hand.get_link_local_transform(link_name)
            ret.append(transform_verts(sensor_rest_pos, self.mano_hand.root_transform @ link_transform))
        return ret

    @property
    def link_sensor_inds_pos(self):
        return zip(self.link_sensor_inds, self.link_sensor_pos)
    
    @property
    def link_sensor_name_inds_pos(self):
        return zip(self.link_names, self.link_sensor_inds, self.link_sensor_pos)


if __name__ == '__main__':
    ti.init(arch=ti.cuda)
    substeps = 5

    root_tf = np.eye(4)
    root_tf[:3, 3] = [0.03, 0.15, 0.26]
    mano_hand = ManoHand(mano_urdf_path, root_tf)
    sim = MpmTetLagSim(origin=np.asarray([-0.5,] * 3))
    for rigid in mano_hand.link_rigids:
        sim.add_kinematic_rigid(rigid)
    box_mesh = trimesh.load_mesh(pjoin('./data/object_meshes', 'box.obj'))
    sim.set_soft(box_mesh)

    sim.init_sim()
    closeness = 0.

    frame = 0
    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        if closeness < 0.7:
            closeness += 0.001

        for i in range(substeps):
            sim.substep()
        sim.update_scene()
        sim.show()
        mano_hand.close(closeness)
        sim.toward_kinematic_target(substeps=substeps)
        frame += 1
    