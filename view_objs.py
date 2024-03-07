import os
from config import mano_urdf_path, points_data_path
import numpy as np
import trimesh
from os.path import join as pjoin
import json
from vedo import show
from icecream import ic


def read_tris(path):
    res = []
    with open(path, 'r') as f:
        for line in f:
            res.append(int(line.strip()))
    return np.asarray(res).reshape(-1, 3)


def read_verts(path):
    res = []
    with open(path, 'r') as f:
        for line in f:
            res.append([float(x) for x in line.strip().split(' ')])
    return np.asarray(res)


if __name__ == '__main__':
    touch_folder = './data/touch/touch0001/'
    hand_tris = read_tris(pjoin(touch_folder, 'pc_hand_t.txt'))
    hand_verts = read_verts(pjoin(touch_folder, 'pc_hand.txt'))
    hand_mesh = trimesh.Trimesh(hand_verts, hand_tris)
    obj_tris = read_tris(pjoin(touch_folder, 'pc_obj_t.txt'))
    obj_verts = read_verts(pjoin(touch_folder, 'pc_obj.txt'))
    obj_mesh = trimesh.Trimesh(obj_verts, obj_tris)
    show([hand_mesh, obj_mesh])
