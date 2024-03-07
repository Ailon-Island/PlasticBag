import re
import yourdfpy
import json
import numpy as np
import trimesh
from icecream import ic
from yourdfpy import URDF
from config import mano_urdf_path, points_data_path
from vedo import show

mano_hand: URDF = URDF.load(mano_urdf_path)
mano_joint_names = mano_hand.actuated_joint_names
lower_config = {k: v.limit.lower for k, v in mano_hand.joint_map.items()}
upper_config = {k: v.limit.upper for k, v in mano_hand.joint_map.items()}
mano_hand.update_cfg(lower_config)
mano_hand.show()
mano_hand.update_cfg(upper_config)
mano_hand.show()

point_index_reg = re.compile(r'\((\d+),\s(\d+)\)')
with open(points_data_path, 'r') as f:
    points_info = json.load(f)

link_points_dict = {k: [] for k in mano_hand.link_map.keys()}
for k, v in points_info.items():
    if k.startswith('$'):
        continue
    pind = point_index_reg.match(k).groups()
    pind = (int(pind[0]), int(pind[1]))
    link_name = v['Item2']
    pos = v['Item3']
    pos = np.asarray([pos['z'], -pos['x'], pos['y']])
    link_points_dict[link_name].append(pos)

scene: trimesh.Scene = mano_hand._scene.copy()

for k, v in link_points_dict.items():
    if not len(v):
        continue
    k += '.stl'
    pcd = trimesh.PointCloud(np.stack(v, axis=0))
    scene.add_geometry(pcd, node_name=k + '_pcd', parent_node_name=k)

scene.show()
meshes = scene.dump(False)
show(meshes)
for geom in meshes:
    ic(geom.vertices.max(axis=0), geom.vertices.min(axis=0))
for _, link in mano_hand.link_map.items():
    ic(_, link.visuals[0].geometry.mesh.filename, scene.graph.get(_)[0])
