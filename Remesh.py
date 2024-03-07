import pyacvd  # pip install pyacvd
import pyvista as pv  # pip install pyvista
import os
import argparse
import numpy as np

# dataDir = "E:\\Data\\CLOTH3D\\train"
# dataDir = "D:\\VR\\Cloth3d\\val_t1"
dataDir = "./data"
print(os.walk(dataDir))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str,
                        help='the selected type for remeshing')
    parser.add_argument('--num', type=int, default=5000,
                        help='the maximum number of vertices after remeshing')
    parser.add_argument('--dynamic', action='store_true',
                        help='whether to use dynamic number of particles')

    args = parser.parse_args()
    # for root, dirs, files in os.walk(dataDir):
    #     for file in files:
    #         if '.obj' in file and args.type in file and 'slim' not in file and 'remesh' not in file:
    file_name = "./data/plasticbag1.obj"   # os.path.join(root, file)
    print("Processing {}".format(file_name))

    mesh = pv.read(file_name)
    vertex_num = mesh.number_of_points
    # mesh.plot(text="original", show_edges=True)

    clus = pyacvd.Clustering(mesh)
    clus.subdivide(3)

    if args.dynamic:
        print(vertex_num)
        num = int(vertex_num * 0.2)
        if num > args.num:
            num = args.num
    else:
        num = args.num

    clus.cluster(num)

    remesh = clus.create_mesh()
    # remesh.plot(text="remesh", show_edges=True)

    # os.path.join(root, "remesh_{}_{}".format(num, file))
    new_file_name = "./data/plasticbag-5e4.obj"
    pv.save_meshio(new_file_name, remesh)
    print("Saving to {}".format(new_file_name))
