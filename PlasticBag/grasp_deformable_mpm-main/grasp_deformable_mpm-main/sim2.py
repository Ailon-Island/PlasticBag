from os.path import join as pjoin
import time
import trimesh
import taichi as ti
import numpy as np
from utils import mat3, scalars, vecs, mats, T, TetMesh
from typing import Optional, List
from vedo import show
from icecream import ic
# current: only neo-hookean + rigid body


class Material:
    pass


class NeoHookean(Material):
    def __init__(self, E: float = 0.1e4, nu: float = 0.2) -> None:
        super().__init__()
        self.mu, self.lam = E / (2 * (1 + nu)), E * \
            nu / ((1+nu) * (1 - 2 * nu))

    @ti.func
    def energy(self, F: mat3):
        logJ = ti.math.log(F.determinant())
        return 0.5 * self.mu * ((F.transpose() * F).trace() - 3) - \
            self.mu * logJ + 0.5 * self.lam * logJ ** 2


class Body:
    def __init__(self) -> None:
        pass

    @property
    def n_pars(self):
        pass


class SoftBody(Body):
    def __init__(self, rest_pars_pos: np.ndarray, material: Material) -> None:
        self.rest_pos: np.ndarray = rest_pars_pos
        self.material: Material = material

    @property
    def n_pars(self):
        return self.rest_pos.shape[0]


class RigidBody(Body):
    def __init__(self, mesh: trimesh.Trimesh, dx=1/128) -> None:
        self.mesh: trimesh.Trimesh = mesh.copy()
        n_points = int(3 * mesh.area / dx ** 2)
        points, face_inds = trimesh.sample.sample_surface_even(
            self.mesh, n_points)
        self.sample_faces_verts = self.mesh.vertices[self.mesh.faces[face_inds]]
        self.sample_bc_weights = trimesh.triangles.points_to_barycentric(
            self.sample_faces_verts, points)
        self.rest_pos: np.ndarray = points
        self.tri_inds: np.ndarray = face_inds
        self.target_pos = self.rest_pos

    @property
    def n_pars(self):
        return self.rest_pos.shape[0]

    @property
    def n_tris(self):
        return self.mesh.faces.shape[0]

    def set_target(self, target_vert_pos: np.ndarray):
        target_faces_verts = target_vert_pos[self.mesh.faces[self.tri_inds]]
        self.target_pos = (self.sample_bc_weights[:, :, None] *
                           target_faces_verts).sum(axis=1)


# NOTE: now soft only one support mesh
# reference: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm_lagrangian_forces.py
@ti.data_oriented
class MpmLagSim:
    def __init__(self, dt: float = 1e-4,
                 origin: np.ndarray = np.zeros((3,), float),
                 n_grids: int = 128,
                 scale: float = 1.0) -> None:
        self.dt = dt
        # 128
        self.n_grids = n_grids
        # scale = 1m
        self.scale = scale
        # 1 / 128
        self.dx = 1. / self.n_grids
        self.inv_dx = self.n_grids
        self.origin = origin
        # 空间范围（超过会回弹）
        self.bound = 30
        self.gravity = 9.8
        self.bending_p = 0.05
        self.lift_state = 0
        self.clear_bodies()

        # TODO: align these parameters in the future
        self.p_vol, self.p_rho = (self.dx * 0.5)**2, 0.1
        self.p_mass = self.p_vol * self.p_rho
        self.rp_rho = 10
        # rigid part
        self.rp_mass = self.p_vol * self.rp_rho

        E, nu = 1e1, 0.49
        self.mu, self.lam = E / (2 * (1 + nu)), E * \
            nu / ((1 + nu) * (1 - 2 * nu))
        self.eps = 1e-6

        self.window = ti.ui.Window("CPIC-Scene", (768, 768))
        self.canvas = self.window.get_canvas()
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.camera.position(+0.7, 0.7, 0.3)
        self.camera.lookat(0.5, 0.5, 0.5)
        self.canvas.set_background_color((1, 1, 1))

    def clear_bodies(self):
        self.rigid_bodies: List[RigidBody] = []
        self.n_rigid_pars = 0
        self.soft_mesh: Optional[trimesh.Trimesh] = None
        self.n_soft_verts = 0
        self.n_soft_tris = 0

    def set_camera_pos(self, x, y, z):
        self.camera.position(x, y, z)

    def camera_lookat(self, x, y, z):
        self.camera.lookat(x, y, z)

    @property
    def n_rigid(self):
        return len(self.rigid_bodies)

    def init_sim(self):
        self.x_soft = vecs(3, T, self.n_soft_verts, needs_grad=True)
        self.v_soft = vecs(3, T, self.n_soft_verts)
        self.C_soft = mats(3, 3, T, self.n_soft_verts)
        self.restInvT_soft = mats(2, 2, T, self.n_soft_tris)
        self.energy_soft = scalars(T, shape=(), needs_grad=True)
        self.tris_soft = scalars(int, (self.n_soft_tris, 3))
        self.nrm_soft = vecs(3, T, self.n_soft_tris)
        self.tris_area_soft = scalars(float, (self.n_soft_tris,))
        # self.x_rigid = vecs(3, T, self.n_rigid_pars)
        # self.v_rigid = vecs(3, T, self.n_rigid_pars)

        self.grid_v = vecs(3, T, (self.n_grids, self.n_grids, self.n_grids))
        self.grid_m = scalars(T, (self.n_grids, self.n_grids, self.n_grids))

        self.x_soft.from_numpy(np.asarray(
            self.soft_mesh.vertices) - self.origin)  # 这里是减
        self.tris_soft.from_numpy(np.asarray(self.soft_mesh.faces))
        soft_face_adjacency = self.soft_mesh.face_adjacency
        self.n_soft_bends = soft_face_adjacency.shape[0]
        self.bending_faces = vecs(2, int, self.n_soft_bends)
        self.rest_bending_soft = scalars(T, shape=(self.n_soft_bends,))
        self.bending_faces.from_numpy(soft_face_adjacency)
        self.bending_edges = vecs(2, int, self.n_soft_bends)
        self.bending_edges.from_numpy(self.soft_mesh.face_adjacency_edges)
        # x_rigid = np.concatenate(
        # [b.rest_pos for b in self.rigid_bodies], axis=0) - self.origin
        # self.x_rigid.from_numpy(x_rigid)
        self.init_field()

        self.lines = ti.Vector.field(3, ti.f32, 8)
        height = 30 / 128
        size = 1
        self.lines[0] = [0, height, 0]
        self.lines[1] = [+size, height, 0]
        self.lines[2] = [+size, height, 0]
        self.lines[3] = [+size, height, +size]
        self.lines[4] = [+size, height, +size]
        self.lines[5] = [0, height, +size]
        self.lines[6] = [0, height, +size]
        self.lines[7] = [0, height, 0]

    @ti.func
    def compute_T_soft(self, i):
        a, b, c = self.tris_soft[i,
                                 0], self.tris_soft[i, 1], self.tris_soft[i, 2]
        xab = self.x_soft[b] - self.x_soft[a]
        xac = self.x_soft[c] - self.x_soft[a]
        return ti.Matrix([
            [xab[0], xac[0]],
            [xab[1], xac[1]],
            [xab[2], xac[2]]
        ])

    @ti.func
    def compute_area_soft(self, i):
        a, b, c = self.tris_soft[i,
                                 0], self.tris_soft[i, 1], self.tris_soft[i, 2]
        xab = self.x_soft[b] - self.x_soft[a]
        xac = self.x_soft[c] - self.x_soft[a]
        return 0.5 * xab.cross(xac).norm()

    @ti.func
    def compute_normal_soft(self, i):
        a, b, c = self.tris_soft[i,
                                 0], self.tris_soft[i, 1], self.tris_soft[i, 2]
        xab = self.x_soft[b] - self.x_soft[a]
        xac = self.x_soft[c] - self.x_soft[a]
        return xab.cross(xac).normalized()

    @ti.kernel
    def init_field(self):
        for i in ti.ndrange(self.n_soft_verts):
            self.v_soft[i] = ti.Vector([0, 0, 0], T)
            # self.v_rigid[i] = ti.Vector.zero(T, 3)
            self.C_soft[i] = ti.Matrix.zero(T, 3, 3)

        for i in range(self.n_soft_tris):
            ds = self.compute_T_soft(i)
            ds0 = ti.Vector([ds[0, 0], ds[1, 0], ds[2, 0]])
            ds1 = ti.Vector([ds[0, 1], ds[1, 1], ds[2, 1]])
            ds0_norm = ds0.norm()
            IB = ti.Matrix([
                [ds0_norm, ds0.dot(ds1) / ds0_norm],
                [0, ds0.cross(ds1).norm() / ds0_norm]
            ]).inverse()
            if ti.math.isnan(IB).sum():
                print('[nan detected during IB computation]')
                IB = ti.Matrix.zero(T, 2, 2)
            self.restInvT_soft[i] = IB
            self.tris_area_soft[i] = self.compute_area_soft(i)
            self.nrm_soft[i] = self.compute_normal_soft(i)

        for bi in range(self.n_soft_bends):
            face_inds = self.bending_faces[bi]
            n0 = self.compute_normal_soft(face_inds[0])
            n1 = self.compute_normal_soft(face_inds[1])
            theta = ti.acos(n0.dot(n1))
            theta = ti.max(theta, ti.abs(self.eps))
            edge_inds = self.bending_edges[bi]
            edge = (self.x_soft[edge_inds[1]] -
                    self.x_soft[edge_inds[0]]).normalized()
            sin_theta = n0.cross(n1).dot(edge)
            if sin_theta < 0:
                theta = - theta
            self.rest_bending_soft[bi] = theta

    def substep(self):
        # 网格复原
        self.grid_m.fill(0)
        self.grid_v.fill(0)
        self.energy_soft[None] = 0
        with ti.ad.Tape(self.energy_soft):
            self.compute_energy_soft()  # TODO
        # 物质点法MPM
        self.P2G()
        self.grid_op()
        self.G2P()

    def substep_1(self):
        self.grid_m.fill(0)
        self.grid_v.fill(0)
        self.energy_soft[None] = 0
        with ti.ad.Tape(self.energy_soft):
            self.compute_energy_soft()  # TODO
        self.P2G()
        self.grid_op()
        self.G2P_1()

    # 粒子传输信息到网格
    @ti.kernel
    def P2G(self):
        # cnt = 0
        for p in self.x_soft:
            # games 201 lec 7 19:01
            # base: 一个三维向量
            # (-23?) / (1) - 0.5 == -24?
            # 传递给所属网格和周围网格
            base = ti.cast(self.x_soft[p] * self.inv_dx - 0.5, ti.i32)
            # if not cnt:
            #     cnt = 1
            #     print(base, self.x_soft[p])

            fx = self.x_soft[p] * self.inv_dx - ti.cast(base, float)
            # 二次衰减函数，权重
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1)
                 ** 2, 0.5 * (fx - 0.5) ** 2]
            affine = self.p_mass * self.C_soft[p]
            # 求 3 * 3 * 3 范围内节点的坐标
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                if not ti.math.isnan(self.x_soft.grad[p]).sum():
                    # 物理量（速度），带权重
                    self.grid_v[base + offset] += weight * (
                        self.p_mass * self.v_soft[p] - self.dt * self.x_soft.grad[p] + affine @ dpos)
                    # 权重的和
                    self.grid_m[base + offset] += weight * self.p_mass

        # for p in self.x_rigid:
        #     base = ti.cast(self.x_rigid[p] * self.inv_dx - 0.5, ti.i32)
        #     fx = self.x_rigid[p] * self.inv_dx - ti.cast(base, float)
        #     w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1)
        #          ** 2, 0.5 * (fx - 0.5) ** 2]
        #     for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
        #         offset = ti.Vector([i, j, k])
        #         dpos = (offset.cast(float) - fx) * self.dx
        #         weight = w[i][0] * w[j][1] * w[k][2]
        #         self.grid_v[base + offset] += weight * \
        #             self.rp_mass * self.v_rigid[p]
        #         self.grid_m[base + offset] += weight * self.rp_mass

    # grid normalization
    @ti.kernel
    def grid_op(self):
        for i, j, k in self.grid_m:
            # 对于所有权重大于零的例子（这个 201 lec7 里说的）
            # 30 / 128 左右的网格会将收到的速度反弹
            if self.grid_m[i, j, k] > 0:
                inv_m = 1 / self.grid_m[i, j, k]
                # 速度加权平均
                self.grid_v[i, j, k] = inv_m * self.grid_v[i, j, k]
                self.grid_v[i, j, k].y -= self.dt * self.gravity
                if i < self.bound and self.grid_v[i, j, k].x < 0:
                    self.grid_v[i, j, k].x *= -0.5
                if i > self.n_grids - self.bound and self.grid_v[i, j, k].x > 0:
                    self.grid_v[i, j, k].x *= -0.5
                if j < self.bound and self.grid_v[i, j, k].y < 0:
                    self.grid_v[i, j, k].y *= -0.001
                    self.grid_v[i, j, k].x *= 0.99
                    self.grid_v[i, j, k].z *= 0.99
                if j > self.n_grids - self.bound and self.grid_v[i, j, k].y > 0:
                    self.grid_v[i, j, k].y *= -0.5
                if k < self.bound and self.grid_v[i, j, k].z < 0:
                    self.grid_v[i, j, k].z *= -0.5
                if k > self.n_grids - self.bound and self.grid_v[i, j, k].z > 0:
                    self.grid_v[i, j, k].z *= -0.5

    @ti.kernel
    def G2P(self):
        # 以粒子为主体
        for p in self.x_soft:
            # 每一个粒子把周围节点上的速度收集一下
            base = ti.cast(self.x_soft[p] * self.inv_dx - 0.5, ti.i32)
            fx = self.x_soft[p] * self.inv_dx - float(base)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0)
                 ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(T, 3)
            new_C = ti.Matrix.zero(T, 3, 3)

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                dpos = ti.Vector([i, j, k]).cast(float) - fx
                g_v = self.grid_v[base + ti.Vector([i, j, k])]
                weight = w[i][0] * w[j][1] * w[k][2]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            self.v_soft[p], self.C_soft[p] = new_v, new_C
            # 显式欧拉法
            self.x_soft[p] += self.dt * self.v_soft[p]  # advection

        # for p in self.x_rigid:
        #     self.x_rigid[p] += self.dt * self.v_rigid[p]

    @ti.kernel
    def G2P_1(self):
        for p in self.x_soft:
            base = ti.cast(self.x_soft[p] * self.inv_dx - 0.5, ti.i32)
            fx = self.x_soft[p] * self.inv_dx - float(base)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0)
                 ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(T, 3)
            new_C = ti.Matrix.zero(T, 3, 3)

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                dpos = ti.Vector([i, j, k]).cast(float) - fx
                g_v = self.grid_v[base + ti.Vector([i, j, k])]
                weight = w[i][0] * w[j][1] * w[k][2]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            self.v_soft[p], self.C_soft[p] = new_v, new_C
            self.x_soft[p] += self.dt * self.v_soft[p]  # advection
        lift_up = [0.0, 0.3, 0.0]
        self.v_soft[100] += lift_up
        self.v_soft[90] += lift_up
        self.v_soft[91] += lift_up
        self.v_soft[92] += lift_up
        self.v_soft[103] += lift_up
        self.v_soft[104] += lift_up

    # reference: https://github.com/zenustech/zeno/blob/master/projects/CuLagrange/fem/Generation.cpp

    @ti.kernel
    def compute_energy_soft(self):
        for i in range(self.n_soft_tris):
            Ds = self.compute_T_soft(i)
            F = Ds @ self.restInvT_soft[i]
            f0 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]])
            f1 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]])
            Estretch = self.mu * self.tris_area_soft[i] * \
                ((f0.norm() - 1) ** 2 + (f1.norm() - 1) ** 2)
            Eshear = self.mu * 0.3 * self.tris_area_soft[i] * f0.dot(f1) ** 2
            self.energy_soft[None] += Eshear + Estretch

        # bending
        for bi in range(self.n_soft_bends):
            face_inds = self.bending_faces[bi]
            n0 = self.compute_normal_soft(face_inds[0])
            n1 = self.compute_normal_soft(face_inds[1])
            theta = ti.acos(n0.dot(n1))
            theta = ti.max(theta, ti.abs(self.eps))
            edge_inds = self.bending_edges[bi]
            edge = (self.x_soft[edge_inds[1]] -
                    self.x_soft[edge_inds[0]]).normalized()
            sin_theta = n0.cross(n1).dot(edge)
            if sin_theta < 0:
                theta = - theta
            area = 0.5 * \
                (self.tris_area_soft[face_inds[0]] +
                 self.tris_area_soft[face_inds[1]])
            self.energy_soft[None] += (theta - self.rest_bending_soft[bi]
                                       ) ** 2 * area * 0.3 * self.mu * self.bending_p

            theta = self.plastic_yield_bending(theta)
            self.energy_soft[None] += (theta - self.rest_bending_soft[bi]
                                       ) ** 2 * area * 0.3 * self.mu

    def plastic_yield_bending(self, theta):
        yield_angle = 0.001  # to be adjusted
        if abs(theta) > yield_angle:
            theta_plastic = yield_angle if theta > 0 else -yield_angle
        else:
            theta_plastic = theta
        return theta_plastic

    # def add_kinematic_rigid(self, body: RigidBody):
    #     self.rigid_bodies.append(body)
    #     self.n_rigid_pars += body.n_pars
    #     # check boundary
    #     pos_mask = (body.rest_pos - self.origin) < 0
    #     pos_mask *= (body.rest_pos - self.origin) > 1
    #     if pos_mask.sum() > 0:
    #         print(
    #             'MpmLagSim: kinematic rigid body trying to be added is out of the bounding box!')

    def set_soft(self, body_mesh: trimesh.Trimesh):
        self.soft_mesh = body_mesh
        self.n_soft_verts = body_mesh.vertices.shape[0]
        self.n_soft_tris = body_mesh.faces.shape[0]
        pos_mask = (body_mesh.vertices - self.origin) < 0
        pos_mask *= (body_mesh.vertices - self.origin) > 1
        if pos_mask.sum() > 0:
            print(
                'MpmLagSim: soft body trying to be added is out of the bounding box!')

    # def toward_kinematic_target(self, substeps=10):
        # rigid_target = np.concatenate(
        #     [b.target_pos for b in self.rigid_bodies], axis=0) - self.origin
        # rigid_vel = (rigid_target - self.x_rigid.to_numpy()) / \
        #     (self.dt * substeps)
        # self.v_rigid.from_numpy(rigid_vel)

    def update_scene(self):
        self.scene.set_camera(self.camera)
        self.scene.ambient_light((0.8, 0.8, 0.8))
        self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

        self.scene.particles(self.x_soft, color=(
            0.68, 0.26, 0.19), radius=0.0005)
        # self.scene.particles(self.x_rigid, color=(
        #     0.19, 0.26, 0.68), radius=0.002)
        self.scene.lines(self.lines, 2)

    def show(self):
        self.canvas.scene(self.scene)
        self.window.show()

# NOTE: now soft only one support mesh
# reference: https://github.com/taichi-dev/taichi/blob/master/python/taichi/examples/simulation/mpm_lagrangian_forces.py


def test_mpm():
    ti.init(arch=ti.cuda)
    dt = 1e-4
    substeps = 10
    sim = MpmSim(origin=np.asarray([-0.5,] * 3), dt=dt)
    nhk = NeoHookean()
    cube: trimesh.Trimesh = trimesh.creation.box((0.1,) * 3)
    cube_points = cube.sample(8192)
    cube_pcd: trimesh.PointCloud = trimesh.PointCloud(cube.sample(8192))
    sponge_box = SoftBody(cube_points, nhk)
    wrist_mesh = trimesh.load_mesh('./data/Mano_URDF/meshes/m_avg_R_Wrist.stl')
    # pos = np.asarray([pos['z'], -pos['x'], pos['y']])
    wrist_verts = np.asarray(wrist_mesh.vertices)
    wrist_mesh.vertices = np.concatenate(
        [wrist_verts[:, [2]], -wrist_verts[:, [0]], wrist_verts[:, [1]]], axis=1)
    wrist_mesh.apply_translation(np.asarray([0., 0.2, 0.15]))
    rigid_wrist = RigidBody(wrist_mesh)

    sim.add_body(sponge_box)
    sim.add_body(rigid_wrist)
    sim.init_system()

    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        # wrist_mesh.apply_translation(np.asarray([0., -0.001, 0.]))
        rigid_wrist.set_target(wrist_mesh.vertices)
        for s in range(substeps):
            sim.substep()
        sim.update_scene()
        sim.show()
        sim.toward_target(substeps)


def test_lag_mpm():
    ti.init(arch=ti.cuda)
    dt = 1e-4
    sim = MpmLagSim(origin=np.asarray([-0.5,] * 3), dt=dt)
    box_mesh = trimesh.load_mesh('./data/plasticbag2.obj')
    # wrist_mesh = trimesh.load_mesh('./data/Mano_URDF/meshes/m_avg_R_Wrist.stl')
    # wrist_verts = np.asarray(wrist_mesh.vertices)
    # wrist_mesh.vertices = np.concatenate(
    #     [wrist_verts[:, [2]], -wrist_verts[:, [0]], wrist_verts[:, [1]]], axis=1)
    # wrist_mesh.apply_translation(np.asarray([0., 0.2, 0.15]))
    # rigid_wrist = RigidBody(wrist_mesh)

    # sim.add_kinematic_rigid(rigid_wrist)
    sim.set_soft(box_mesh)
    sim.init_sim()

    while not sim.window.is_pressed(ti.GUI.ESCAPE):
        # wrist_mesh.apply_translation(np.asarray([0., -0.001, 0.]))
        # rigid_wrist.set_target(wrist_mesh.vertices)
        sim.substep()
        sim.update_scene()
        sim.show()
        # sim.toward_kinematic_target()


if __name__ == '__main__':
    # test_mpm()
    test_lag_mpm()
