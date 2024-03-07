from os.path import join as pjoin
import time
import trimesh
import taichi as ti
import numpy as np
from utils import mat3, scalars, vecs, mats, T, TetMesh
import utils
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
    def __init__(self, dt: float = 8e-5,
                 origin: np.ndarray = np.asarray([-0.5, -0.4, -0.5]),
                 n_grids: int = 180,
                 scale: float = 1.0) -> None:
        # 压缩按键
        self.compress_force = scalars(ti.i32, shape=())
        # 抬升按键
        self.lift_up_is_on = scalars(ti.i32, shape=())
        # [init param]
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
        # 这些在 taichi 中都是常数
        self.BOUND = 30
        self.GRAVITY = 9.8
        self.COMPRESS_ACC = 200
        # 韧劲系数
        self.BENDING_P = 9.0
        self.lift_state = 0
        self.frame_cnt = 0
        self.theta_cnt = scalars(ti.i32, ())
        self.MAX_FOLD_ANGLE = 160
        self.clear_bodies()

        self.TO_LIFT = scalars(ti.i32, shape=(6))
        self.TO_LIFT.from_numpy(np.array([1155, 1156, 1157, 2500, 2501, 2502]))

        # E 早先采用 1e1，很软
        # 调大 E 可以变硬
        E, nu = 3e1, 0.49
        # 这什么系数啊
        self.mu, self.lam = E / (2 * (1 + nu)), E * \
            nu / ((1 + nu) * (1 - 2 * nu))
        self.eps = 1e-6

        self.window = ti.ui.Window("CPIC-Scene", (900, 1200))
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
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
        # 前面进行 set_soft()
        # TODO: align these parameters in the future
        # what is p? oh particle
        # 这里怎么是网格体积啊，不对吧
        self.p_vol, self.p_rho = (1 / 180 * 0.5) ** 2 * \
            5e3 / self.n_soft_verts, 0.1
        self.p_mass = self.p_vol * self.p_rho
        # rigid part
        self.rp_rho = 1
        self.rp_mass = self.p_vol * self.rp_rho

        self.axis_particles = vecs(3, T, 4)
        self.axis_particles.from_numpy(
            np.array(
                [
                    [0, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            )
        )
        self.axis_particles_color = vecs(3, ti.f32, 4)
        self.axis_particles_color.from_numpy(
            np.array(
                [
                    [1, 1, 1],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            )
        )

        self.balls_cnt = 2
        self.balls_x = vecs(3, T, self.balls_cnt, needs_grad=True)
        self.balls_x.from_numpy(np.array(
            [
                [0.5, 0.4, 0.5],
                [0.3, 0.3, 0.3],
            ]
        )
        )
        self.balls_v = vecs(3, T, self.balls_cnt)
        self.balls_v.from_numpy(np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )
        )
        self.ball_radius = 0.03

        self.x_soft = vecs(3, T, self.n_soft_verts, needs_grad=True)
        self.v_soft = vecs(3, T, self.n_soft_verts)
        # 好像是上一帧求出来的什么玩意
        # ref: lec7 29:42
        self.C_soft = mats(3, 3, T, self.n_soft_verts)
        # i 号三角形专属的神奇矩阵
        self.restInvT_soft = mats(2, 2, T, self.n_soft_tris)  # 数量 = 三角形数量
        self.soft_energy = scalars(T, shape=(), needs_grad=True)
        self.nrm_soft = vecs(3, T, self.n_soft_tris)
        # i 号三角形的面积
        self.tris_area_soft = scalars(float, (self.n_soft_tris,))
        # self.x_rigid = vecs(3, T, self.n_rigid_pars)
        # self.v_rigid = vecs(3, T, self.n_rigid_pars)

        self.grid_v = vecs(3, T, (self.n_grids, self.n_grids, self.n_grids))
        self.grid_m = scalars(T, (self.n_grids, self.n_grids, self.n_grids))

        self.x_soft.from_numpy(np.asarray(
            self.soft_mesh.vertices) - self.origin)  # 这里是减 origin

        self.init_center_x = ti.Vector.field(3, ti.f32, ())

        self.tris_soft = scalars(int, (self.n_soft_tris, 3))
        self.tris_soft.from_numpy(np.asarray(self.soft_mesh.faces))

        self.tris_soft_expanded = scalars(ti.i32, self.n_soft_tris * 3)
        for i, j in ti.ndrange(self.n_soft_tris, 3):
            self.tris_soft_expanded[i * 3 + j] = self.tris_soft[i, j]

        soft_face_adjacency = self.soft_mesh.face_adjacency
        self.n_soft_bends = soft_face_adjacency.shape[0]
        self.bending_faces = vecs(2, int, self.n_soft_bends)
        # 原始情况下的弯角度
        self.rest_bending_soft = scalars(T, shape=(self.n_soft_bends,))
        self.rest_bending_sheared = scalars(T, shape=(self.n_soft_bends,))
        # 类似于 (序号 1, 序号 2) 表示这两个下表的三角形相邻（估计） 三角形.下标.tuple
        self.bending_faces.from_numpy(soft_face_adjacency)
        # soft_mesh 用 trimesh 库加载的一个文件，加载时属性都有了
        # 这里是点.下标.tuple
        self.bending_edges = vecs(2, int, self.n_soft_bends)
        self.bending_edges.from_numpy(self.soft_mesh.face_adjacency_edges)

        self.edge_fold_coef = scalars(ti.f32, shape=(self.n_soft_bends,))

        self.verts_linked_edges = [[] for _ in range(self.n_soft_verts)]
        for i, j in ti.ndrange(self.n_soft_bends, 2):
            self.verts_linked_edges[self.bending_edges[i][j]].append(i)

        self.bending_edges_expanded = scalars(ti.i32, self.n_soft_bends * 2)
        for i, j in ti.ndrange(self.n_soft_bends, 2):
            self.bending_edges_expanded[i * 2 + j] = self.bending_edges[i][j]

        self.soft_verts_color = vecs(3, ti.f32, self.n_soft_verts)
        self.verts_linked_to_how_many_tris = scalars(int, self.n_soft_verts)
        for i, j in ti.ndrange(self.n_soft_tris, 3):
            self.verts_linked_to_how_many_tris[self.tris_soft[i, j]] += 1
        # print('#1', self.n_soft_verts, self.x_soft.shape, self.tris_soft.shape)
        # #1 4997 (4997,) (9997, 3)
        # x_rigid = np.concatenate(
        # [b.rest_pos for b in self.rigid_bodies], axis=0) - self.origin
        # self.x_rigid.from_numpy(x_rigid)
        self.init_field()

        # 地面显示
        # self.lines = ti.Vector.field(3, ti.f32, 8)
        height = 29 / self.n_grids
        size = 1
        # self.lines[0] = [0, height, 0]
        # self.lines[1] = [+size, height, 0]
        # self.lines[2] = [+size, height, 0]
        # self.lines[3] = [+size, height, +size]
        # self.lines[4] = [+size, height, +size]
        # self.lines[5] = [0, height, +size]
        # self.lines[6] = [0, height, +size]
        # self.lines[7] = [0, height, 0]

        self.plane = vecs(3, ti.f32, 4)
        self.plane.from_numpy(np.array(
            [
                [0, height, 0],
                [size, height, 0],
                [size, height, size],
                [0, height, size]
            ]
        ))
        self.plane_vertices_color = vecs(3, ti.f32, 4)
        self.plane_vertices_color.from_numpy(np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
            ]
        ))

        self.plane_triangles = vecs(1, ti.i32, 6)
        self.plane_triangles.from_numpy(np.array(
            [
                [0],
                [1],
                [2],
                [0],
                [3],
                [2],
            ]
        ))

    @ti.func
    def compute_T_soft(self, i):
        # what's this 三角..
        a, b, c = \
            self.tris_soft[i, 0], self.tris_soft[i, 1], self.tris_soft[i, 2]
        xab = self.x_soft[b] - self.x_soft[a]
        xac = self.x_soft[c] - self.x_soft[a]
        # 第一列是 a -> b 这条边
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

    # i 号三角形的法线向量
    @ti.func
    def compute_normal_soft(self, i):
        a, b, c = \
            self.tris_soft[i, 0], self.tris_soft[i, 1], self.tris_soft[i, 2]
        xab = self.x_soft[b] - self.x_soft[a]
        xac = self.x_soft[c] - self.x_soft[a]
        return xab.cross(xac).normalized()

    @ti.kernel
    def init_field(self):
        self.rest_bending_sheared.fill(1)
        for i in self.x_soft:
            self.init_center_x[None] += self.x_soft[i]
        self.init_center_x[None] /= self.n_soft_verts
        for i in ti.ndrange(self.n_soft_verts):
            self.v_soft[i] = ti.Vector([0, 0, 0], T)
            # self.v_rigid[i] = ti.Vector.zero(T, 3)
            self.C_soft[i] = ti.Matrix.zero(T, 3, 3)

        for i in range(self.n_soft_tris):
            ds = self.compute_T_soft(i)  # i 号三角形的各条边
            ds0 = ti.Vector([ds[0, 0], ds[1, 0], ds[2, 0]])
            ds1 = ti.Vector([ds[0, 1], ds[1, 1], ds[2, 1]])
            ds0_norm = ds0.norm()
            # 神奇矩阵的逆
            IB = ti.Matrix([
                [ds0_norm, ds0.dot(ds1) / ds0_norm],
                [0, ds0.cross(ds1).norm() / ds0_norm]
            ]).inverse()
            if ti.math.isnan(IB).sum():
                print('[nan detected during IB computation]')
                IB = ti.Matrix.zero(T, 2, 2)
            # i 号三角形专属的神奇矩阵
            self.restInvT_soft[i] = IB
            self.tris_area_soft[i] = self.compute_area_soft(i)
            # i 号三角形的朝向 (ab 为 x 轴，ac 为 y 轴情况下的 z 轴方向)
            self.nrm_soft[i] = self.compute_normal_soft(i)

        # 对折线
        for bi in range(self.n_soft_bends):
            # 果然是相邻三角形的下表吼
            face_inds = self.bending_faces[bi]
            n0 = self.compute_normal_soft(face_inds[0])
            n1 = self.compute_normal_soft(face_inds[1])
            theta = ti.acos(n0.dot(n1))
            theta = ti.max(theta, ti.abs(self.eps))
            edge_inds = self.bending_edges[bi]
            # 边向量
            edge = (self.x_soft[edge_inds[1]] -
                    self.x_soft[edge_inds[0]]).normalized()
            sin_theta = n0.cross(n1).dot(edge)
            if sin_theta < 0:
                theta = - theta
            # 存了一个 ((n0 cross n1) dot edge).abs
            # 这里 n0 cross n1 和 edge 总是平行的
            # 韧性系数呢？
            self.rest_bending_soft[bi] = theta

    def substep(self):
        # 网格复原
        self.grid_m.fill(0)
        self.grid_v.fill(0)
        self.edge_fold_coef.fill(0)
        self.soft_energy[None] = 0
        self.soft_verts_color.fill(0.5)
        self.ui_check()
        with ti.ad.Tape(self.soft_energy):
            # 这玩意为什么执行了两次
            self.compute_tris_energy()
            self.compute_bending_energy()
        self.compute_color()
        # 物质点
        self.p2g()
        self.grid_op()
        self.g2p()

    # 粒子传输信息到网格
    @ti.kernel
    def p2g(self):

        # cnt = 0
        for p in self.x_soft:
            # p 是一个下标...
            # ref: games 201 lec 7 19:01
            # base: 一个三维向量表示下标，全是正的
            # 传递给所属网格和周围网格
            # x_soft: 50.1, 50.5, 50.8
            # dx = inv_dx = 1.0
            # base = 49, 50, 50
            base = ti.cast(self.x_soft[p] * self.inv_dx - 0.5, ti.i32)
            # if not cnt:
            #     cnt = 1
            #     print(base, self.x_soft[p])

            # fx = 1.1, 0.5, 0.8
            fx = self.x_soft[p] * self.inv_dx - ti.cast(base, float)
            # 二次衰减函数，权重
            # 50.1 - 49, 50, 51: w = [[x: 0.08..], [x: 0.74], [x: 0.18] ]
            # 50.5 - 50, 51, 52: w = [[x: 0.5..], [x: 0.5], [x: 0] ] 神奇
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1)
                 ** 2, 0.5 * (fx - 0.5) ** 2]
            # lec 7 也有这个
            # 这是用来解决 g (27 dof) -> p (3 dof) 信息丢失问题的
            # pic 对速度场进行平均化，dilation 运动由于 grid 速度不一样（比如一正一反），所以会趋于 0
            # affine 速度场
            affine = self.p_mass * self.C_soft[p]
            # for j in ti.static(ti.ndrange(3)):
            #     self.soft_verts_color[p][j] += max(-0.5, min(
            #         0.5, self.x_soft.grad[p][j] / 0.003 * 0.5))
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                # (1, 1, 1) - (1.1, 0.5, 0.8) =  (-0.1, 0.5, 0.2)
                dpos = (offset.cast(float) - fx) * \
                    self.dx  # ijk 格点相对 x_soft 的坐标
                # x, y, z 分量上的权重乘起来
                # i: 0, 1, 2  j: 0, 1, 2
                weight = w[i][0] * w[j][1] * w[k][2]
                if not ti.math.isnan(self.x_soft.grad[p]).sum():
                    # 物理量（速度），带权重
                    # wt hell is x_soft.grad?
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
    def ui_check(self):
        if self.window.is_pressed("z"):
            self.compress_force[None] = 1
        elif self.window.is_pressed("x"):
            self.compress_force[None] = -1
        else:
            self.compress_force[None] = 0
        if self.window.is_pressed(ti.GUI.UP):
            self.lift_up_is_on[None] = 1
        else:
            self.lift_up_is_on[None] = 0

    # grid normalization
    @ti.kernel
    def grid_op(self):
        center_x = ti.Vector.zero(T, 3)
        for i in self.x_soft:
            center_x += self.x_soft[i]
        center_x /= self.n_soft_verts
        # center_x = self.init_center_x[None]

        for i, j, k in self.grid_m:
            # 对于所有权重大于零的例子（这个 201 lec7 里说的）
            # 30 / 128 左右的网格会将收到的速度反弹
            if self.grid_m[i, j, k] > 0:
                inv_m = 1 / self.grid_m[i, j, k]
                # 速度加权平均
                self.grid_v[i, j, k] = inv_m * self.grid_v[i, j, k]
                self.grid_v[i, j, k].y -= self.dt * self.GRAVITY
                # 猜的
                if self.compress_force[None] != 0:
                    grid_x = ti.Vector([i, j, k]) * self.dx
                    # 最终目标是有折叠性（没有什么延展性，类似于折叠门！）
                    compress_dir = (center_x - grid_x).normalized() * \
                        self.compress_force[None]
                    self.grid_v[i, j, k] += self.COMPRESS_ACC * \
                        self.dt * compress_dir
                if i < self.BOUND and self.grid_v[i, j, k].x < 0:
                    self.grid_v[i, j, k].x *= -0.5
                if i > self.n_grids - self.BOUND and self.grid_v[i, j, k].x > 0:
                    self.grid_v[i, j, k].x *= -0.5
                if j < self.BOUND and self.grid_v[i, j, k].y < 0:
                    self.grid_v[i, j, k].y *= -0.001
                    self.grid_v[i, j, k].x *= 0.99
                    self.grid_v[i, j, k].z *= 0.99
                if j > self.n_grids - self.BOUND and self.grid_v[i, j, k].y > 0:
                    self.grid_v[i, j, k].y *= -0.5
                if k < self.BOUND and self.grid_v[i, j, k].z < 0:
                    self.grid_v[i, j, k].z *= -0.5
                if k > self.n_grids - self.BOUND and self.grid_v[i, j, k].z > 0:
                    self.grid_v[i, j, k].z *= -0.5

    @ti.kernel
    def g2p(self):
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
                # ref: lec 7 35:42
                # C: 速度的梯度
                new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

            self.v_soft[p], self.C_soft[p] = new_v, new_C
            self.x_soft[p] += self.dt * self.v_soft[p]  # advection

            # J[p] *= 1 + dt * new_C.trace()
        for p in self.balls_x:
            self.balls_x[p] += self.dt * self.balls_v[p]

        if self.lift_up_is_on[None] == 1:
            lift_up = ti.Vector([0.0, 1.3, 0.0])
            for k in range(self.TO_LIFT.shape[0]):
                i = self.TO_LIFT[k]
                self.v_soft[i] += lift_up
                self.highlight_vertex(i)

        # for p in self.x_rigid:
        #     self.x_rigid[p] += self.dt * self.v_rigid[p]

    @ti.kernel
    def compute_tris_energy(self):
        # get deformation gradient?
        # 求了一个 grad
        for i in range(self.n_soft_tris):
            # i 号三角形的各条边
            Ds = self.compute_T_soft(i)
            F = Ds @ self.restInvT_soft[i]
            f0 = ti.Vector([F[0, 0], F[1, 0], F[2, 0]])
            f1 = ti.Vector([F[0, 1], F[1, 1], F[2, 1]])
            Estretch = self.mu * self.tris_area_soft[i] * \
                ((f0.norm() - 1) ** 2 + (f1.norm() - 1) ** 2)
            # 压力红色，张力蓝色，默认灰色
            # 剪切力，断裂力
            Eshear = self.mu * 0.3 * self.tris_area_soft[i] * f0.dot(f1) ** 2

            # if i == 0:
            #     print('#2', Estretch, Eshear)
            # 通常是 0 并且不是 0 也很小
            # 抗面积变化
            self.soft_energy[None] += Eshear + Estretch

    @ti.kernel
    def compute_bending_energy(self):
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
            # bending, 抗弯折
            # 放松状态下
            # 这是带 bending_p 的
            # 可以改变舒服角度
            self.soft_energy[None] += (theta - self.rest_bending_soft[bi]
                                       ) ** 2 * area * 0.3 * self.mu * self.BENDING_P * self.rest_bending_sheared[bi]
            # if abs(edge_inds[0] - edge_inds[1]) == 1 and abs(theta) > abs(self.rest_bending_soft[bi]) and abs(theta) < deg2rad(160):
            # todo 实验中 参数调整
            if abs(theta) > abs(self.rest_bending_soft[bi]) and abs(theta) < utils.deg2rad(self.MAX_FOLD_ANGLE):
                # 降低相邻折痕难度（向量方向相近者更可能）
                # 对于 edge_inds[0] [1] 的相邻边...
                # for i in range(2):
                #     for j in self.verts_linked_edges[edge_inds[i]]:
                #         edge_inds = self.bending_edges[j]
                #         edge = (self.x_soft[edge_inds[1]] -
                #                 self.x_soft[edge_inds[0]]).normalized()
                #         self.edge_fold_coef[j] += () ** 2
                # 最后使用加成 1 - exp(-x)

                self.rest_bending_soft[bi] += (theta -
                                               self.rest_bending_soft[bi]) * 0.5
                self.rest_bending_sheared[bi] = 1 + \
                    self.rest_bending_soft[bi] / utils.deg2rad(60)
                self.theta_cnt[None] += 1

            if abs(theta) < abs(self.rest_bending_soft[bi]) and abs(theta) < utils.deg2rad(self.MAX_FOLD_ANGLE):
                self.rest_bending_soft[bi] += (theta -
                                               self.rest_bending_soft[bi]) * 0.1
                self.rest_bending_sheared[bi] = 1 + \
                    self.rest_bending_soft[bi] / utils.deg2rad(60)
                self.theta_cnt[None] += 1

            # if 2000 <= edge_inds[0] <= 2500 and abs(edge_inds[0] - edge_inds[1]) == 1:
            #     self.rest_bending_soft[bi] = deg2rad(45)
            #     self.theta_cnt[None] += 1
            #     self.soft_energy[None] += (theta - self.rest_bending_soft[bi]
            #                                ) ** 2 * area * 0.3 * self.mu * self.bending_p * 10
            # else:
            #     self.soft_energy[None] += (theta - self.rest_bending_soft[bi]
            #                                ) ** 2 * area * 0.3 * self.mu * self.bending_p
            # print('bi:', bi, 'theta:', rad2deg(theta))

            # if bi < self.n_soft_bends // 2:
            #     self.rest_bending_soft[bi] = theta

            # theta = self.plastic_yield_bending(theta)
            #
            # self.energy_soft[None] += (theta - self.rest_bending_soft[bi]
            #                            ) ** 2 * area * 0.3 * self.mu

    @ti.kernel
    def compute_color(self):
        for i in range(self.n_soft_tris):
            area_coeff = (self.compute_area_soft(
                i) / self.tris_area_soft[i] - 1) / 2.0 * 10
            area_coeff = max(-0.5, min(0.5, area_coeff))
            # if i == 9:
            #     print('#1', area_coeff)
            # max_coe = max(max_coe, area_coeff)
            for j in range(3):
                self.soft_verts_color[self.tris_soft[i, j]
                                      ][0] += max(-area_coeff, 0) / self.verts_linked_to_how_many_tris[self.tris_soft[i, j]]
                self.soft_verts_color[self.tris_soft[i, j]
                                      ][2] += max(area_coeff, 0) / self.verts_linked_to_how_many_tris[self.tris_soft[i, j]]

    # 将输入角度的绝对值限制在一个范围内
    def plastic_yield_bending(self, theta):
        yield_angle = 0.001  # to be adjusted
        if abs(theta) > yield_angle:
            theta_plastic = yield_angle if theta > 0 else -yield_angle
        else:
            theta_plastic = theta
        return theta_plastic

    @ti.func
    def highlight_vertex(self, i):
        self.soft_verts_color[i][1] = 1.0

    # def add_kinematic_rigid(self, body: RigidBody):
    #     self.rigid_bodies.append(body)
    #     self.n_rigid_pars += body.n_pars
    #     # check boundary
    #     pos_mask = (body.rest_pos - self.origin) < 0
    #     pos_mask *= (body.rest_pos - self.origin) > 1
    #     if pos_mask.sum() > 0:
    #         print(
    #             'MpmLagSim: kinematic rigid body trying to be added is out of the bounding box!')

    # 主函数中调用

    def set_soft(self, body_mesh: trimesh.Trimesh):
        # 来自这里
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

        self.scene.particles(
            self.axis_particles, per_vertex_color=self.axis_particles_color, radius=0.05)
        self.scene.particles(self.balls_x, color=(
            0.68, 0.26, 0.19), radius=self.ball_radius)
        # self.scene.particles(self.x_soft, color=(
        #     0.68, 0.26, 0.19), radius=0.0005)
        # self.scene.particles(self.x_rigid, color=(
        #     0.19, 0.26, 0.68), radius=0.002)
        # print(self.x_soft.shape, self.bending_edges_expanded.shape)
        # self.scene.lines(self.x_soft, width=1,
        #                  indices=self.bending_edges_expanded)
        self.frame_cnt += 1
        self.soft_verts_color[self.frame_cnt % self.n_soft_verts][1] = 1.0
        self.scene.mesh(self.x_soft, self.tris_soft_expanded,
                        per_vertex_color=self.soft_verts_color)
        self.scene.mesh(self.plane, self.plane_triangles)
        with self.gui.sub_window("Debug", 0.05, 0.1, 0.2, 0.15) as w:
            w.text(
                text=f'Highlight vertex: {self.frame_cnt % self.n_soft_verts}')
            w.text(text=f'Theta cnt: {self.theta_cnt[None]}')
        # per_vertex_color=self.plane_vertices_color)

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
