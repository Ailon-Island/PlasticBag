import taichi as ti
import taichi.math as tm
import numpy as np


@ti.data_oriented
class Ball:
    def __init__(self, radius):
        ...
        self.radius = radius
        self.pos = ti.Vector([0.4, 0.4, 0.4])
        self.vel = ti.Vector([0.0, 0.0, 0.0])
        self.selected = False
        self.drag_offset = ti.Vector([0.0, 0.0, 0.0])
        self.drag_depth = 0
        self.last_mouse_pressed = False

    def ray_hit(self, ray_origin, ray_dir) -> tuple[ti.f32, bool]:
        line = self.pos - ray_origin
        proj = line.dot(ray_dir)
        if proj < 0:
            return -1, False
        perp = line - proj * ray_dir
        dist = perp.norm()
        hit = dist <= self.radius
        if not hit:
            return -1, False
        return line.norm() - ti.sqrt(self.radius ** 2 - dist ** 2), True

    def mouse_ray(self, win: ti.ui.Window, cam: ti.ui.Camera) -> tuple[ti.Vector, ti.Vector]:
        ...
        # calculate ray from mouse
        ndc_pos = np.array(win.get_cursor_pos()) * 2 - 1
        res = win.get_window_shape()
        inv_cam_mat = np.linalg.inv(
            cam.get_view_matrix() @ cam.get_projection_matrix(res[0] / res[1]))
        ray_ndc = np.array([ndc_pos[0], ndc_pos[1], 1, 1]) # z = -1 or 1?
        ray_world = ray_ndc @ inv_cam_mat
        ray_world /= ray_world[-1]
        ray_origin = cam.curr_position.to_numpy()
        ray_dir = ray_world[:-1] - ray_origin
        ray_dir /= np.linalg.norm(ray_dir)
        ray_origin = ti.Vector(ray_origin)
        ray_dir = ti.Vector(ray_dir)
        return ray_origin, ray_dir

    def mouse_ray_hit(self, win: ti.ui.Window, cam: ti.ui.Camera) -> tuple[ti.f32, bool]:
        return self.ray_hit(*self.mouse_ray(win, cam))

    def update(self, win: ti.ui.Window, cam: ti.ui.Camera, sim: 'MpmLagSim'):
        ...
        # check if mouse is clicked
        # 需要检查在至少一个 ball 选中的情况下，不移动镜头
        mouse_pressed = win.is_pressed(ti.ui.LMB)
        if mouse_pressed:
            if not self.last_mouse_pressed:
                origin, dir = self.mouse_ray(win, cam)
                hit_depth, hit = self.ray_hit(origin, dir)
                if hit:
                    if not self.selected:
                        self.selected = True
                        sim.object_selected = 1
                        print('Selected')
                        self.drag_depth = hit_depth
                        self.drag_offset = origin + dir * hit_depth - self.pos
            elif self.selected:
                origin, dir = self.mouse_ray(win, cam)
                drag_tgt = origin + dir * self.drag_depth
                new_pos = drag_tgt - self.drag_offset
                self.vel = (new_pos - self.pos) / sim.dt
                self.pos = new_pos
        else:
            if self.selected:
                self.selected = False
                sim.object_selected = 0
                self.vel = ti.Vector([0.0, 0.0, 0.0])
                print('Unselected')
        self.last_mouse_pressed = mouse_pressed
        # next, calculate a ray(position and direction) from mouse
