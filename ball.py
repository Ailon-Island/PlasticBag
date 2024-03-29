import taichi as ti
import taichi.math as tm
import numpy as np


@ti.data_oriented
class Ball:
    def __init__(self, radius):
        ...
        self.radius = radius
        self.pos = ti.Vector([0.4, 0.4, 0.4])
        self.selected = False
        self.last_mouse_pressed = False

    def ray_dis(self, ray_origin, ray_dir) -> ti.f32:
        v = self.pos - ray_origin
        # print(f"v dir: {v.normalized()}")
        perp = v - v.dot(ray_dir) * ray_dir
        return perp.norm()

    def ray_hit(self, ray_origin, ray_dir) -> bool:
        dis = self.ray_dis(ray_origin, ray_dir)
        # print(f"ray_dis: {dis}")
        return dis <= self.radius

    def mouse_ray(self, win: ti.ui.Window, cam: ti.ui.Camera) -> tuple[ti.Vector, ti.Vector]:
        ...
        # calculate ray from mouse
        ndc_pos = np.array(win.get_cursor_pos()) * 2 - 1
        # ndc_pos = tm.vec2(2 * mouse_pos - 1, 1 - 2 * mouse_pos)
        res = win.get_window_shape()
        inv_cam_mat = np.linalg.inv(
            cam.get_view_matrix() @ cam.get_projection_matrix(res[0] / res[1]))
        ray_ndc = np.array([ndc_pos[0], ndc_pos[1], -1, 1])
        ray_world = ray_ndc @ inv_cam_mat
        ray_world /= ray_world[-1]
        ray_origin = cam.curr_position.to_numpy()
        ray_dir = ray_world[:-1] - ray_origin
        ray_dir /= np.linalg.norm(ray_dir)
        ray_origin = ti.Vector(ray_origin)
        ray_dir = ti.Vector(ray_dir)
        return ray_origin, ray_dir

    def mouse_ray_hit(self, win: ti.ui.Window, cam: ti.ui.Camera) -> bool:
        return self.ray_hit(*self.mouse_ray(win, cam))

    def update(self, win: ti.ui.Window, cam: ti.ui.Camera):
        ...
        # check if mouse is clicked
        # 需要检查在至少一个 ball 选中的情况下，不移动镜头
        mouse_pressed = win.is_pressed(ti.ui.LMB)
        if mouse_pressed:
            if not self.last_mouse_pressed and self.mouse_ray_hit(win, cam):
                if not self.selected:
                    self.selected = True
                    print('Selected')
        else:
            if self.selected:
                self.selected = False
                print('Unselected')
        self.last_mouse_pressed = mouse_pressed
        # next, calculate a ray(position and direction) from mouse
