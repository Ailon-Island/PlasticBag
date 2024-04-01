import taichi as ti
import taichi.math as tm
import numpy as np

from utils import get_mouse_ray

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

    def mouse_ray_hit(self, win: ti.ui.Window, cam: ti.ui.Camera) -> tuple[ti.f32, bool]:
        return self.ray_hit(*get_mouse_ray(win, cam))

    def drag(self, win: ti.ui.Window, cam: ti.ui.Camera, sim: 'MpmLagSim'):
        # check if mouse is clicked
        # 需要检查在至少一个 ball 选中的情况下，不移动镜头
        mouse_pressed = win.is_pressed(ti.ui.LMB)
        if mouse_pressed:
            origin, dir = get_mouse_ray(win, cam)
            if not self.last_mouse_pressed:
                hit_depth, hit = self.ray_hit(origin, dir)
                if hit:
                    if not self.selected:
                        self.selected = True
                        sim.object_selected[None] = 1
                        print('Selected')
                        self.drag_depth = hit_depth
                        self.drag_offset = origin + dir * hit_depth - self.pos
            elif self.selected:
                drag_tgt = origin + dir * self.drag_depth
                new_pos = drag_tgt - self.drag_offset
                self.vel = (new_pos - self.pos) / sim.dt
                self.pos = new_pos
        else:
            if self.selected:
                self.selected = False
                sim.object_selected[None] = 0
                self.vel = ti.Vector([0.0, 0.0, 0.0])
                print('Unselected')
        self.last_mouse_pressed = mouse_pressed
        # next, calculate a ray(position and direction) from mouse
