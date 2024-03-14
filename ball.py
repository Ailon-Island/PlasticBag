import taichi as ti
import taichi.math as tm


@ti.data_oriented
class Ball:
    def __init__(self, radius):
        ...
        self.radius = radius
        self.pos = ti.Vector([0.4, 0.4, 0.4])
        self.selected = False
        self.last_mouse_pressed = False

    def ray_dis(self, ray_origin, ray_dir) -> ti.f32:
        proj = (self.pos - ray_origin).dot(ray_dir)
        proj_pos = ray_origin + proj * ray_dir
        distance = (self.pos - proj_pos).norm()
        return distance

    def ray_hit(self, ray_origin, ray_dir) -> bool:
        return self.ray_dis(ray_origin, ray_dir) <= self.radius

    def mouse_ray(self, win: ti.ui.Window, cam: ti.ui.Camera) -> tuple[ti.Vector, ti.Vector]:
        ...
        # calculate ray from mouse
        mouse_pos = win.get_cursor_pos()
        ndc_pos = tm.vec2(2 * mouse_pos - 1, 1 - 2 * mouse_pos)
        inv_view_proj = tm.inverse(
            cam.get_view_matrix() * cam.get_projection_matrix)
        ray_ndc = tm.vec4(ndc_pos, -1, 1)
        ray_world = inv_view_proj @ ray_ndc
        ray_world /= ray_world.w
        ray_origin = cam.position
        ray_dir = tm.normalize(ray_world.xyz - ray_origin)
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
