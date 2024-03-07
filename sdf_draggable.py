import taichi as ti

ti.init(arch=ti.gpu)  # assuming you have a GPU


@ti.data_oriented
class SdfDraggable:
    def __init__(self):
        self.selecting_cnt = 0
        self.rot_speed = 120.0
        self.selecting_me = False
        self.mouse_move_v = ti.Vector([0.0, 0.0, 0.0])
        self.pressed = False
        self.offset = ti.Vector([0.0, 0.0, 0.0])
        self.highlighted = False
        self.normal_mat = ti.Matrix(3, 3, dt=ti.f32)
        self.highlight_mat = ti.Matrix(3, 3, dt=ti.f32)
        self.dt = 0.0
        self.last_x = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def sdf(self, pos: ti.template()):
        # Implement your Signed Distance Function here
        pass

    @ti.kernel
    def start(self):
        self.last_x = ti.Vector(self.sdf)
        # Load materials here

    @ti.kernel
    def access_to_global_variable(self):
        # Access to global variables
        pass

    @ti.kernel
    def update(self):
        self.access_to_global_variable()

        if ti.mouse_click(ti.GUI.LMB):
            self.pressed = True
            # Check if mouse hits
            # Update selection count and set selecting_me

        if ti.mouse_release(ti.GUI.LMB):
            self.pressed = False

        if self.selecting_me:
            if not self.highlighted:
                # Set highlighted material
                self.highlighted = True

            rot_amount = self.rot_speed * self.dt

            if ti.get_key(ti.KEY_W):
                # Rotate
                pass
            elif ti.get_key(ti.KEY_S):
                pass

            # Implement rotations for other keys

        else:
            if self.highlighted:
                # Set normal material
                self.highlighted = False

        if self.pressed and self.selecting_me:
            # Update position and mouse_move_v
            pass


# Main script
gui = ti.GUI("SDF Draggable", res=(800, 600))

sdf_draggable = SdfDraggable()

while gui.running:
    sdf_draggable.update()
    gui.show()
