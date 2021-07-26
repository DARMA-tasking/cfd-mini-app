import math
import sys

class BoundaryConditions:
    def __init__(self, mesh, boundary_conditions):
        self.m = mesh

        # Define all necessary boundary condtion variables
        self.pressure_bc = boundary_conditions.get('pressure').get('type')
        self.velocity_bc = boundary_conditions.get('velocity').get('type')

        # Handle sides case with default 0 values
        if self.velocity_bc == 'dirichlet':
            self.u_top = boundary_conditions.get('velocity').get('values').get('u_top', 0)
            self.v_top = boundary_conditions.get('velocity').get('values').get('v_top', 0)
            self.u_bot = boundary_conditions.get('velocity').get('values').get('u_bot', 0)
            self.v_bot = boundary_conditions.get('velocity').get('values').get('v_bot', 0)
            self.u_left = boundary_conditions.get('velocity').get('values').get('u_left', 0)
            self.v_left = boundary_conditions.get('velocity').get('values').get('v_left', 0)
            self.u_right = boundary_conditions.get('velocity').get('values').get('u_right', 0)
            self.v_right = boundary_conditions.get('velocity').get('values').get('v_right', 0)

        if self.pressure_bc == 'dirichlet':
            self.p_val = boundary_conditions.get('pressure').get('values').get('p_val', 0)

    def apply_velocity_boundary_conditions(self):
        if self.velocity_bc == 'dirichlet':
            for i in range(0, self.m.n_cells_x+1):
                self.m.set_point_vector_u(i, self.m.n_cells_y, self.u_top)
                self.m.set_point_vector_v(i, self.m.n_cells_y, self.v_top)
                self.m.set_point_vector_u(i, 0, self.u_bot)
                self.m.set_point_vector_v(i, 0, self.v_bot)

            for j in range(0, self.m.n_cells_y+1):
                self.m.set_point_vector_u(0, j, self.u_left)
                self.m.set_point_vector_v(0, j, self.v_left)
                self.m.set_point_vector_u(self.m.n_cells_x, j, self.u_right)
                self.m.set_point_vector_v(self.m.n_cells_x, j, self.v_right)

            return math.sqrt(max([
                self.u_top * self.u_top + self.v_top * self.v_top,
                self.u_bot * self.u_bot + self.v_bot * self.v_bot,
                self.u_left * self.u_left + self.v_left * self.v_left,
                self.u_right * self.u_right + self.v_right * self.v_right]))


    def apply_pressure_boundary_conditions(self):
        if self.pressure_bc == 'dirichlet':
            for i in range(self.m.n_cells_x + 1):
                self.m.set_cell_scalar(i, 0, self.p_val)
                self.m.set_cell_scalar(i, self.m.n_cells_y + 1, self.p_val)

            for j in range(self.m.n_cells_y + 1):
                self.m.set_cell_scalar(0, j, self.p_val)
                self.m.set_cell_scalar(self.m.n_cells_x + 1, j, self.p_val)
