import Mesh
import Solver

mesh = Mesh.Mesh(20, 20, 0.05)
mesh.set_point_vectors_name('Velocity')
mesh.set_cell_scalars_name('Pressure')

density = 1.225
dynamic_viscosity = 0.3 # Water : 0.001028

delta_t = 0.001
t_final = 0.1
max_C = 0.1

s = Solver.Solver(mesh, delta_t, t_final, density, dynamic_viscosity, max_C)

boundary_conditions = {
    'pressure' : 'dirichlet',
    'velocity' : 'sides'
    }

pressure_values = {'p_val': 0}
velocity_values = {
    'u_top' : 1,
    'v_top' : 0,
    'u_bot' : 0,
    'v_bot' : 0,
    'u_left' : 0,
    'v_left' : 0,
    'u_right' : 0,
    'v_right' : 0
    }

new_boundary_conditions = {
    'pressure': {
        'type': 'von_neumann',
        'values': pressure_values
        },
    'velocity': {
        'type': 'dirichlet',
        'values': velocity_values
        }
    }

s.solve(new_boundary_conditions, velocity_values)

s.m.write_vtk('simulation.vti')
