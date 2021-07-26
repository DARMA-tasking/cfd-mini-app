import sys
import os
import math
import numpy as np
import copy
from datetime import datetime
from skimage import io
import matplotlib.pyplot as plt

import Mesh
import BoundaryConditions
import Laplacian_JLJ
import Laplacian_PPP

class Solver:
    def __init__(self, mesh, delta_t, t_final, density, dynamic_viscosity, max_C, verbose=True):
        # Mesh variables
        self.nx, self.ny = mesh.dim

        self.m = copy.deepcopy(mesh)
        self.m.set_cell_scalars_name = 'Pressure'
        self.m.set_point_vectors_name = 'Velocity'

        # Fluid variables
        self.nu = dynamic_viscosity / density
        self.rho = density

        # Solver Variables
        self.delta_t = delta_t
        self.t_final = t_final

        # Desired maximum Courant number across mesh
        self.max_C = max_C

        # Verbosity level
        self.verbose = verbose

    def laplacian(self, L00=0):
        inv_h_sq = 1 / self.m.h ** 2
        if self.boundary_conditions.pressure_bc == 'dirichlet':
            n = self.nx
            self.laplacian = inv_h_sq * Laplacian_JLJ.fill(np.zeros((n*n, n*n)), n)

        elif self.boundary_conditions.pressure_bc == 'von_neumann':
            n = self.nx
            self.laplacian = inv_h_sq * Laplacian_PPP.fill(np.zeros((n*n, n*n)), n)

            # Add necessary conditions to allow inverse matrix
            self.laplacian[0][0] = L00

        # Inverse the matrix
        self.laplacian_inv = np.linalg.inv(self.laplacian)

    def apply_boundary_conditions(self, u_top, u_bot, v_left, v_right, vel_in):
        if self.velocity_bc == 'sides':
            BoundaryConditions.sides(self.m, u_top, u_bot, v_left, v_right)

        elif self.velocity_bc == 'velocity_input':
            BoundaryConditions.velocity_input(self.m, vel_in)

    def predictor_step(self):
        # Initialize containers for predicted velocities
        us, vs = [], []

        # Compute common factors
        inv_size_sq = 1 / (self.m.h ** 2)
        inv_size_db = 1 / (2 * self.m.h)

        # Iterate in j major way
        for j in range(1, self.m.n_cells_y):
            for i in range(1, self.m.n_cells_x):
                # Predict new u component
                v = 0.25 * (self.m.get_point_vector_v(i-1, j) + self.m.get_point_vector_v(i, j) + self.m.get_point_vector_v(i, j+1))
                dudux2 = (self.m.get_point_vector_u(i-1, j) - 2 * self.m.get_point_vector_u(i, j) + self.m.get_point_vector_u(i+1, j)) * inv_size_sq
                duduy2 = (self.m.get_point_vector_u(i, j-1) - 2 * self.m.get_point_vector_u(i, j) + self.m.get_point_vector_u(i, j+1)) * inv_size_sq
                dudux = self.m.get_point_vector_u(i, j) * (self.m.get_point_vector_u(i+1, j) - self.m.get_point_vector_u(i-1, j)) * inv_size_db
                duduy = v * (self.m.get_point_vector_u(i, j+1) - self.m.get_point_vector_u(i, j-1)) * inv_size_db
                us += [self.m.get_point_vector_u(i, j) + self.delta_t * (self.nu * (dudux2 + duduy2) - (self.m.get_point_vector_u(i, j) * dudux + v * duduy))]

                # Predict new v component
                u = 0.25 * (self.m.get_point_vector_u(i, j-1) + self.m.get_point_vector_u(i, j) + self.m.get_point_vector_u(i, j+1))
                dvdvx2 = (self.m.get_point_vector_v(i-1, j) - 2 * self.m.get_point_vector_v(i, j) + self.m.get_point_vector_v(i+1, j)) * inv_size_sq
                dvdvy2 = (self.m.get_point_vector_v(i, j-1) - 2 * self.m.get_point_vector_v(i, j) + self.m.get_point_vector_v(i, j+1)) * inv_size_sq
                dvdvy = self.m.get_point_vector_v(i, j) * (self.m.get_point_vector_v(i+1, j) - self.m.get_point_vector_v(i-1, j)) * inv_size_db
                dvdvx = u * (self.m.get_point_vector_v(i, j+1) - self.m.get_point_vector_v(i, j-1)) * inv_size_db
                vs += [self.m.get_point_vector_v(i, j) + self.delta_t * (self.nu * (dvdvx2 + dvdvy2) - (self.m.get_point_vector_v(i, j) * dvdvx + u * dvdvy))]

        # Assign new values for both components
        k = 0
        for j in range(1, self.m.n_cells_y):
            for i in range(1, self.m.n_cells_x):
                self.m.set_point_vector_u(i, j, us[k])
                self.m.set_point_vector_v(i, j, vs[k])
                k += 1

    def poisson_RHS(self): #side = 0
        self.RHS = []
        fac = self.rho / self.delta_t
        for j in range(0, self.m.n_cells_y):
            for i in range(0, self.m.n_cells_x):
                u_r = self.m.get_point_vector_u(i+1, j) + self.m.get_point_vector_u(i+1, j+1)
                u_l = self.m.get_point_vector_u(i, j) + self.m.get_point_vector_u(i, j+1)
                v_t = self.m.get_point_vector_v(i, j+1) + self.m.get_point_vector_v(i+1, j+1)
                v_b = self.m.get_point_vector_v(i, j) + self.m.get_point_vector_v(i+1, j)
                inv_h = 1 / (2 * self.m.h)
                self.RHS += [fac * ((u_r - u_l + v_t - v_b) * inv_h)]


        self.RHS = np.array(self.RHS).reshape(len(self.RHS))

    def poisson_solve_pressure(self):
        self.P = []
        if self.boundary_conditions.pressure_bc == 'dirichlet':
            for j in range(0, self.ny):
                for i in range(0, self.nx):
                    self.P.append([self.m.get_cell_scalar(i,j)])
        elif self.boundary_conditions.pressure_bc == 'von_neumann':
            for j in range(0, self.ny):
                for i in range(0, self.nx):
                    self.P.append([self.m.get_cell_scalar(i,j)])
        self.P = np.array(self.P)

        p = np.dot(self.laplacian_inv, self.RHS)
        for i in range(len(p)):
            self.P[i][0] = p[i]

        for k in range(len(self.P)):
            i, j = self.m.index_to_cartesian(k, self.nx, (self.nx)**2+1)
            self.m.set_cell_scalar(i, j, self.P[k][0])

    def corrector_step(self):
        for j in range(1, self.m.n_cells_y):
            for i in range(1, self.m.n_cells_x):
                self.m.set_point_vector_u(i, j, (self.m.get_point_vector_u(i, j) - (self.delta_t/self.rho) * (self.m.get_cell_scalar(i, j) - self.m.get_cell_scalar(i-1, j)) * 1/self.m.h))
        for j in range(1, self.m.n_cells_y):
            for i in range(1, self.m.n_cells_x):
                self.m.set_point_vector_v(i, j, (self.m.get_point_vector_v(i, j) - (self.delta_t/self.rho) * (self.m.get_cell_scalar(i, j) - self.m.get_cell_scalar(i, j-1)) * 1/self.m.h))

    def compute_cell_courant_number(self, i, j):
        return (self.m.get_point_vector_u(i,j) + self.m.get_point_vector_v(i,j)) * self.delta_t / self.m.h

    def compute_global_courant_number(self):
        max_C = - math.inf
        for j in range(1, self.m.n_points_y):
            for i in range(1, self.m.n_points_x):
                C = self.compute_cell_courant_number(i, j)
                if C > max_C:
                    max_C = C
        return max_C

    def average(self, type):
        res = 0
        c = 0
        if type == 'vx':
            for j in range(1, self.m.n_cells_y):
                for i in range(1, self.m.n_cells_x):
                    c +=1
                    res += self.m.get_point_vector_u(i,j)
        elif type == 'vy':
            for j in range(1, self.m.n_cells_y):
                for i in range(1, self.m.n_cells_x):
                    c +=1
                    res += self.m.get_point_vector_v(i,j)
        elif type == 'p':
            for j in range(self.m.n_cells_y):
                for i in range(self.m.n_cells_x):
                    c +=1
                    res += self.m.get_cell_scalar(i,j)
        return res/c

    def visualize_courant(self):
        image = np.zeros((self.m.n_points_x, self.m.n_points_y))
        for j in range(1, self.m.n_points_y):
            for i in range(1, self.m.n_points_x):
                C = self.compute_cell_courant_number(i,j)
                image[i][j] = C
        io.imshow(image)
        plt.show()

    def plot_velocity_x(self):
        Y = []
        VALUES = []
        m = self.m.n_cells_x // 2
        y = 0
        for j in range(0, self.m.n_cells_y + 1):
            VALUES.append(self.m.get_point_vector_u(m, j))
            y += self.m.h
            Y.append(y)
        plt.close('all')
        plt.plot(Y, VALUES)
        plt.show()


    def solve(self, boundary_conditions, values, side = 0):

        self.boundary_conditions = BoundaryConditions.BoundaryConditions(self.m, boundary_conditions)

        # Convergence test related variables

        I = []
        Vx = []
        Vy = []
        P = []
        maxC = []

        # Begin simulation
        print('# Time Step: {}, Time Final: {}'.format(self.delta_t, self.t_final))
        calculation_beginning = datetime.now()
        print('# Applying velocity boundary conditions...')
        velocity_bc_max = self.boundary_conditions.apply_velocity_boundary_conditions()
        L = self.m.h * max([self.m.n_cells_x, self.m.n_cells_y])
        Re = velocity_bc_max * L / self.nu
        print('  Reynolds number = {:.6g}'.format(Re))
        print('# Computing laplacian matrix...')
        self.laplacian()
        print('# Running simulation...')

        # Define time related variables
        t = 0
        iteration = 0

        # Start simulation
        while t < self.t_final:
            iteration += 1
            #print('Progress:', "{:.1f}".format((iteration/total_iteration) * 100),'%', end = '\r')
            t += self.delta_t
            if self.verbose:
                print('  Entering iteration {} at time {:.6g}'.format(iteration, t))

            self.predictor_step()
            self.poisson_RHS()
            self.poisson_solve_pressure()
            self.corrector_step()

            # CFL based adaptive time step
            max_C = self.compute_global_courant_number()
            if self.verbose:
                print('    Computed global CFL = {:.6g} (target max = {:.6g})'.format(max_C, self.max_C))
            if max_C > self.max_C:
                # Decrease time step due to excessive CFL
                adjusted_delta_t = self.delta_t / (max_C * 100)
                if self.verbose:
                    print('  - Decreased time step from {:.6g} to {:.6g}'.format(self.delta_t, adjusted_delta_t))
                self.delta_t = adjusted_delta_t
            elif max_C < 0.06 * self.max_C:
                # Increase time step if possible
                adjusted_delta_t = self.delta_t * 1.06
                if adjusted_delta_t > self.delta_t:
                    if self.verbose:
                        print('  + Increased time step from {:.6g} to {:.6g}'.format(self.delta_t, adjusted_delta_t))
                    self.delta_t = adjusted_delta_t

            # Plot convergence information
            I.append(iteration)
            Vx.append(self.average('vx'))
            Vy.append(self.average('vy'))
            P.append(self.average('p'))
            maxC.append(max_C)

        calculation_end = datetime.now()
        calculation_time = calculation_end - calculation_beginning
        print()
        print('Done.')
        print('Calculation Time: {} seconds, {} microseconds'.format(calculation_time.seconds, calculation_time.microseconds))

        #self.visualize_courant()

        #plt.close('all')
        #plt.subplot(3, 2, 1)
        #plt.plot(I, Vx)
        #plt.subplot(3,2,2)
        #plt.plot(I, Vy)
        #plt.subplot(3,2,3)
        #plt.plot(I, P)
        #plt.subplot(3,2,4)
        #plt.plot(I, maxC)
        #plt.show()

        #self.plot_velocity_x()
