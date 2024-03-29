#pragma once
#include <array>
#include <string>
#include <cstdio>
#include <limits>
#include <cmath>
#include <map>

#include <Kokkos_Core.hpp>

#include "mesh.h"
#include "boundaryconditions.h"

class Solver
{
  public:
    Solver(Mesh& m, BoundaryConditions& b_c, double d_t, double t_f, double r, double d_v, double m_C, int v)
      : mesh(m)
      , boundary_conditions(b_c)
      , delta_t(d_t)
      , t_final(t_f)
      , rho(r)
      , max_C(m_C)
      , verbosity(v)
    {
      // compute and store kinematic viscosity
      this->nu = d_v / r;
    }

    // getter for the laplacian for unit testing
    Kokkos::View<double**> get_laplacian() {return this->laplacian;}

    // provide stopping points for debugging purposes
    enum struct stopping_point: uint8_t{
      NONE,
      AFTER_BOUNDARY_CONDITION,
      AFTER_LAPLACIAN,
      AFTER_FIRST_ITERATION
    };

    // allow for different linear solvers for the pressure
    enum struct linear_solver: uint8_t{
      NONE,
      CONJUGATE_GRADIENT
    };

    // allow for adaptative time stepping to be enabled/disabled
    enum struct adaptative_time_step: uint8_t{
      ON,
      OFF
    };

    // main solver routine
    void solve(stopping_point s_p = stopping_point::NONE, linear_solver l_s = linear_solver::CONJUGATE_GRADIENT, adaptative_time_step ats = adaptative_time_step::OFF);

  private:
    // assemble Laplacian matrix for Poisson solver
    void assemble_laplacian();

    // compute predicted velocities without pressure term
    void predict_velocity();

    // build poisson equation right hand side vector
    void assemble_poisson_RHS();

    // conjugate gradient based solver
    Kokkos::View<double*> conjugate_gradient_solve(double r_tol);

    // solve poisson pressure equation using conjugate gradient method
    void poisson_solve_pressure(double r_tol, linear_solver l_s);

    // apply corrector step
    void correct_velocity();

    // compute courant number in a certain cell
    double compute_cell_courant_number(int i, int j);

    // compute maximum courant number over all mesh cells
    double compute_global_courant_number();

    // reference to mesh onto which solve is performed
    Mesh& mesh;

    // physics and simulation control related variables
    double nu = 0.0008;
    double rho = 1.225;
    double delta_t = 0.001;
    double t_final = 0.001;
    double max_C = 0.5;
    double Re = 100;
    int verbosity = 1;
    double time_predictor_step = 0;
    double time_assemble_RHS = 0;
    double time_poisson_solve = 0;
    double time_corrector_step = 0;

    // laplacian matrix and its inverse
    Kokkos::View<double**> laplacian = {};

    // poisson equation right hand side vector
    Kokkos::View<double*> RHS = {};

    // boundary conditions
    BoundaryConditions boundary_conditions;
};
