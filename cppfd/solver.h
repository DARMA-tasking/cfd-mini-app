#pragma once
#include<array>
#include<string>
#include<cstdio>
#include<limits>
#include<cmath>
#include<map>

#include<Kokkos_Core.hpp>

#include<KokkosBlas1_axpby.hpp>
#include<KokkosBlas1_dot.hpp>
#include<KokkosBlas2_gemv.hpp>
#include<KokkosBlas3_gemm.hpp>

#include"mesh.h"
#include"boundaryconditions.h"

class Solver
{
  public:
    // delete default constructor
    Solver() = delete;

    // initialization constructor
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

    enum struct stopping_point: uint8_t{
      NONE,
      AFTER_LAPLACIAN,
      AFTER_RHS
    };

    // main solver routine
    void solve(stopping_point s_p = stopping_point::NONE);

  private:
    // assemble Laplacian matrix for Poisson solver
    void assemble_laplacian();

    // compute predicted velocities without pressure term
    void predict_velocity();

    // build poisson equation right hand side vector
    void assemble_poisson_RHS();

    // conjugate gradient based solver
    Kokkos::View<double*> conjugate_gradient_solve(Kokkos::View<double**> A, Kokkos::View<double*>b);

    // solve poisson pressure equation using conjugate gradient method
    void poisson_solve_pressure();

    // apply corrector step
    void correct_velocity();

    // compute courant number in a certain cell
    double compute_cell_courant_number(int i, int j);

    // compute maximum courant number over all mesh cells
    double compute_global_courant_number();

    // mesh object to be worked on
    Mesh mesh;

    // physics and simulation control related variables
    double nu, rho, delta_t, t_final, max_C, Re;
    int verbosity;

    // laplacian matrix and its inverse
    Kokkos::View<double**> laplacian = {};
    Kokkos::View<double**> laplacian_inv = {};

    // poisson equation right hand side vector
    Kokkos::View<double*> RHS = {};

    // poisson equation left hand side pressure vector
    Kokkos::View<double*> pressure_vector = {};

    // boundary conditions
    BoundaryConditions boundary_conditions;
};
