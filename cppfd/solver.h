#pragma once
#include<array>
#include<string>
#include<cstdio>

#include<Kokkos_Core.hpp>

#include"mesh.h"
#include"boundaryconditions.h"

class Solver
{
  public:
    // constructor
    Solver(Mesh& m, double d_t, double t_f, double r, double d_v, double m_C, int v = 1);

    // main solver routine
    void solve(BoundaryConditions& b_c);

  private:
    // assemble Laplacian matrix for Poisson solver
    void assemble_laplacian();

    // compute predicted velocities without pressure term
    void predict_velocity();

    // mesh object to be worked on
    Mesh mesh;

    // physics and simulation control related variables
    double nu, rho, delta_t, t_final, max_C;
    int verbosity;

    // laplacian matrix and its inverse
    Kokkos::View<double**> laplacian = {};
    Kokkos::View<double**> laplacian_inv = {};
};
