#pragma once
#include<array>
#include<string>
#include<cstdio>

#include<Kokkos_Core.hpp>

#include"mesh.h"

class Solver
{
  public:
    Solver(Mesh m, double d_t, double t_f, double r, double d_v, double m_C, int v = 1);
    void set_nu(double n){this->nu = n;};
    double get_nu(){return this->nu;};
    void set_rho(double r){this->rho = r;};
    double get_rho(){return this->rho;};
    void set_delta_t(double d_t){this->delta_t = d_t;};
    double get_delta_t(){return this->delta_t;};
    void set_t_final(double t_f){this->t_final = t_f;};
    double get_t_final(){return this->t_final;};

  private:
    // assemble Laplacian matrix for Poisson solver
    void assemble_laplacian();

    // mesh object to be worked on
    Mesh mesh;

    // physics and simulation control related variables
    double nu, rho, delta_t, t_final, max_C;
    int verbosity;

    // laplacian matrix and its inverse
    Kokkos::View<double**> laplacian = {};
    Kokkos::View<double**> laplacian_inv = {};
};
