#pragma once
#include<array>
#include<string>
#include<cstdio>

#include<Kokkos_Core.hpp>

#include"mesh.h"

class Solver
{
  public:
    void set_nu(double n){this->nu = n;};
    double get_nu(){return this->nu;};
    void set_rho(double r){this->rho = r;};
    double get_rho(){return this->rho;};
    void set_delta_t(double d_t){this->delta_t = d_t;};
    double get_delta_t(){return this->delta_t;};
    void set_t_final(double t_f){this->t_final = t_f;};
    double get_t_final(){return this->t_final;};

    void set_verbose_level(std::string v){this->verbose = v;};
    std::string get_verbose_level(){return this->verbose;};

  private:
    // mesh object to be worked on
    Mesh m;

    // physics and simulation control related variables
    double nu, rho, delta_t, t_final, max_C;
    std::string verbose;

    // laplacian matrix
    Kokkos::View<double**> laplacian = {};
};
