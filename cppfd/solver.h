#pragma once
#include <array>
#include <string>
#include <cstdio>
#include <limits>
#include <cmath>
#include <map>

#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosSparse_CrsMatrix.hpp>

//#include "mesh_chunk.h"
#include "boundary_conditions.h"
#include "parallel_mesh.h"

class Solver
{
  public:
  Solver(std::shared_ptr<MeshChunk> m,
      BoundaryConditions& b_c,
      double d_t,
      double t_f,
      double r,
      double d_v,
      double m_C,
      int v,
      uint64_t domain_size_x,
      uint64_t domain_size_y,
      double cell_size,
      uint64_t n_pmesh_x,
      uint64_t n_pmesh_y,
      uint64_t colors_x,
      uint64_t colors_y)
      : mesh_chunk(m)
      , boundary_conditions(b_c)
      , delta_t(d_t)
      , t_final(t_f)
      , rho(r)
      , max_C(m_C)
      , verbosity(v)
      , nu(d_v / r)
      , domain_size_x(domain_size_x)
      , domain_size_y(domain_size_y)
      , h(cell_size)
      , p(n_pmesh_x * n_pmesh_y)
      , n_p_mesh_x(n_pmesh_x)
      , n_p_mesh_y(n_pmesh_y)
      , n_colors_x(colors_x)
      , n_colors_y(colors_y)
    {}

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
      CONJUGATE_GRADIENT,
      GAUSS_SEIDEL
    };

    // allow for adaptative time stepping to be enabled/disabled
    enum struct adaptative_time_step: uint8_t{
      ON,
      OFF
    };

    // getter for the Laplacian for unit testing
    using exec_space = typename Kokkos::DefaultExecutionSpace;
    using mem_space = typename exec_space::memory_space;
    using device_type = typename Kokkos::
      Device<Kokkos::DefaultExecutionSpace, mem_space>;
  using matrix_type = typename KokkosSparse::CrsMatrix<double, int64_t, device_type, void, int64_t>;
    matrix_type get_Laplacian() {return this->Laplacian;}

    // main solver routine
    void solve(stopping_point s_p = stopping_point::NONE, linear_solver l_s = linear_solver::GAUSS_SEIDEL, adaptative_time_step ats = adaptative_time_step::OFF);

    // global vtk file export
    uint64_t write_vtms(const std::string&) const;

  private:
    // assemble parallel meshes that will be used depending on number of ranks
    void assemble_parallel_meshes();

    // set mesh chunk border types depending on their position in the global and parallel mesh
    void set_mesh_chunk_borders();

    // assign existing mesh chunk to new parallel mesh
    void assign_mesh_chunk_to_parallel_mesh(uint64_t i, uint64_t j);

    // apply boundary conditions globally
    void apply_velocity_bc(std::map<std::string, double> velocity_values);

    // assemble Laplacian matrix for Poisson solver and return density
    double assemble_Laplacian();

    // convenience method to add entry into CRS matrix
    void assign_CRS_entry(uint64_t &idx,
			  bool &first_in_row,
			  const uint64_t k,
			  const uint64_t offset,
			  const double value,
			  Kokkos::View<int64_t*> row_ptrs,
			  Kokkos::View<int64_t*> col_ids,
			  Kokkos::View<double*> values);

    // compute predicted velocities without pressure term
    void predict_velocity();

    // compute predicted velocities without pressure term using MPI
    void MPI_predict_velocity();

    // build poisson equation right hand side vector
    void assemble_poisson_RHS();

    // solve poisson pressure equation using conjugate gradient method
    void poisson_solve_pressure(double r_tol, linear_solver l_s);

    // conjugate gradient solver
    Kokkos::View<double*> conjugate_gradient_solve(double r_tol);

    // Gauss-Seidel solver
    Kokkos::View<double*> gauss_seidel_solve(double r_tol, int max_it, int n_sweeps);

    // apply corrector step
    void correct_velocity();

    // compute courant number in a certain cell
    double compute_cell_courant_number(int i, int j);

    // compute maximum courant number over all mesh cells
    double compute_global_courant_number();

    // reference to mesh onto which solve is performed
    std::shared_ptr<MeshChunk> mesh_chunk;

    // storage for parallel meshes
    std::map<std::array<uint64_t, 2>, ParallelMesh> parallel_meshes = {};

    // store Kokkos kernels zero and unit values
    double zero = Kokkos::ArithTraits<double>::zero();
    double one = Kokkos::ArithTraits<double>::one();

    // parallel mesh variables
    uint64_t p = 1;
    uint64_t n_p_mesh_x = 1;
    uint64_t n_p_mesh_y = 1;
    uint64_t n_colors_x = 1;
    uint64_t n_colors_y = 1;

    // default physics variable values
    double nu = 0.0008;
    double rho = 1.225;
    double delta_t = 0.001;
    double t_final = 0.001;
    double max_C = 0.5;
    double Re = 100;

    // parallel mesh variables
    uint64_t domain_size_x;
    uint64_t domain_size_y;
    double h = 1.;

    // default verbosity level
    int verbosity = 1;

    // Laplacian matrix and its inverse
    matrix_type Laplacian = {};

    // poisson equation right hand side vector
    Kokkos::View<double*> RHS = {};

    // boundary conditions
    BoundaryConditions boundary_conditions;
};
