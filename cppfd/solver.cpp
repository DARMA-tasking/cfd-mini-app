#include<Kokkos_Core.hpp>

#include"solver.h"

Solver::Solver(Mesh& m, double d_t, double t_f, double r, double d_v, double m_C, int v)
  : mesh(m)
  , delta_t(d_t)
  , t_final(t_f)
  , rho(r)
  , max_C(m_C)
  , verbosity(v)
  {
    // compute and store kinematic viscosity
    this->nu = d_v / r;
  }

// main function to solve N-S equation on time steps
void Solver::solve(BoundaryConditions& b_c)
{
  // implement me

  b_c.apply_velocity_bc();
  this->predict_velocity();
}

// implementation of the predictor step
void Solver::predict_velocity()
{
  Kokkos::View<double*[2]> predicted_velocity("velocity", this->mesh.get_n_points_x() * this->mesh.get_n_points_y());

  // compute common factors
  const double inv_size_sq = 1 / (this->mesh.get_cell_size() * this->mesh.get_cell_size());
  const double inv_size_db = 1 / (2 * this->mesh.get_cell_size());

  // predict vwlocity components using finite difference discretization
  for(int j = 1; j <= this->mesh.get_n_cells_y() - 1; j++)
  {
    for(int i = 1; i <= this->mesh.get_n_cells_x() - 1; i++)
    {
      // factors needed to predict new u component
      double v = 0.25 * (this->mesh.get_velocity_v(i-1, j) + this->mesh.get_velocity_v(i, j) + this->mesh.get_velocity_v(i, j+1));
      double dudx2 = (this->mesh.get_velocity_u(i-1, j) - 2 * this->mesh.get_velocity_u(i, j) + this->mesh.get_velocity_u(i+1, j)) * inv_size_sq;
      double dudy2 = (this->mesh.get_velocity_u(i, j-1) - 2 * this->mesh.get_velocity_u(i, j) + this->mesh.get_velocity_u(i, j+1)) * inv_size_sq;
      double dudx = this->mesh.get_velocity_u(i, j) * (this->mesh.get_velocity_u(i+1, j) - this->mesh.get_velocity_u(i-1, j)) * inv_size_db;
      double dudy = v * (this->mesh.get_velocity_u(i, j+1) - this->mesh.get_velocity_u(i, j-1)) * inv_size_db;

      // factors needed to predict new v component
      double u = 0.25 * (this->mesh.get_velocity_u(i, j-1) + this->mesh.get_velocity_u(i, j) + this->mesh.get_velocity_u(i, j+1));
      double dvdx2 = (this->mesh.get_velocity_v(i-1, j) - 2 * this->mesh.get_velocity_v(i, j) + this->mesh.get_velocity_v(i+1, j)) * inv_size_sq;
      double dvdy2 = (this->mesh.get_velocity_v(i, j-1) - 2 * this->mesh.get_velocity_v(i, j) + this->mesh.get_velocity_v(i, j+1)) * inv_size_sq;
      double dvdy = this->mesh.get_velocity_v(i, j) * (this->mesh.get_velocity_v(i+1, j) - this->mesh.get_velocity_v(i-1, j)) * inv_size_db;
      double dvdx = u * (this->mesh.get_velocity_v(i, j+1) - this->mesh.get_velocity_v(i, j-1)) * inv_size_db;

      // assign predicted u and v components to predicted_velocity storage
      int k = this->mesh.cartesian_to_index(i, j, this->mesh.get_n_points_x(), this->mesh.get_n_points_y());
      predicted_velocity(k, 0) = this->mesh.get_velocity_u(i, j) + this->delta_t * (this->nu * (dudx2 + dudy2) - (this->mesh.get_velocity_u(i, j) * dudx + v * dudy));
      predicted_velocity(k, 1) = this->mesh.get_velocity_v(i, j) + this->delta_t * (this->nu * (dvdx2 + dvdy2) - (this->mesh.get_velocity_v(i, j) * dvdx + u * dvdy));
    }
  }

  // assign predicted velocity vectors to mesh
  this->mesh.set_mesh_velocities(predicted_velocity);

}

// computation of laplacian matrix
void Solver::assemble_laplacian()
  {
    // initialize container for Laplacian
    const int mn = this->mesh.get_n_cells_x() * this->mesh.get_n_cells_y();
    this->laplacian = Kokkos::View<double**>("laplacian", mn, mn);

    for(int j = 1; j <= mn - 1; j++)
    {
      for(int i = 1; i <= mn - 1; i++)
      {
        int k = this->mesh.cartesian_to_index(i, j, mn, mn);
        // initialize diagonal value
        int v = -4;

        //detect corners and borders
        if(i == 0 || i == mn - 1 || j == 0 || j == mn - 1)
        {
          ++v;
        }

        // assign diagonal entry
        this->laplacian(k, k) = v;

        // assign unit entries
        if(i == 1)
        {
          this->laplacian(k, k-1) = 1;
        }
        if(i < mn - 1)
        {
          this->laplacian(k, k + 1) = 1;
        }
        if(j == 1)
        {
          this->laplacian(k, k - mn) = 1;
        }
        if(j < mn - 1)
        {
          this->laplacian(k, k + mn) = 1;
        }
      }
    }
  }
