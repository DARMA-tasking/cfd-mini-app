#include<Kokkos_Core.hpp>

#include"solver.h"

// main function to solve N-S equation on time steps
void Solver::solve(){
  // begin simulation
  if(this->verbosity >= 1){
    std::cout<<"Initial timestep : " << this->delta_t << ", " << "Time final : " << this->t_final <<std::endl;
  }

  Kokkos::Timer timer;

  if(this->verbosity >= 1){
    std::cout<<"Applying velocity boundary conditions..."<<std::endl;
  }
  this->boundary_conditions.apply_velocity_bc();

  double velocity_bc_max = this->boundary_conditions.get_velocity_bc_max_norm();
  double l = this->mesh.get_cell_size() * std::max(this->mesh.get_n_cells_x(), this->mesh.get_n_cells_y());
  this->Re = velocity_bc_max * l / this->nu;
  if(this->verbosity > 0){
    std::cout << "Reynolds number : " << this->Re <<std::endl;
  }

  double time1 = timer.seconds();
  if(this->verbosity > 0){
    std::cout<<"Computing laplacian matrix..."<<std::endl;
  }

  this->assemble_laplacian();
  double time2 = timer.seconds();
  if(this->verbosity > 0){
    std::cout<<"Laplacian computation duration: " << time2 - time1 <<std::endl;
  }

  if(this->verbosity > 0){
    std::cout<<"Running simulation..."<<std::endl;
  }

  double t = 0;
  int iteration = 0;

  //start iterating on time steps
  while(t < this->t_final){
    ++iteration;
    t += this->delta_t;

    if(this->verbosity >= 2){
      std::cout<<"Entering iteration " << iteration << " at time " << t <<std::endl;
    }
  }

  this->boundary_conditions.apply_velocity_bc();
  this->predict_velocity();
}

// computation of laplacian matrix
void Solver::assemble_laplacian(){
  // initialize container for Laplacian
  const int m = this->mesh.get_n_cells_x();
  const int n = this->mesh.get_n_cells_y();
  const int mn = m * n;
  this->laplacian = Kokkos::View<double**>("laplacian", mn, mn);

  for(int j = 1; j <= n - 1; j++){
    for(int i = 1; i <= m - 1; i++){
      int k = this->mesh.cartesian_to_index(i, j, m, n);
      // initialize diagonal value
      int v = -4;

      //detect corners and borders
      if(i == 0 || i == m - 1 || j == 0 || j == n - 1)
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
      if(i < m - 1)
      {
        this->laplacian(k, k + 1) = 1;
      }
      if(j == 1)
      {
        this->laplacian(k, k - m) = 1;
      }
      if(j < n - 1)
      {
        this->laplacian(k, k + m) = 1;
      }
    }
  }
}

// implementation of the predictor step
void Solver::predict_velocity(){
  Kokkos::View<double*[2]> predicted_velocity("velocity", this->mesh.get_n_points_x() * this->mesh.get_n_points_y());

  // compute common factors
  const double inv_size_sq = 1 / (this->mesh.get_cell_size() * this->mesh.get_cell_size());
  const double inv_size_db = 1 / (2 * this->mesh.get_cell_size());

  // predict vwlocity components using finite difference discretization
  for(int j = 1; j <= this->mesh.get_n_cells_y() - 1; j++){
    for(int i = 1; i <= this->mesh.get_n_cells_x() - 1; i++){
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

double Solver::compute_cell_courant_number(int i, int j){
  return (this->mesh.get_velocity_u(i, j) + this->mesh.get_velocity_v(i, j)) * this->delta_t / this->mesh.get_cell_size();
}

double Solver::compute_global_courant_number(){
  double max_C = std::numeric_limits<int>::max();
  for(int j = 1; j <= this->mesh.get_n_cells_y(); j++){
    for(int i = 1; i <= this->mesh.get_n_cells_x(); i++){
      double c = this->compute_cell_courant_number(i, j);
      if(c > max_C){
        max_C = c;
      }
    }
  }
  return max_C;
}
