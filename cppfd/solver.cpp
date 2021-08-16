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
    std::cout<<std::endl;
    std::cout<<"Computing laplacian matrix..."<<std::endl;
  }

  this->assemble_laplacian();
  double time2 = timer.seconds();
  if(this->verbosity > 0){
    std::cout<<"Laplacian computation duration: " << time2 - time1 <<std::endl;
  }

  if(this->verbosity > 0){
    std::cout<<std::endl;
    std::cout<<"Running simulation..."<<std::endl;
  }

  double t = 0;
  int iteration = 0;

  //start iterating on time steps
  while(t < this->t_final){
    ++iteration;
    t += this->delta_t;

    if(this->verbosity > 1){
      std::cout<<"Entering iteration " << iteration << " at time " << t <<std::endl;
    }

    // run all necessary solving steps
    this->predict_velocity();
    this->assemble_poisson_RHS();
    this->poisson_solve_pressure();
    this->correct_velocity();

    // // CFL based adaptative time step
    // double max_m_C = this->compute_global_courant_number();
    // double adjusted_delta_t = this->delta_t;
    // if(this->verbosity > 1){
    //   std::cout<<"    Computed global CFL = " << max_m_C << "; target max = " << this->max_C <<std::endl;
    // }
    // if(max_m_C > this->max_C){
    //   // decrease time step due to excessive CFL
    //   adjusted_delta_t = this->delta_t / (max_m_C * 100);
    //   if(this->verbosity > 1){
    //     std::cout<<"  - Decreased time step from " << this->delta_t << " to " << adjusted_delta_t <<std::endl;
    //   }
    //   this->delta_t = adjusted_delta_t;
    // }else if(max_m_C < 0.06 * this->max_C){
    //   // increase time step if possible to accelerate computation
    //   adjusted_delta_t = this->delta_t * 1.06;
    //   if(this->verbosity > 1){
    //     std::cout<<"  + Increased time step from " << this->delta_t << " to " << adjusted_delta_t <<std::endl;
    //   }
    //   this->delta_t = adjusted_delta_t;
    // }
  }

  double time3 = timer.seconds();
  if(this->verbosity > 0){
    std::cout<<std::endl;
    std::cout<<"Done."<<std::endl;
    std::cout<<"Total computation duration: " << time3 - time2 <<std::endl;
  }
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

  // predict velocity components using finite difference discretization
  for(int j = 1; j < this->mesh.get_n_cells_y(); j++){
    for(int i = 1; i < this->mesh.get_n_cells_x(); i++){
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

// implementation of the poisson RHS assembly
void Solver::assemble_poisson_RHS(){
  this->RHS = Kokkos::View<double*[1]>("RHS", this->mesh.get_n_points_x() * this->mesh.get_n_points_y());
  double factor = this->rho / this->delta_t;
  for(int j = 0; j < this->mesh.get_n_cells_y(); j++){
    for(int i = 1; i < this->mesh.get_n_cells_x(); i++){
      double u_r = this->mesh.get_velocity_u(i+1, j) + this->mesh.get_velocity_u(i+1, j+1);
      double u_l = this->mesh.get_velocity_u(i, j) + this->mesh.get_velocity_u(i, j+1);
      double v_t = this->mesh.get_velocity_v(i, j+1) + this->mesh.get_velocity_v(i+1, j+1);
      double v_b = this->mesh.get_velocity_v(i, j) + this->mesh.get_velocity_v(i+1, j);
      double inv_h = 1 / (2 * this->mesh.get_cell_size());
      int k = this->mesh.cartesian_to_index(i, j, this->mesh.get_n_points_x(), this->mesh.get_n_points_y());
      this->RHS(k, 0) = factor * ((u_r - u_l + v_t - v_b) * inv_h);
    }
  }
}

//implementation of the poisson equation solver
void Solver::poisson_solve_pressure(){
  std::cout<<"*solves poisson pressure equation*"<<std::endl;
}

// implementation of the corrector step
void Solver::correct_velocity(){
  for(int j = 1; j <= this->mesh.get_n_cells_y(); j++){
    for(int i = 1; i <= this->mesh.get_n_cells_x(); i++){
      this->mesh.set_velocity_u(i, j, (this->mesh.get_velocity_u(i, j) - (this->delta_t/this->rho) * (this->mesh.get_pressure(i, j) - this->mesh.get_pressure(i-1, j)) * 1/this->mesh.get_cell_size()));
    }
  }
  for(int j = 1; j <= this->mesh.get_n_cells_y(); j++){
    for(int i = 1; i <= this->mesh.get_n_cells_x(); i++){
      this->mesh.set_velocity_v(i, j, (this->mesh.get_velocity_u(i, j) - (this->delta_t/this->rho) * (this->mesh.get_pressure(i, j) - this->mesh.get_pressure(i, j-1)) * 1/this->mesh.get_cell_size()));
    }
  }
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
