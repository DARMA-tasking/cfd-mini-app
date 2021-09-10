#include <Kokkos_Core.hpp>

#include "solver.h"

#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas3_gemm.hpp>

// main function to solve N-S equation on time steps
void Solver::solve(stopping_point s_p, linear_solver l_s,adaptative_time_step ats){
  // begin simulation
  if(this->verbosity >= 1){
    std::cout<<"Initial timestep : " << this->delta_t << ", " << "Time final : " << this->t_final <<std::endl;
  }

  Kokkos::Timer timer;

  if(this->verbosity >= 1){
    std::cout<<"Applying velocity boundary conditions..."<<std::endl;
  }
  this->boundary_conditions.apply_velocity_bc();
  if(s_p == stopping_point::AFTER_BOUNDARY_CONDITION){
    for(int j = 0; j < this->mesh.get_n_points_y(); j++){
      for(int i = 0; i < this->mesh.get_n_points_x(); i++){
        std::cout<<i<<" "<<j<<" : "<<this->mesh.get_velocity_u(i, j)<<" , "<<this->mesh.get_velocity_v(i, j)<<std::endl;
      }
    }
    return;
  }

  // compute Reynolds number from input parameters
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
  if(s_p == stopping_point::AFTER_LAPLACIAN){
    for(int j = 0 ; j<this->laplacian.extent(1); j++){
      for(int i = 0; i < this->laplacian.extent(0); i++){
        std::cout<<" "<<this->laplacian(i, j);
      }
      std::cout<<std::endl;
    }
    return;
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
    Kokkos::Timer timer2;

    this->predict_velocity();
    double time_post_predict = timer2.seconds();
    this->time_predictor_step += time_post_predict;
    if(s_p == stopping_point::AFTER_FIRST_ITERATION){
      std::cout<<" predicted velocity:\n";
      for(int j = 0; j < this->mesh.get_n_points_y(); j++){
        for(int i = 0; i < this->mesh.get_n_points_x(); i++){
          std::cout<<"  "<<i<<" "<<j<<" : "<<this->mesh.get_velocity_u(i, j)<<" , "<<this->mesh.get_velocity_v(i, j)<<std::endl;
        }
      }
    }

    this->assemble_poisson_RHS();
    double time_post_RHS = timer2.seconds();
    this->time_assemble_RHS += time_post_RHS - time_post_predict;
    if(s_p == stopping_point::AFTER_FIRST_ITERATION){
      std::cout<<" RHS:\n";
      for(int k = 0; k < this->RHS.extent(0); k++){
        std::cout<<"  "<<k<<" : "<<this->RHS(k)<<std::endl;
      }
    }

    this->poisson_solve_pressure(1e-6, l_s);
    double time_post_poisson_solve = timer2.seconds();
    this->time_poisson_solve += time_post_poisson_solve - time_post_RHS;
    if(s_p == stopping_point::AFTER_FIRST_ITERATION){
      std::cout<<" pressure:\n";
      for(int j = 0; j < this->mesh.get_n_cells_y(); j++){
        for(int i = 0; i < this->mesh.get_n_cells_x(); i++){
          std::cout<<"  "<<i<<" "<<j<<" : "<<this->mesh.get_pressure(i, j)<<std::endl;
        }
      }
    }

    this->correct_velocity();
    double time_post_correct = timer2.seconds();
    this->time_corrector_step += time_post_correct - time_post_poisson_solve;
    if(s_p == stopping_point::AFTER_FIRST_ITERATION){
      std::cout<<" corrected velocity:\n";
      for(int j = 0; j < this->mesh.get_n_points_y(); j++){
        for(int i = 0; i < this->mesh.get_n_points_x(); i++){
          std::cout<<"  "<<i<<" "<<j<<" : "<<this->mesh.get_velocity_u(i, j)<<" , "<<this->mesh.get_velocity_v(i, j)<<std::endl;
        }
      }
    }

    if(ats == adaptative_time_step::ON){
      // CFL based adaptative time step
      double max_m_C = this->compute_global_courant_number();
      double adjusted_delta_t = this->delta_t;
      if(this->verbosity > 1){
        std::cout<<"    Computed global CFL = " << max_m_C << "; target max = " << this->max_C <<std::endl;
      }
      if(max_m_C > this->max_C){
        // decrease time step due to excessive CFL
        adjusted_delta_t = this->delta_t / (max_m_C * 100);
        if(this->verbosity > 1){
          std::cout<<"  - Decreased time step from " << this->delta_t << " to " << adjusted_delta_t <<std::endl;
        }
        this->delta_t = adjusted_delta_t;
      }else if(max_m_C < 0.06 * this->max_C){
        // increase time step if possible to accelerate computation
        adjusted_delta_t = this->delta_t * 1.06;
        if(this->verbosity > 1){
          std::cout<<"  + Increased time step from " << this->delta_t << " to " << adjusted_delta_t <<std::endl;
        }
        this->delta_t = adjusted_delta_t;
      }
    }

    if(s_p == stopping_point::AFTER_FIRST_ITERATION){
      std::cout<<"Stopping after first iteration.\n";
      return;
    }
  }

  double time3 = timer.seconds();
  if(this->verbosity > 0){
    std::cout<<std::endl;
    std::cout<<"Done."<<std::endl;
    std::cout<<"Total computation duration: " << time3 - time2 <<std::endl;
    std::cout<<std::endl;
    std::cout<<"Predictor step total duration: " << this->time_predictor_step <<" ("<<this->time_predictor_step * 100/ (time3 - time2)<<" %)"<<std::endl;
    std::cout<<"RHS assembly total duration: " << this->time_assemble_RHS <<" ("<<this->time_assemble_RHS * 100/ (time3 - time2)<<" %)"<<std::endl;
    std::cout<<"Linear solver for poisson equation total duration: " << this->time_poisson_solve <<" ("<<this->time_poisson_solve * 100/ (time3 - time2)<<" %)"<<std::endl;
    std::cout<<"Corrector step total duration: " << this->time_corrector_step <<" ("<<this->time_corrector_step * 100/ (time3 - time2)<<" %)"<<std::endl;
  }
  timer.reset();
}

// computation of laplacian matrix
void Solver::assemble_laplacian(){
  // initialize container for Laplacian
  const int m = this->mesh.get_n_cells_x();
  const int n = this->mesh.get_n_cells_y();
  const int mn = m * n;
  this->laplacian = Kokkos::View<double**>("laplacian", mn, mn);

  for(int j = 0; j < n; j++){
    for(int i = 0; i < m; i++){
      int k = this->mesh.cartesian_to_index(i, j, m, n);
      // initialize diagonal value
      int v = -4;

      // detect corners and borders
      if(i == 0 || i == m - 1)
      {
        ++v;
      }
      if(j == 0 || j == n - 1)
      {
        ++v;
      }

      // assign diagonal entry
      this->laplacian(k, k) = v;

      // assign unit entries
      if(i > 0)
      {
        this->laplacian(k, k-1) = 1;
      }
      if(i < m - 1)
      {
        this->laplacian(k, k + 1) = 1;
      }
      if(j > 0)
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
  for(uint64_t j = 1; j < this->mesh.get_n_points_y() - 1; j++){
    for(uint64_t i = 1; i < this->mesh.get_n_points_x() - 1; i++){
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
      uint64_t k = this->mesh.cartesian_to_index(i, j, this->mesh.get_n_points_x(), this->mesh.get_n_points_y());
      predicted_velocity(k, 0) = this->mesh.get_velocity_u(i, j) + this->delta_t * (this->nu * (dudx2 + dudy2) - (this->mesh.get_velocity_u(i, j) * dudx + v * dudy));
      predicted_velocity(k, 1) = this->mesh.get_velocity_v(i, j) + this->delta_t * (this->nu * (dvdx2 + dvdy2) - (this->mesh.get_velocity_v(i, j) * dvdx + u * dvdy));
    }
  }

  // assign interior predicted velocity vectors to mesh
  for(int j = 1; j < this->mesh.get_n_points_y() - 1; j++){
    for(int i = 1; i < this->mesh.get_n_points_x() - 1; i++){
      int k = this->mesh.cartesian_to_index(i, j, this->mesh.get_n_points_x(), this->mesh.get_n_points_y());
      this->mesh.set_velocity_u(i, j, predicted_velocity(k, 0));
      this->mesh.set_velocity_v(i, j, predicted_velocity(k, 1));
    }
  }
}

// implementation of the poisson RHS assembly
void Solver::assemble_poisson_RHS(){
  this->RHS = Kokkos::View<double*>("RHS", this->mesh.get_n_cells_x() * this->mesh.get_n_cells_y());
  double factor = this->rho / this->delta_t;
  for(int j = 0; j < this->mesh.get_n_cells_y(); j++){
    for(int i = 0; i < this->mesh.get_n_cells_x(); i++){
      double u_r = this->mesh.get_velocity_u(i+1, j) + this->mesh.get_velocity_u(i+1, j+1);
      double u_l = this->mesh.get_velocity_u(i, j) + this->mesh.get_velocity_u(i, j+1);
      double v_t = this->mesh.get_velocity_v(i, j+1) + this->mesh.get_velocity_v(i+1, j+1);
      double v_b = this->mesh.get_velocity_v(i, j) + this->mesh.get_velocity_v(i+1, j);
      double inv_h = 1 / (2 * this->mesh.get_cell_size());
      uint64_t k = this->mesh.cartesian_to_index(i, j, this->mesh.get_n_cells_x(), this->mesh.get_n_cells_y());
      this->RHS(k) = factor * ((u_r - u_l + v_t - v_b) * inv_h);
    }
  }
}

Kokkos::View<double*> Solver::conjugate_gradient_solve(double r_tol){
  // initialize approximate solution with null guess
  uint64_t n_x = this->mesh.get_n_cells_x();
  uint64_t n_y = this->mesh.get_n_cells_y();
  Kokkos::View<double*> x("x", n_x * n_y);

  // initialize residual
  Kokkos::View<double*> residual("residual", n_x * n_y);
  Kokkos::deep_copy (residual, this->RHS);
  double factor = 1 / (this->mesh.get_cell_size() * this->mesh.get_cell_size());
  KokkosBlas::gemv("N", - factor, this->laplacian, x, 1, residual);
  double rms2 = KokkosBlas::dot(residual, residual);

  // compute square norm of RHS for relative error
  double RHS2 = KokkosBlas::dot(this->RHS, this->RHS);

  // terminate early when possible
  if(rms2 / RHS2 < r_tol){
    return x;
  }

  // initialize first direction of the conjugate basis with residual
  Kokkos::View<double*> direction("direction", n_x * n_y);
  Kokkos::deep_copy (direction, residual);

  // storage for laplacian dot direction intermediate vector
  Kokkos::View<double*> intermediate("intermediate", n_x * n_y);

  // iterate for at most the dimension of the matrix
  for(int k = 0; k < this->laplacian.extent(0); k++){
    // compute step length
    KokkosBlas::gemv("N", factor, this->laplacian, direction, 0, intermediate);
    double alpha = rms2 / KokkosBlas::dot(direction, intermediate);

    // update solution
    KokkosBlas::axpy(alpha, direction, x);

    // update residual
    KokkosBlas::axpy(-alpha, intermediate, residual);
    double new_rms2 = KokkosBlas::dot(residual, residual);

    // terminate early when possible
    if(new_rms2 / RHS2 < r_tol){
      return x;
    }

    // compute new direction
    KokkosBlas::axpby(1, residual, - new_rms2 / rms2, direction);

    // update residual squared L2
    rms2 = new_rms2;
    if(this->verbosity > 1){
      std::cout<<"  relative error squared: "<<rms2 / RHS2<<std::endl;
    }
  }

  // return approximate solution
  return x;
}

// implementation of the Poisson equation solver
void Solver::poisson_solve_pressure(double r_tol, linear_solver l_s){
  if(l_s == linear_solver::CONJUGATE_GRADIENT){
    this->mesh.set_pressure(this->conjugate_gradient_solve(r_tol));
  }
  else{
    std::cout<<"  pressure Poisson equation ignored"<<std::endl;
  }
}

// implementation of the corrector step
void Solver::correct_velocity(){
  double factor = .5 / this->mesh.get_cell_size();
  double t_to_r = this->delta_t / this->rho;
  for(uint64_t j = 1; j < this->mesh.get_n_points_y() - 1; j++){
    for(uint64_t i = 1; i < this->mesh.get_n_points_x() - 1; i++){
      double p_ur = this->mesh.get_pressure(i, j);
      double p_ul = this->mesh.get_pressure(i-1, j);
      double p_ll = this->mesh.get_pressure(i-1, j-1);
      double p_lr = this->mesh.get_pressure(i, j-1);
      this->mesh.set_velocity_u(i, j, (this->mesh.get_velocity_u(i, j) - t_to_r * (p_ur - p_ul + p_lr - p_ll) * factor));
      this->mesh.set_velocity_v(i, j, (this->mesh.get_velocity_v(i, j) - t_to_r * (p_ur - p_lr + p_ul - p_ll) * factor));
    }
  }
}

double Solver::compute_cell_courant_number(int i, int j){
  return (this->mesh.get_velocity_u(i, j) + this->mesh.get_velocity_v(i, j)) * this->delta_t / this->mesh.get_cell_size();
}

double Solver::compute_global_courant_number(){
  double max_C = std::numeric_limits<int>::min();
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
