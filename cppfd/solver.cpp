#include "solver.h"

#include <Kokkos_Core.hpp>

#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_nrm2_squared.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_gauss_seidel.hpp>

// main function to solve N-S equation on time steps
void Solver::solve(stopping_point s_p, linear_solver l_s, adaptative_time_step ats){
  // begin simulation
  if(this->verbosity >= 1)
    std::cout << "Initial timestep: "
	      << this->delta_t
	      << ", "
	      <<"Final time: "
	      << this->t_final
	      << std::endl;

  if(this->verbosity >= 1){
    std::cout << "Applying velocity boundary conditions..." << std::endl;
  }
  this->boundary_conditions.apply_velocity_bc();
  if(s_p == stopping_point::AFTER_BOUNDARY_CONDITION){
    for(int j = 0; j < this->mesh_chunk->get_n_points_y(); j++){
      for(int i = 0; i < this->mesh_chunk->get_n_points_x(); i++){
	std::cout << i << " " << j << " : " << this->mesh_chunk->get_velocity_x(i, j) << " , " << this->mesh_chunk->get_velocity_y(i, j) << std::endl;
      }
    }
    return;
  }

  // compute Reynolds number from input parameters
  double velocity_bc_max = this->boundary_conditions.get_velocity_bc_max_norm();
  double l = this->mesh_chunk->get_cell_size() * std::max(this->mesh_chunk->get_n_cells_x(), this->mesh_chunk->get_n_cells_y());
  this->Re = velocity_bc_max * this->one / this->nu;
  if(this->verbosity > 0){
    std::cout << "Reynolds number : " << this->Re << std::endl;
  }

  Kokkos::Timer timer;
  double time1 = timer.seconds();
  if(this->verbosity > 0){
    std::cout << std::endl;
    std::cout << "Computing Laplacian matrix..." << std::endl;
  }

  double lap_d = this->assemble_Laplacian();
  double time2 = timer.seconds();
  if(this->verbosity > 0){
    std::cout << "Laplacian density: " << std::setprecision (4)
	      << 100. * lap_d << " %\n";
    std::cout << "Laplacian computation duration: " << time2 - time1 << std::endl;
  }
  if(s_p == stopping_point::AFTER_LAPLACIAN){
    for (uint64_t j = 0; j < this->Laplacian.numRows(); j++) {
      auto row = this->Laplacian.row(j);
      uint64_t i = 0;
      for (uint64_t k = 0; k < row.length; k++) {
	auto val = row.value(k);
	auto col = row.colidx(k);
	while (i++ < col)
	  std::cout << 0 << " ";
	std::cout << val << " ";
      }
      while (i++ < this->Laplacian.numCols())
	std::cout << 0 << " ";
      std::cout << std::endl;
    }
    return;
  }

  if(this->verbosity > 0){
    std::cout << std::endl;
    std::cout << "Running simulation..." << std::endl;
  }

  // initialize counters
  double t = 0.;
  double time_predictor_step = 0.;
  double time_assemble_RHS = 0.;
  double time_poisson_solve = 0.;
  double time_corrector_step = 0.;

  //start iterating on time steps
  int iteration = 0;
  while(t < this->t_final){
    ++iteration;
    t += this->delta_t;

    if(this->verbosity > 1){
      std::cout << "Entering iteration " << iteration << " at time " << t << std::endl;
    }

    // run all necessary solving steps
    Kokkos::Timer timer2;

    // predict velocity without pressure
    this->predict_velocity();
    double time_post_predict = timer2.seconds();
    time_predictor_step += time_post_predict;
    if(s_p == stopping_point::AFTER_FIRST_ITERATION){
      std::cout << " predicted velocity:\n";
      for(uint64_t j = 0; j < this->mesh_chunk->get_n_points_y(); j++){
	for(uint64_t i = 0; i < this->mesh_chunk->get_n_points_x(); i++){
	  std::cout << "  " << i << " " << j << " : " << this->mesh_chunk->get_velocity_x(i, j) << " , " << this->mesh_chunk->get_velocity_y(i, j) << std::endl;
	}
      }
    }

    // assemble PPE RHS
    this->assemble_poisson_RHS();
    double time_post_RHS = timer2.seconds();
    time_assemble_RHS += time_post_RHS - time_post_predict;
    if(s_p == stopping_point::AFTER_FIRST_ITERATION){
      std::cout << " RHS:\n";
      for(uint64_t k = 0; k < this->RHS.extent(0); k++){
	std::cout << "  " << k << " : " << this->RHS(k) << std::endl;
      }
    }

    // solve PPE
    this->poisson_solve_pressure(1e-2, l_s);
    double time_post_poisson_solve = timer2.seconds();
    time_poisson_solve += time_post_poisson_solve - time_post_RHS;
    if(s_p == stopping_point::AFTER_FIRST_ITERATION){
      std::cout << " pressure:\n";
      for(uint64_t j = 0; j < this->mesh_chunk->get_n_cells_y(); j++){
	for(uint64_t i = 0; i < this->mesh_chunk->get_n_cells_x(); i++){
	  std::cout << "  " << i << " " << j << " : " << this->mesh_chunk->get_pressure(i, j) << std::endl;
	}
      }
    }

    // correct velocity with pressure
    this->correct_velocity();
    double time_post_correct = timer2.seconds();
    time_corrector_step += time_post_correct - time_post_poisson_solve;
    if(s_p == stopping_point::AFTER_FIRST_ITERATION){
      std::cout << " corrected velocity:\n";
      for(uint64_t j = 0; j < this->mesh_chunk->get_n_points_y(); j++){
	for(uint64_t i = 0; i < this->mesh_chunk->get_n_points_x(); i++){
	  std::cout << "  " << i << " " << j << " : " << this->mesh_chunk->get_velocity_x(i, j) << " , " << this->mesh_chunk->get_velocity_y(i, j) << std::endl;
	}
      }
    }

    if(ats == adaptative_time_step::ON){
      // CFL based adaptative time step
      double max_m_C = this->compute_global_courant_number();
      double adjusted_delta_t = this->delta_t;
      if(this->verbosity > 1){
	std::cout << "    Computed global CFL = " << max_m_C << "; target max = " << this->max_C << std::endl;
      }
      if(max_m_C > this->max_C){
	// decrease time step due to excessive CFL
	adjusted_delta_t = this->delta_t / (max_m_C * 100);
	if(this->verbosity > 1){
	  std::cout << "  - Decreased time step from " << this->delta_t << " to " << adjusted_delta_t << std::endl;
	}
	this->delta_t = adjusted_delta_t;
      }else if(max_m_C < 0.06 * this->max_C){
	// increase time step if possible to accelerate computation
	adjusted_delta_t = this->delta_t * 1.06;
	if(this->verbosity > 1){
	  std::cout << "  + Increased time step from " << this->delta_t << " to " << adjusted_delta_t << std::endl;
	}
	this->delta_t = adjusted_delta_t;
      }
    }

    if(s_p == stopping_point::AFTER_FIRST_ITERATION){
      std::cout << "Stopping after first iteration.\n";
      return;
    }
  }

  double time3 = timer.seconds();
  if(this->verbosity > 0){
    std::cout << std::endl;
    std::cout << "Done."
	      << std::endl;
    std::cout << "Total computation duration: "
	      << time3 - time2
	      << std::endl;
    std::cout << std::endl;
    std::cout << "Predictor step total duration: "
	      << time_predictor_step
	      << " ("
	      << time_predictor_step * 100 / (time3 - time2)
	      << " %)"
	      << std::endl;
    std::cout << "RHS assembly total duration: "
	      << time_assemble_RHS
	      << " ("
	      << time_assemble_RHS * 100 / (time3 - time2)
	      << " %)"
	      << std::endl;
    std::cout << "Linear solver for poisson equation total duration: "
	      << time_poisson_solve
	      << " ("
	      << time_poisson_solve * 100 / (time3 - time2)
	      << " %)"
	      << std::endl;
    std::cout << "Corrector step total duration: "
	      << time_corrector_step
	      << " ("
	      << time_corrector_step * 100 / (time3 - time2)
	      << " %)"
	      << std::endl;
  }
  timer.reset();
}

void Solver::assign_CRS_entry(uint64_t &idx,
			      bool &first_in_row,
			      const uint64_t k,
			      const uint64_t offset,
			      const double value,
			      Kokkos::View<int64_t*> row_ptrs,
			      Kokkos::View<int64_t*> col_ids,
			      Kokkos::View<double*> values){

  // treat first entry in row differently
  if (first_in_row)
    {
      // new row starts at current index
      row_ptrs[k] = idx;
      first_in_row =  false;
    }

  // assign off-diagonal column index
  col_ids[idx] = k + offset;

  // assign unit value
  values[idx++] = value;
}

// computation of Laplacian matrix
double Solver::assemble_Laplacian(){
  // initialize containers for sparse storage of Laplacian
  const uint64_t m = this->mesh_chunk->get_n_cells_x();
  const uint64_t mm1 = m - 1;
  const uint64_t n = this->mesh_chunk->get_n_cells_y();
  const uint64_t nm1 = n - 1;
  const uint64_t mn = m * n;
  const uint64_t nnz = 5 * mn - 2 * (m + n);
  Kokkos::View<int64_t*> row_ptrs("row pointers", mn + 1);
  Kokkos::View<int64_t*> col_ids("column indices", nnz);
  Kokkos::View<double*> values("values", nnz);

  // iterate over m*n cells to construct mn*mn Laplacian
  uint64_t idx = 0;
  for(uint64_t j = 0; j < n; j++){
    for(uint64_t i = 0; i < m; i++){
      uint64_t k = this->mesh_chunk->Cartesian_to_index(i, j, m, n);
      bool first_in_row = true;
      // assign below diagonal entries when relevant
      if(j > 0)
	this->assign_CRS_entry(idx,
			       first_in_row,
			       k, -m, this->one,
			       row_ptrs, col_ids, values);
      if(i > 0)
	this->assign_CRS_entry(idx,
			       first_in_row,
			       k, -1, this->one,
			       row_ptrs, col_ids, values);

      // assign diagonal entry
      double v = -4 * this->one;
      if(i == 0 || i == mm1)
	{
	  ++v;
	}
      if(j == 0 || j == nm1)
	{
	  ++v;
	}
      this->assign_CRS_entry(idx,
			     first_in_row,
			     k, 0, v,
			     row_ptrs, col_ids, values);

      // assign above diagonal entries when relevant
      if(i < mm1)
	this->assign_CRS_entry(idx,
			       first_in_row,
			       k, 1, this->one,
			       row_ptrs, col_ids, values);
      if(j < nm1)
	this->assign_CRS_entry(idx,
			       first_in_row,
			       k, m, this->one,
			       row_ptrs, col_ids, values);
    }
  }

  // append NNZ at end of row pointers
  row_ptrs[mn] = nnz;

  // instantiate Laplacian as CRS matrix
  typename matrix_type::staticcrsgraph_type mygraph(col_ids, row_ptrs);
  this->Laplacian = matrix_type("Laplacian", mn, values, mygraph);

  // return density of sparse Laplacian
  return nnz / (double) (mn * mn);
}

// implementation of the predictor step
void Solver::predict_velocity(){
  Kokkos::View<double*[2]> v_star("predicted velocity", this->mesh_chunk->get_n_points_x() * this->mesh_chunk->get_n_points_y());
  const uint64_t m = this->mesh_chunk->get_n_points_x();
  const uint64_t mm1 = m - 1;
  const uint64_t n = this->mesh_chunk->get_n_points_y();
  const uint64_t nm1 = n - 1;

  // compute common factors
  const double h = this->mesh_chunk->get_cell_size();
  const double inv_2sz = 1. / (2. * h);
  const double inv_sz2 = 1. / (h * h);

  // predict velocity components using finite difference discretization
  for(uint64_t j = 1; j < nm1; j++){
    for(uint64_t i = 1; i < mm1; i++){
      // retrieve velocity at stencil nodes only once
      double v_x_ij = this->mesh_chunk->get_velocity_x(i, j);
      double v_x_ij_l = this->mesh_chunk->get_velocity_x(i - 1, j);
      double v_x_ij_r = this->mesh_chunk->get_velocity_x(i + 1, j);
      double v_x_ij_t = this->mesh_chunk->get_velocity_x(i, j + 1);
      double v_x_ij_b = this->mesh_chunk->get_velocity_x(i, j - 1);
      double v_y_ij = this->mesh_chunk->get_velocity_y(i, j);
      double v_y_ij_l = this->mesh_chunk->get_velocity_y(i - 1, j);
      double v_y_ij_r = this->mesh_chunk->get_velocity_y(i + 1, j);
      double v_y_ij_t = this->mesh_chunk->get_velocity_y(i, j + 1);
      double v_y_ij_b = this->mesh_chunk->get_velocity_y(i, j - 1);

      // factors needed to predict new x component
      double v_y = .25 * (v_y_ij_l + v_y_ij + v_y_ij_t);
      double dudx = inv_2sz * v_x_ij * (v_x_ij_r - v_x_ij_l);
      double dudy = inv_2sz * v_y * (v_x_ij_t - v_x_ij_b);
      double dudx2 = inv_sz2 * (v_x_ij_l - 2 * v_x_ij + v_x_ij_r);
      double dudy2 = inv_sz2 * (v_x_ij_b - 2 * v_x_ij + v_x_ij_t);

      // factors needed to predict new y component
      double v_x = .25 * (v_x_ij_b + v_x_ij + v_x_ij_t);
      double dvdy = inv_2sz * v_y_ij * (v_y_ij_r - v_y_ij_l);
      double dvdx = inv_2sz * v_x * (v_y_ij_t - v_y_ij_b);
      double dvdx2 = inv_sz2 * (v_y_ij_l - 2 * v_y_ij + v_y_ij_r);
      double dvdy2 = inv_sz2 * (v_y_ij_b - 2 * v_y_ij + v_y_ij_t);

      // assign predicted u and v components to predicted_velocity storage
      uint64_t k = this->mesh_chunk->Cartesian_to_index(i, j, m, n);
      v_star(k, 0) = v_x_ij + this->delta_t * (this->nu * (dudx2 + dudy2) - (v_x_ij * dudx + v_y * dudy));
      v_star(k, 1) = v_y_ij + this->delta_t * (this->nu * (dvdx2 + dvdy2) - (v_y_ij * dvdx + v_x * dvdy));
    }
  }

  // assign interior predicted velocity vectors to mesh
  for(int j = 1; j < nm1; j++){
    for(int i = 1; i < mm1; i++){
      int k = this->mesh_chunk->Cartesian_to_index(i, j, m, n);
      this->mesh_chunk->set_velocity_x(i, j, v_star(k, 0));
      this->mesh_chunk->set_velocity_y(i, j, v_star(k, 1));
    }
  }
}

// implementation of the poisson RHS assembly
void Solver::assemble_poisson_RHS(){
  this->RHS = Kokkos::View<double*>("RHS", this->mesh_chunk->get_n_cells_x() * this->mesh_chunk->get_n_cells_y());
  double factor = this->rho / this->delta_t / (2 * this->mesh_chunk->get_cell_size());
  for(int j = 0; j < this->mesh_chunk->get_n_cells_y(); j++){
    for(int i = 0; i < this->mesh_chunk->get_n_cells_x(); i++){
      double u_r = this->mesh_chunk->get_velocity_x(i+1, j) + this->mesh_chunk->get_velocity_x(i+1, j+1);
      double u_l = this->mesh_chunk->get_velocity_x(i, j) + this->mesh_chunk->get_velocity_x(i, j+1);
      double v_t = this->mesh_chunk->get_velocity_y(i, j+1) + this->mesh_chunk->get_velocity_y(i+1, j+1);
      double v_b = this->mesh_chunk->get_velocity_y(i, j) + this->mesh_chunk->get_velocity_y(i+1, j);
      uint64_t k = this->mesh_chunk->Cartesian_to_index(i, j, this->mesh_chunk->get_n_cells_x(), this->mesh_chunk->get_n_cells_y());
      this->RHS(k) = factor * (u_r - u_l + v_t - v_b);
    }
  }
}

Kokkos::View<double*> Solver::conjugate_gradient_solve(double r_tol){
  // initialize approximate solution with null guess
  uint64_t n_x = this->mesh_chunk->get_n_cells_x();
  uint64_t n_y = this->mesh_chunk->get_n_cells_y();
  uint64_t mn = n_x * n_y;
  Kokkos::View<double*> x("x", mn);

  // initialize scalar factor of Laplacian
  double factor = this->one / (this->mesh_chunk->get_cell_size() * this->mesh_chunk->get_cell_size());

  // initialize residual
  Kokkos::View<double*> residual("residual", mn);
  Kokkos::deep_copy (residual, this->RHS);
  KokkosSparse::spmv("N", - factor, this->Laplacian, x, this->one, residual);
  double rms2 = KokkosBlas::nrm2_squared(residual);

  // terminate early when possible
  double RHS2 = KokkosBlas::nrm2_squared(this->RHS);
  if(rms2 / RHS2 < r_tol){
    return x;
  }

  // initialize first direction of the conjugate basis with residual
  Kokkos::View<double*> direction("direction", mn);
  Kokkos::deep_copy (direction, residual);

  // storage for Laplacian dot direction intermediate vector
  Kokkos::View<double*> intermediate("intermediate", mn);

  // iterate for at most the dimension of the matrix
  for(uint64_t k = 0; k < mn; k++){
    // compute step length
    KokkosSparse::spmv("N", factor, this->Laplacian, direction, this->zero, intermediate);
    double alpha = rms2 / KokkosBlas::dot(direction, intermediate);

    // update solution
    KokkosBlas::axpy(alpha, direction, x);

    // update residual
    KokkosBlas::axpy(-alpha, intermediate, residual);
    double new_rms2 = KokkosBlas::nrm2_squared(residual);

    // terminate early when possible
    if(new_rms2 / RHS2 < r_tol){
      return x;
    }

    // compute new direction
    KokkosBlas::axpby(this->one, residual, - new_rms2 / rms2, direction);

    // update residual squared L2
    rms2 = new_rms2;
    if(this->verbosity > 1){
      std::cout << "  relative error squared: " <<rms2 / RHS2<< std::endl;
    }
  }

  // return approximate solution
  return x;
}

Kokkos::View<double*> Solver::gauss_seidel_solve(double r_tol, int max_it, int n_sweeps){
  // initialize approximate solution with null guess
  uint64_t n_x = this->mesh_chunk->get_n_cells_x();
  uint64_t n_y = this->mesh_chunk->get_n_cells_y();
  uint64_t mn = n_x * n_y;
  Kokkos::View<double*> x("x", mn);

  // initialize scalar factor of Laplacian
  double factor = this->one / (this->mesh_chunk->get_cell_size() * this->mesh_chunk->get_cell_size());

  // initialize residual
  Kokkos::View<double*> residual("residual", mn);
  Kokkos::deep_copy (residual, this->RHS);
  KokkosSparse::spmv("N", - factor, this->Laplacian, x, this->one, residual);
  double rms2 = KokkosBlas::nrm2_squared(residual);

  // terminate early when possible
  double RHS2 = KokkosBlas::nrm2_squared(this->RHS);
  if(rms2 / RHS2 < r_tol){
    return x;
  }

  // create handle to Kokkos Gauss-Seidel kernel
  Kokkos::View<double*> scaled_values("scaled values", this->Laplacian.nnz());
  KokkosKernels::Experimental::
    KokkosKernelsHandle<int64_t, int64_t, double, exec_space, mem_space, mem_space> handle;
  handle.create_gs_handle(KokkosSparse::GS_DEFAULT);
  KokkosSparse::Experimental::
    gauss_seidel_symbolic(&handle, mn, mn,
			  this->Laplacian.graph.row_map,
			  this->Laplacian.graph.entries,
			  true);
  KokkosBlas::scal(scaled_values, factor, this->Laplacian.values);
  KokkosSparse::Experimental::
    gauss_seidel_numeric(&handle, mn, mn,
			 this->Laplacian.graph.row_map,
			 this->Laplacian.graph.entries,
			 scaled_values,
			 true);

  // iteratively solve
  bool first_iter = true;
  while(max_it-- > 0 && rms2 / RHS2 > r_tol){
    // perform pairs of forward then backward sweeps
    KokkosSparse::Experimental::
      symmetric_gauss_seidel_apply(&handle, mn, mn,
				   this->Laplacian.graph.row_map,
				   this->Laplacian.graph.entries,
				   scaled_values,
				   x,
				   this->RHS,
				   first_iter, first_iter,
				   this->one, n_sweeps);
    first_iter = false;

    // terminate early when possible
    Kokkos::deep_copy(residual, this->RHS);
    KokkosSparse::spmv("N", - factor, this->Laplacian, x, this->one, residual);
    rms2 = KokkosBlas::nrm2_squared(residual);
    if(rms2 / RHS2 < r_tol){
      return x;
    }
  }

  // destroy handle to Kokkos kernel
  handle.destroy_gs_handle();

  // return approximate solution
  return x;
}

// implementation of the Poisson equation solver
void Solver::poisson_solve_pressure(double r_tol, linear_solver l_s){
  if(l_s == linear_solver::CONJUGATE_GRADIENT){
    this->mesh_chunk->set_pressure(this->conjugate_gradient_solve(r_tol));
  } else if(l_s == linear_solver::GAUSS_SEIDEL){
    this->mesh_chunk->set_pressure(this->gauss_seidel_solve(r_tol, 5, 10));
  } else{
    std::cout << "  Pressure Poisson equation ignored." << std::endl;
  }
}

// implementation of the corrector step
void Solver::correct_velocity(){
  double factor = .5 / this->mesh_chunk->get_cell_size();
  double t_to_r = this->delta_t / this->rho;
  for(uint64_t j = 1; j < this->mesh_chunk->get_n_points_y() - 1; j++){
    for(uint64_t i = 1; i < this->mesh_chunk->get_n_points_x() - 1; i++){
      double p_tl = this->mesh_chunk->get_pressure(i - 1, j);
      double p_tr = this->mesh_chunk->get_pressure(i, j);
      double p_bl = this->mesh_chunk->get_pressure(i-1, j - 1);
      double p_br = this->mesh_chunk->get_pressure(i, j - 1);
      this->mesh_chunk->set_velocity_x(i, j, (this->mesh_chunk->get_velocity_x(i, j) - t_to_r * (p_tr - p_tl + p_br - p_bl) * factor));
      this->mesh_chunk->set_velocity_y(i, j, (this->mesh_chunk->get_velocity_y(i, j) - t_to_r * (p_tr - p_br + p_tl - p_bl) * factor));
    }
  }
}

double Solver::compute_cell_courant_number(int i, int j){
  return (this->mesh_chunk->get_velocity_x(i, j) + this->mesh_chunk->get_velocity_y(i, j)) * this->delta_t / this->mesh_chunk->get_cell_size();
}

double Solver::compute_global_courant_number(){
  double max_C = std::numeric_limits<int>::min();
  for(int j = 1; j <= this->mesh_chunk->get_n_cells_y(); j++){
    for(int i = 1; i <= this->mesh_chunk->get_n_cells_x(); i++){
      double c = this->compute_cell_courant_number(i, j);
      if(c > max_C){
	max_C = c;
      }
    }
  }
  return max_C;
}
