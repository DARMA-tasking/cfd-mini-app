#include "boundary_conditions.h"

void BoundaryConditions::apply_velocity_bc(){
  const uint64_t m = this->mesh_chunk.get_n_points_x() - 1;
  const uint64_t mm1 = m - 1;
  const uint64_t n = this->mesh_chunk.get_n_points_y() - 1;
  const uint64_t nm1 = n - 1;
  for(uint64_t i = 0; i < m; i++){
    this->mesh_chunk.set_velocity_x(i, nm1, this->v_x_t);
    this->mesh_chunk.set_velocity_y(i, nm1, this->v_y_t);
    this->mesh_chunk.set_velocity_x(i, 0, this->v_x_b);
    this->mesh_chunk.set_velocity_y(i, 0, this->v_y_b);
  }
  for(uint64_t j = 0; j < n; j++){
    this->mesh_chunk.set_velocity_x(0, j, this->v_x_l);
    this->mesh_chunk.set_velocity_y(0, j, this->v_y_l);
    this->mesh_chunk.set_velocity_x(mm1, j, this->v_x_r);
    this->mesh_chunk.set_velocity_y(mm1, j, this->v_y_r);
  }
}

double BoundaryConditions::get_velocity_bc_max_norm(){
  return std::
    sqrt(std::max({
	     this->v_x_t * this->v_x_t + this->v_y_t * this->v_y_t,
	     this->v_x_b * this->v_x_b + this->v_y_b * this->v_y_b,
	     this->v_x_l * this->v_x_l + this->v_y_l * this->v_y_l,
	     this->v_x_r * this->v_x_r + this->v_y_r * this->v_y_r}));
}
