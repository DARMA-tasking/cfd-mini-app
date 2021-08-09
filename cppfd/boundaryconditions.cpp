#include"boundaryconditions.h"

BoundaryConditions::BoundaryConditions(Mesh& m, std::map<std::string, double> velocity_values)
  : mesh(m)
  , u_top(velocity_values["u_top"])
  , v_top(velocity_values["v_top"])
  , u_bot(velocity_values["u_bot"])
  , v_bot(velocity_values["v_bot"])
  , u_left(velocity_values["u_left"])
  , v_left(velocity_values["v_left"])
  , u_right(velocity_values["u_right"])
  , v_right(velocity_values["v_right"])
  {}

void BoundaryConditions::apply_velocity_bc()
{
  for(int i = 0; i < this->mesh.get_n_points_x(); i++)
  {
    this->mesh.set_velocity_u(i, this->mesh.get_n_cells_y(), this->u_top);
    this->mesh.set_velocity_v(i, this->mesh.get_n_cells_y(), this->v_top);
    this->mesh.set_velocity_u(i, 0, this->u_bot);
    this->mesh.set_velocity_v(i, 0, this->v_bot);
  }
  for(int j = 0; j < this->mesh.get_n_points_x(); j++)
  {
    this->mesh.set_velocity_u(0, j, this->u_left);
    this->mesh.set_velocity_v(0, j, this->v_left);
    this->mesh.set_velocity_u(this->mesh.get_n_cells_x(), j, this->u_right);
    this->mesh.set_velocity_v(this->mesh.get_n_cells_x(), j, this->v_right);
  }
}

double BoundaryConditions::get_velocity_bc_max_norm()
{
  return std::sqrt(std::max({this->u_top * this->u_top + this->v_top * this->v_top,
                this->u_bot * this->u_bot + this->v_bot * this->v_bot,
                this->u_left * this->u_left + this->v_left * this->v_left,
                this->u_right * this->u_right + this->v_right * this->v_right}));
}
