#include "mesh_chunk.h"
#include "parallel_mesh.h"

#include <iostream>
#include <array>
#include <Kokkos_Core.hpp>

#ifdef OUTPUT_VTK_FILES
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkUniformGrid.h>
#include <vtkSmartPointer.h>
#include <vtkXMLImageDataWriter.h>
#endif

MeshChunk::
MeshChunk(ParallelMesh* pp_mesh, uint64_t n_x, uint64_t n_y, double cell_size,
	  const std::map<PointIndexEnum, PointTypeEnum>& point_types,
    uint64_t n_ch_glob_x, uint64_t n_ch_glob_y,
    uint64_t chunk_position_global_x, uint64_t chunk_position_global_y,
	  double o_x, double o_y)
  : parent_parallel_mesh(pp_mesh)
	, n_cells_x(n_x)
  , n_cells_y(n_y)
  , h(cell_size)
  , n_chunks_global_x(n_ch_glob_x)
  , n_chunks_global_y(n_ch_glob_y)
  , global_position({chunk_position_global_x, chunk_position_global_y})
  , origin({o_x, o_y}){

  // instantiate internal containers
  this->point_type = Kokkos::
    View<PointTypeEnum**>("type", n_x + 1, n_y + 1);
  this->pressure = Kokkos::
    View<double*>("pressure", n_x * n_y);
  this->velocity = Kokkos::
    View<double**[2]>("velocity", n_x + 1, n_y + 1);

  // set boundary point types
  for (const auto& kv : point_types){
    // interior points
    if (kv.first == PointIndexEnum::INTERIOR)
      for(uint64_t j = 1; j < n_y; j++)
	for(uint64_t i = 1; i < n_x; i++)
	  this->point_type(i, j) = kv.second;

    // non-corner edge points in QUAD8 order
    else if (kv.first == PointIndexEnum::EDGE_0)
      for(uint64_t i = 1; i < n_x; i++)
	this->point_type(i, 0) = kv.second;
    else if (kv.first == PointIndexEnum::EDGE_1)
      for(uint64_t j = 1; j < n_y; j++)
	this->point_type(n_x, j) = kv.second;
    else if (kv.first == PointIndexEnum::EDGE_2)
      for(uint64_t i = 1; i < n_x; i++)
	this->point_type(i, n_y) = kv.second;
    else if (kv.first == PointIndexEnum::EDGE_3)
      for(uint64_t j = 1; j < n_y; j++)
	this->point_type(0, j) = kv.second;

    // corner points in QUAD4 order
    else if (kv.first == PointIndexEnum::CORNER_0)
      this->point_type(0, 0) = kv.second;
    else if (kv.first == PointIndexEnum::CORNER_1)
      this->point_type(n_x, 0) = kv.second;
    else if (kv.first == PointIndexEnum::CORNER_2)
      this->point_type(n_x, n_y) = kv.second;
    else if (kv.first == PointIndexEnum::CORNER_3)
      this->point_type(0, n_y) = kv.second;
  }
}

std::array<uint64_t,2> MeshChunk::
index_to_Cartesian(uint64_t k, uint64_t n, uint64_t nmax) const{
  // Return invalid value when index is out of bounds
  if (k >= nmax)
    return  {(uint64_t) -1, (uint64_t) -1};
  else{
    std::ldiv_t dv = std::ldiv(k, n);
    return {(uint64_t) dv.rem, (uint64_t) dv.quot};
  }
}

uint64_t MeshChunk::
Cartesian_to_index(uint64_t i, uint64_t j, uint64_t ni, uint64_t nj) const{
  // Return invalid value when coordinates are out of bounds
  if(i >= ni || j >= nj)
    return -1;
  else
    return j * ni + i;
}

void MeshChunk::
set_point_type(uint64_t i, uint64_t j, PointTypeEnum t){
  if(i < this->get_n_points_x() && j < this->get_n_points_y())
    this->point_type(i, j) = t;
}

PointTypeEnum MeshChunk::
get_point_type(uint64_t i, uint64_t j) const{
  // Return invalid type when indices are out of bounds
  if(i < this->get_n_points_x() && j < this->get_n_points_y())
    return this->point_type(i, j);
  else
    return PointTypeEnum::INVALID;
}

void MeshChunk::
set_velocity_x(uint64_t i, uint64_t j, double u){
  if(i < this->get_n_points_x() && j < this->get_n_points_y())
    this->velocity(i, j, 0) = u;
}

void MeshChunk::
set_velocity_y(uint64_t i, uint64_t j, double v){
  if(i < this->get_n_points_x() && j < this->get_n_points_y())
    this->velocity(i, j, 1) = v;
}

double MeshChunk::
get_velocity_x(int64_t i, int64_t j) {
	//std::cout << "////// getting velocity x " << "(i, j) : (" << i << ", " << j << ")" << '\n';
	if(i < this->get_n_points_x() && j < this->get_n_points_y() && i > 0 && j > 0){
		//std::cout << "//// Owned or shared point" << '\n';
		if((this->get_point_type(i, j) == PointTypeEnum::INTERIOR) || (this->get_point_type(i, j) == PointTypeEnum::SHARED_OWNED))
    	return this->velocity(i, j, 0);
		else if(this->get_point_type(i, j) == PointTypeEnum::BOUNDARY){
			if(this->get_point_type(i, j) == PointTypeEnum::BOUNDARY){
				std::string boundary;
				if(j == this->get_n_points_y() - 1){
					boundary = "v_x_t";
				}
				if(i == this->get_n_points_x() - 1){
					boundary = "v_x_r";
				}
				if(i == 0){
					boundary = "v_x_l";
				}
				if(j == 0){
					boundary = "v_x_b";
				}
				return this->parent_parallel_mesh->get_boundary_velocity_value(boundary);
			}
		}

	}
	else{
		//std::cout << "//// Not owned" << '\n';
		if(this->get_point_type(i, j) == PointTypeEnum::BOUNDARY){
			std::string boundary;
			if(j == this->get_n_points_y() - 1){
				boundary = "v_x_t";
			}
			if(i == this->get_n_points_x() - 1){
				boundary = "v_x_r";
			}
			if(i == 0){
				boundary = "v_x_l";
			}
			if(j == 0){
				boundary = "v_x_b";
			}
			return this->parent_parallel_mesh->get_boundary_velocity_value(boundary);
		}
		int64_t chunk_position_x = this->global_position[0];
		int64_t chunk_position_y = this->global_position[1];
		uint64_t local_point_coordinate_x = i;
		uint64_t local_point_coordinate_y = j;
		if(i == this->get_n_points_x() + 1){
			chunk_position_x = this->global_position[0] + 1;
			local_point_coordinate_x = 1;
		}
		if(j == this->get_n_points_y() + 1){
			chunk_position_y = this->global_position[1] + 1;
			local_point_coordinate_y = 1;
		}
		if(i == 0){
			chunk_position_x = this->global_position[0] - 1;
			local_point_coordinate_x = this->parent_parallel_mesh->get_n_points_x_mesh_chunk(chunk_position_x, chunk_position_y) - 1;
		}
		if(j == 0){
			chunk_position_y = this->global_position[1] - 1;
			local_point_coordinate_y = this->parent_parallel_mesh->get_n_points_y_mesh_chunk(chunk_position_x, chunk_position_y) - 1;
		}

		return this->parent_parallel_mesh->get_velocity_mesh_chunk_x(chunk_position_x, chunk_position_y, local_point_coordinate_x, local_point_coordinate_y);
	}
	std::cout << "Error in x velocity get" << '\n';
	return std::nan("");
}

double MeshChunk::
get_velocity_y(int64_t i, int64_t j) {
	//std::cout << "////// getting velocity y " << "(i, j) : (" << i << ", " << j << ")" << '\n';
	if(i < this->get_n_points_x() && j < this->get_n_points_y() && i > 0 && j > 0){
		if((this->get_point_type(i, j) == PointTypeEnum::INTERIOR) || (this->get_point_type(i, j) == PointTypeEnum::SHARED_OWNED))
    	return this->velocity(i, j, 0);
		else if(this->get_point_type(i, j) == PointTypeEnum::BOUNDARY){
			if(this->get_point_type(i, j) == PointTypeEnum::BOUNDARY){
				std::string boundary;
				if(j == this->get_n_points_y() - 1){
					boundary = "v_y_t";
				}
				if(i == this->get_n_points_x() - 1){
					boundary = "v_y_r";
				}
				if(i == 0){
					boundary = "v_y_l";
				}
				if(j == 0){
					boundary = "v_y_b";
				}
				return this->parent_parallel_mesh->get_boundary_velocity_value(boundary);
			}
		}
	}
	else{
		//std::cout << "//// Not owned" << '\n';
		if(this->get_point_type(i, j) == PointTypeEnum::BOUNDARY){
			std::string boundary;
			if(j == this->get_n_points_y() - 1){
				boundary = "v_y_t";
			}
			if(i == this->get_n_points_x() - 1){
				boundary = "v_y_r";
			}
			if(i == 0){
				boundary = "v_y_l";
			}
			if(j == 0){
				boundary = "v_y_b";
			}
			return this->parent_parallel_mesh->get_boundary_velocity_value(boundary);
		}
		int64_t chunk_position_x = this->global_position[0];
		int64_t chunk_position_y = this->global_position[1];
		uint64_t local_point_coordinate_x = i;
		uint64_t local_point_coordinate_y = j;
		//std::cout << "//// Chunk position : (" << chunk_position_x << ", " << chunk_position_y << ")" << '\n';
		//std::cout << "//// Local point coordinates : (" << local_point_coordinate_x << ", " << local_point_coordinate_y << ")" << '\n';
		if(i == this->get_n_points_x() + 1){
			chunk_position_x = this->global_position[0] + 1;
			local_point_coordinate_x = 1;
		}
		if(j == this->get_n_points_y() + 1){
			chunk_position_y = this->global_position[1] + 1;
			local_point_coordinate_y = 1;
		}
		if(i == 0){
			chunk_position_x = this->global_position[0] - 1;
			local_point_coordinate_x = this->parent_parallel_mesh->get_n_points_x_mesh_chunk(chunk_position_x, chunk_position_y) - 1;
		}
		if(j == 0){
			chunk_position_y = this->global_position[1] - 1;
			local_point_coordinate_y = this->parent_parallel_mesh->get_n_points_y_mesh_chunk(chunk_position_x, chunk_position_y) - 1;
		}

		return this->parent_parallel_mesh->get_velocity_mesh_chunk_y(chunk_position_x, chunk_position_y, local_point_coordinate_x, local_point_coordinate_y);
	}
	std::cout << "Error in y velocity get" << '\n';
	return std::nan("");
}

void MeshChunk::
set_pressure(uint64_t i, uint64_t j, double scalar){
  uint64_t k = this->Cartesian_to_index(i, j, this->n_cells_x, this->n_cells_y);
  if(k != -1)
    this->pressure(k) = scalar;
}

double MeshChunk::
get_pressure(uint64_t i, uint64_t j) const{
  // Return invalid mesh chunk when indices are out of bounds
  uint64_t k = this->Cartesian_to_index(i, j, this->n_cells_x, this->n_cells_y);
  if(k == -1)
    return std::nan("");
  else
    return this->pressure(k);
}

void MeshChunk::apply_velocity_bc(std::map<std::string, double> velocity_values){
  // left side
  if(this->get_global_position()[0] == 0){
    for(uint64_t j = 0; j < this->get_n_points_y() + 1; j++){
      this->set_velocity_x(0, j, velocity_values["v_x_l"]);
      this->set_velocity_y(0, j, velocity_values["v_y_l"]);
    }
  }
  // right side
  else if(this->get_global_position()[0] == this->n_chunks_global_x - 1){
    for(uint64_t j = 0; j < this->get_n_points_y() + 1; j++){
      this->set_velocity_x(this->get_n_points_x() - 1, j, velocity_values["v_x_r"]);
      this->set_velocity_y(this->get_n_points_x() - 1, j, velocity_values["v_y_r"]);
    }
  }
  // bottom side
  if(this->get_global_position()[1] == 0){
    for(uint64_t i = 0; i < this->get_n_points_x() + 1; i++){
      this->set_velocity_x(i, 0, velocity_values["v_x_b"]);
      this->set_velocity_y(i, 0, velocity_values["v_y_b"]);
    }
  }
  // top side
  else if(this->get_global_position()[1] == this->n_chunks_global_y - 1){
    for(uint64_t i = 0; i < this->get_n_points_x() + 1; i++){
      this->set_velocity_x(i, this->get_n_points_y() - 1, velocity_values["v_x_t"]);
      this->set_velocity_y(i, this->get_n_points_y() - 1, velocity_values["v_y_t"]);
    }
  }
}

void MeshChunk::chunk_predict_velocity(double delta_t, double nu){
	Kokkos::View<double*[2]> v_star("predicted velocity", this->get_n_points_x() * this->get_n_points_y());
  const uint64_t m = this->get_n_points_x();
  const uint64_t mm1 = m - 1;
  const uint64_t n = this->get_n_points_y();
  const uint64_t nm1 = n - 1;

  // compute common factors
  const double h = this->get_cell_size();
  const double inv_2sz = 1. / (2. * h);
  const double inv_sz2 = 1. / (h * h);

  // predict velocity components using finite difference discretization
  for(uint64_t j = 0; j < n; j++){
    for(uint64_t i = 0; i < m; i++){
			if((this->get_point_type(i, j) == PointTypeEnum::INTERIOR) || (this->get_point_type(i, j) == PointTypeEnum::SHARED_OWNED)) {
				// retrieve velocity at stencil nodes only once
	      double v_x_ij = this->get_velocity_x(i, j);
	      double v_x_ij_l = this->get_velocity_x(i - 1, j);
	      double v_x_ij_r = this->get_velocity_x(i + 1, j);
	      double v_x_ij_t = this->get_velocity_x(i, j + 1);
	      double v_x_ij_b = this->get_velocity_x(i, j - 1);
	      double v_y_ij = this->get_velocity_y(i, j);
	      double v_y_ij_l = this->get_velocity_y(i - 1, j);
	      double v_y_ij_r = this->get_velocity_y(i + 1, j);
	      double v_y_ij_t = this->get_velocity_y(i, j + 1);
	      double v_y_ij_b = this->get_velocity_y(i, j - 1);

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
	      uint64_t k = this->Cartesian_to_index(i, j, m, n);
	      v_star(k, 0) = v_x_ij + delta_t * (nu * (dudx2 + dudy2) - (v_x_ij * dudx + v_y * dudy));
	      v_star(k, 1) = v_y_ij + delta_t * (nu * (dvdx2 + dvdy2) - (v_y_ij * dvdx + v_x * dvdy));
			}
    }
	}
	// assign interior predicted velocity vectors to mesh
  for(int j = 0; j < n; j++){
    for(int i = 0; i < m; i++){
			if((this->get_point_type(i, j) == PointTypeEnum::INTERIOR) || (this->get_point_type(i, j) == PointTypeEnum::SHARED_OWNED)) {
	      int k = this->Cartesian_to_index(i, j, m, n);
	      this->set_velocity_x(i, j, v_star(k, 0));
	      this->set_velocity_y(i, j, v_star(k, 1));
			}
    }
  }
}

#ifdef USE_MPI
void MeshChunk::
mpi_send_border_velocity_x(double border_velocity_x, uint64_t pos_x, uint64_t pos_y, uint64_t dest_rank, uint8_t border) {
	MPI_Send(&border_velocity_x, 1, MPI_DOUBLE, dest_rank, border, MPI_COMM_WORLD);
}

void MeshChunk::
mpi_send_border_velocity_y(double border_velocity_y, uint64_t pos_x, uint64_t pos_y, uint64_t dest_rank, uint8_t border) {
	MPI_Send(&border_velocity_y, 1, MPI_DOUBLE, dest_rank, border, MPI_COMM_WORLD);
}

void MeshChunk::
mpi_receive_border_velocity_x(double border_velocity_x, uint64_t pos_x, uint64_t pos_y, uint64_t dest_rank, uint8_t border, MPI_Status status) {
	MPI_Recv(&border_velocity_x, 1, MPI_DOUBLE, dest_rank, border, MPI_COMM_WORLD, &status);
}

void MeshChunk::
mpi_receive_border_velocity_y(double border_velocity_y, uint64_t pos_x, uint64_t pos_y, uint64_t dest_rank, uint8_t border, MPI_Status status) {
	MPI_Recv(&border_velocity_y, 1, MPI_DOUBLE, dest_rank, border, MPI_COMM_WORLD, &status);
}
#endif

#ifdef OUTPUT_VTK_FILES
vtkSmartPointer<vtkUniformGrid> MeshChunk::
make_VTK_uniform_grid() const{
  // instantiate VTK uniform grid from mesh chunk parameters
  vtkSmartPointer<vtkUniformGrid> ug = vtkSmartPointer<vtkUniformGrid>::New();
  uint64_t n_c_x = this->n_cells_x;
  uint64_t n_c_y = this->n_cells_y;
  uint64_t n_p_x = n_c_x + 1;
  uint64_t n_p_y = n_c_y + 1;
  ug->SetDimensions(n_p_x, n_p_y, 1);
  ug->SetOrigin(this->origin[0], this->origin[1], 0);
  ug->SetSpacing(this->h, this->h, 0);

  // create point centered type and velocity fields
  vtkSmartPointer<vtkIntArray>
    point_type = vtkSmartPointer<vtkIntArray>::New();
  vtkSmartPointer<vtkDoubleArray>
    point_data = vtkSmartPointer<vtkDoubleArray>::New();
  point_type->SetNumberOfComponents(1);
  point_type->SetName("Type");
  point_type->SetNumberOfTuples(n_p_x * n_p_y);
  point_data->SetNumberOfComponents(3);
  point_data->SetName("Velocity");
  point_data->SetNumberOfTuples(n_p_x * n_p_y);
  for(uint64_t j = 0; j < n_p_y; j++){
    for(uint64_t i = 0; i < n_p_x; i++){
      point_type->SetTuple1(j * n_p_x + i, static_cast<int>(this->point_type(i, j)));
      point_data->SetTuple3(j * n_p_x + i, this->velocity(i, j, 0), this->velocity(i, j, 1), 0);
    }
  }
  ug->GetPointData()->SetScalars(point_type);
  ug->GetPointData()->SetVectors(point_data);

  // create cell centered pressure field
  vtkSmartPointer<vtkDoubleArray>
    cell_data = vtkSmartPointer<vtkDoubleArray>::New();
  cell_data->SetNumberOfComponents(1);
  cell_data->SetName("Pressure");
  cell_data->SetNumberOfValues(n_c_x * n_c_y);
  for(int j = 0; j < n_c_y; j++){
    for(int i = 0; i < n_c_x; i++){
      cell_data->SetValue(j * n_c_x + i, this->get_pressure(i,j));
    }
  }
  ug->GetCellData()->SetScalars(cell_data);

  // return VTK smart pointer to uniform grid
  return ug;
}

std::string MeshChunk::
write_vti(const std::string& file_name) const{
  // assemble full file name with extension
  std::string full_file_name = file_name + ".vti";

  // write VTK image data (vti) file
  vtkSmartPointer<vtkXMLImageDataWriter>
    output_file = vtkSmartPointer<vtkXMLImageDataWriter>::New();
  output_file->SetFileName(full_file_name.c_str());
  output_file->SetInputData(this->make_VTK_uniform_grid());
  output_file->Write();

  // return fill name with extension
  return full_file_name;
}
#endif
