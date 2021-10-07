#include "mesh_chunk.h"

#include <iostream>
#include <array>
#include <cmath>
#include <Kokkos_Core.hpp>

#include <vtkDoubleArray.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkUniformGrid.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLImageDataReader.h>


MeshChunk::MeshChunk(int n_x, int n_y, double cell_size,
		     std::map<std::string, PointTypeEnum> point_types){

  // set instance variables
  this->n_cells_x = n_x;
  this->n_cells_y = n_y;
  this->origin = {0, 0};
  this->h = cell_size;

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
    if (kv.first == "i")
      for(uint64_t j = 1; j < n_y; j++)
	for(uint64_t i = 1; i < n_x; i++)
	  this->point_type(i, j) = kv.second;

    // non-corner edge points
    else if (kv.first == "b")
      for(uint64_t i = 1; i < n_x; i++)
	this->point_type(i, 0) = kv.second;
    else if (kv.first == "t")
      for(uint64_t i = 1; i < n_x; i++)
	this->point_type(i, n_y) = kv.second;
    else if (kv.first == "l")
      for(uint64_t j = 1; j < n_y; j++)
	this->point_type(0, j) = kv.second;
    else if (kv.first == "r")
      for(uint64_t j = 1; j < n_y; j++)
	this->point_type(n_x, j) = kv.second;

    // corner points
    else if (kv.first == "bl")
      this->point_type(0, 0) = kv.second;
    else if (kv.first == "br")
      this->point_type(n_x, 0) = kv.second;
    else if (kv.first == "tl")
      this->point_type(0, n_y) = kv.second;
    else if (kv.first == "tr")
      this->point_type(n_x, n_y) = kv.second;
  }
}

void MeshChunk::set_origin(double x, double y){
  this->origin[0] = x;
  this->origin[1] = y;
}

std::array<int,2> MeshChunk::index_to_cartesian(int k, int n, int nmax){
  if (k < 0 || k >= nmax){
  // Return invalid values when index is out of bounds
    return  {-1, -1};
  } else{
    std::div_t dv = std::div(k, n);
    return {dv.rem, dv.quot};
  }
}

int MeshChunk::cartesian_to_index(int i, int j, int ni, int nj){
  if(i<0 || i>=ni || j<0 || j>=nj){
    // Return invalid value when coordinates are out of bounds
    return -1;
  } else{
    return j * ni + i;
  }
}

void MeshChunk::set_point_type(int i, int j, PointTypeEnum t){
  if(i > -1 && i < this->get_n_points_x() && j > -1 && j < this->get_n_points_y()){
    this->point_type(i, j) = t;
  }
}

PointTypeEnum MeshChunk::get_point_type(int i, int j){
  if(i > -1 && i < this->get_n_points_x() && j > -1 && j < this->get_n_points_y()){
    return this->point_type(i, j);
  } else{
    return PointTypeEnum::INVALID;
  }
}

void MeshChunk::set_velocity_x(int i, int j, double u){
  if(i > -1 && i < this->get_n_points_x() && j > -1 && j < this->get_n_points_y()){
    this->velocity(i, j, 0) = u;
  }
}

void MeshChunk::set_velocity_y(int i, int j, double v){
  if(i > -1 && i < this->get_n_points_x() && j > -1 && j < this->get_n_points_y()){
    this->velocity(i, j, 1) = v;
  }
}

double MeshChunk::get_velocity_x(int i, int j){
  if(i > -1 && i < this->get_n_points_x() && j > -1 && j < this->get_n_points_y()){
    return this->velocity(i, j, 0);
  } else{
    return std::nan("");
  }
}

double MeshChunk::get_velocity_y(int i, int j){
  if(i > -1 && i < this->get_n_points_x() && j > -1 && j < this->get_n_points_y()){
    return this->velocity(i, j, 1);
  } else{
    return std::nan("");
  }
}

void MeshChunk::set_pressure(int i, int j, double scalar){
  int k = this->cartesian_to_index(i, j, this->n_cells_x, this->n_cells_y);
  if(k != -1){
    this->pressure(k) = scalar;
  }
}

double MeshChunk::get_pressure(int i, int j){
  int k = this->cartesian_to_index(i, j, this->n_cells_x, this->n_cells_y);
  if(k == -1){
    return std::nan("");
  } else{
    return this->pressure(k);
  }
}

void MeshChunk::write_vtk(std::string file_name){
  vtkNew<vtkUniformGrid> ug;
  uint64_t nx = this->n_cells_x;
  uint64_t ny = this->n_cells_y;
  ug->SetDimensions(nx + 1, ny + 1, 1);
  ug->SetOrigin(this->origin[0], this->origin[0], 0);
  ug->SetSpacing(this->h, this->h, 0);

  // create cell centered scalar field
  vtkNew<vtkDoubleArray> cell_data;
  cell_data->SetNumberOfComponents(1);
  cell_data->SetName("Pressure");
  cell_data->SetNumberOfValues(nx * ny);
  for(int j = 0; j < ny; j++){
    for(int i = 0; i < nx; i++){
      cell_data->SetValue(j*nx + i, this->get_pressure(i,j));
    }
  }
  ug->GetCellData()->SetScalars(cell_data);

  // create point centered vector field
  vtkNew<vtkDoubleArray> point_data;
  point_data->SetNumberOfComponents(3);
  point_data->SetName("Velocity");
  point_data->SetNumberOfTuples((nx + 1) * (ny + 1));
  for(int j = 0; j < ny +1; j++){
    for(int i = 0; i < nx + 1; i++){
      point_data->SetTuple3(j * (nx + 1) + i, this->get_velocity_x(i, j), this->get_velocity_y(i, j), 0);
    }
  }
  ug->GetPointData()->SetVectors(point_data);

  // write vti visualization file
  vtkNew<vtkXMLImageDataWriter> output_file;
  output_file->SetFileName(file_name.c_str());
  output_file->SetInputData(ug);
  output_file->Write();

  std::cout<<std::endl;
  std::cout<<"Visualization file created: \""<<file_name<<"\""<<std::endl;
}
