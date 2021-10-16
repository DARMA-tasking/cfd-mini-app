#include "mesh_chunk.h"

#include <iostream>
#include <array>
#include <Kokkos_Core.hpp>

#include <vtkDoubleArray.h>
#include <vtkIntArray.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkUniformGrid.h>
#include <vtkSmartPointer.h>
#include <vtkXMLImageDataWriter.h>

MeshChunk::
MeshChunk(uint64_t n_x, uint64_t n_y, double cell_size,
		     const std::map<PointIndexEnum, PointTypeEnum>& point_types,
		     double o_x, double o_y)
  : n_cells_x(n_x)
  , n_cells_y(n_y)
  , h(cell_size)
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
  if (k < 0 || k >= nmax)
    return  {(uint64_t) -1, (uint64_t) -1};
  else{
    std::ldiv_t dv = std::ldiv(k, n);
    return {(uint64_t) dv.rem, (uint64_t) dv.quot};
  }
}

uint64_t MeshChunk::
Cartesian_to_index(uint64_t i, uint64_t j, uint64_t ni, uint64_t nj) const{
  // Return invalid value when coordinates are out of bounds
  if(i<0 || i>=ni || j<0 || j>=nj)
    return -1;
  else
    return j * ni + i;
}

void MeshChunk::
set_point_type(uint64_t i, uint64_t j, PointTypeEnum t){
  if(i > -1 && i < this->get_n_points_x() && j > -1 && j < this->get_n_points_y())
    this->point_type(i, j) = t;
}

PointTypeEnum MeshChunk::
get_point_type(uint64_t i, uint64_t j) const{
  // Return invalid type when indices are out of bounds
  if(i > -1 && i < this->get_n_points_x() && j > -1 && j < this->get_n_points_y())
    return this->point_type(i, j);
  else
    return PointTypeEnum::INVALID;
}

void MeshChunk::
set_velocity_x(uint64_t i, uint64_t j, double u){
  if(i > -1 && i < this->get_n_points_x() && j > -1 && j < this->get_n_points_y())
    this->velocity(i, j, 0) = u;
}

void MeshChunk::
set_velocity_y(uint64_t i, uint64_t j, double v){
  if(i > -1 && i < this->get_n_points_x() && j > -1 && j < this->get_n_points_y())
    this->velocity(i, j, 1) = v;
}

double MeshChunk::
get_velocity_x(uint64_t i, uint64_t j) const{
  // Return invalid velocity when indices are out of bounds
  if(i > -1 && i < this->get_n_points_x() && j > -1 && j < this->get_n_points_y())
    return this->velocity(i, j, 0);
  else
    return std::nan("");
}

double MeshChunk::
get_velocity_y(uint64_t i, uint64_t j) const{
  // Return invalid velocity when indices are out of bounds
  if(i > -1 && i < this->get_n_points_x() && j > -1 && j < this->get_n_points_y())
    return this->velocity(i, j, 1);
  else
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
  for(uint64_t  j = 0; j < n_p_y; j++){
    for(uint64_t  i = 0; i < n_p_x; i++){
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
