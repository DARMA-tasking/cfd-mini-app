#include <iostream>
#include <array>
#include <cmath>
#include <Kokkos_Core.hpp>

#include "mesh.h"

#include <vtkDoubleArray.h>
#include <vtkCellData.h>
#include <vtkPointData.h>
#include <vtkUniformGrid.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLImageDataReader.h>

////////////////////////////////////////////////////////////////
// BASIC
////////////////////////////////////////////////////////////////

void Mesh::set_origin(double x, double y){
  this->O[0] = x;
  this->O[1] = y;
}

std::array<int,2> Mesh::index_to_cartesian(int k, int n, int nmax){
  if (k < 0 || k >= nmax){
  // Return invalid values when index is out of bounds
    return  {-1, -1};
  } else{
    std::div_t dv = std::div(k, n);
    return {dv.rem, dv.quot};
  }
}

int Mesh::cartesian_to_index(int i, int j, int ni, int nj){
  if(i<0 || i>=ni || j<0 || j>=nj){
    // Return invalid value when coordinates are out of bounds
    return -1;
  } else{
    return j * ni + i;
  }
}

////////////////////////////////////////////////////////////////
// CELLS (PRESSURE)
////////////////////////////////////////////////////////////////

void Mesh::set_pressure(int i, int j, double scalar){
  int k = this->cartesian_to_index(i, j, this->n_cells_x, this->n_cells_y);
  if(k != -1){
    this->pressure(k) = scalar;
  }
}

double Mesh::get_pressure(int i, int j){
  int k = this->cartesian_to_index(i, j, this->n_cells_x, this->n_cells_y);
  if(k == -1){
    return std::nan("");
  } else{
    return this->pressure(k);
  }
}

////////////////////////////////////////////////////////////////
// POINTS (VELOCITY)
////////////////////////////////////////////////////////////////

void Mesh::set_velocity_x(int i, int j, double u){
  int k = this->cartesian_to_index(i, j, this->get_n_points_x(), this->get_n_points_y());
  if(k != -1){
    this->velocity(k, 0) = u;
  }
}

void Mesh::set_velocity_y(int i, int j, double v){
  int k = this->cartesian_to_index(i, j, this->get_n_points_x(), this->get_n_points_y());
  if(k != -1){
    this->velocity(k, 1) = v;
  }
}

double Mesh::get_velocity_x(int i, int j){
  int k = this->cartesian_to_index(i, j, this->get_n_points_x(), this->get_n_points_y());
  if(k != -1){
    return this->velocity(k, 0);
  } else{
    return std::nan("");
  }
}

double Mesh::get_velocity_y(int i, int j){
  int k = this->cartesian_to_index(i, j, this->get_n_points_x(), this->get_n_points_y());
  if(k != -1){
    return this->velocity(k, 1);
  } else{
    return std::nan("");
  }
}

void Mesh::write_vtk(std::string file_name){
  vtkNew<vtkUniformGrid> ug;
  uint64_t nx = this->n_cells_x;
  uint64_t ny = this->n_cells_y;
  ug->SetDimensions(nx+1, ny+1, 1);
  ug->SetOrigin(this->O[0], this->O[0], 0);
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
