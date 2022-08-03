#include <iostream>
#include <array>
#include <map>
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "mesh_chunk.cpp"
#include "boundary_conditions.cpp"
#include "solver.cpp"

struct solver_test : testing::Test{
  std::unique_ptr<Solver> solver;

  solver_test()
  {
    // define input parameters
    double density = 1.225;
    double dynamic_viscosity = 0.3;
    double delta_t = 0.001;
    double t_final = 0.1;
    double max_C = 0.5;
    uint64_t n_cells = 3;
    uint64_t n_parallel_meshes = 1;
    uint64_t n_colors_x = 1;
    uint64_t n_colors_y = 1;

    // create mesh
    std::map<PointIndexEnum, PointTypeEnum> point_types = {
      {PointIndexEnum::CORNER_0, PointTypeEnum::BOUNDARY},
      {PointIndexEnum::CORNER_1, PointTypeEnum::BOUNDARY},
      {PointIndexEnum::CORNER_2, PointTypeEnum::BOUNDARY},
      {PointIndexEnum::CORNER_3, PointTypeEnum::BOUNDARY},
      {PointIndexEnum::EDGE_0, PointTypeEnum::BOUNDARY},
      {PointIndexEnum::EDGE_1, PointTypeEnum::BOUNDARY},
      {PointIndexEnum::EDGE_2, PointTypeEnum::BOUNDARY},
      {PointIndexEnum::EDGE_3, PointTypeEnum::BOUNDARY}
    };
    auto mesh = std::make_shared<MeshChunk>
      (n_cells, n_cells, 1. / n_cells, point_types);

    // define boundary conditions
    std::map<std::string, double> velocity_values = {
      {"u_top", 1.0},
      {"v_top", 0.0},
      {"u_bot", 0.0},
      {"v_bot", 0.0},
      {"u_left", 0.0},
      {"v_left", 0.0},
      {"u_right", 0.0},
      {"v_right", 0.0}
    };
    BoundaryConditions b_c(mesh, velocity_values);

    // instantiate solver
    this->solver = std::make_unique<Solver>
      (mesh, b_c, delta_t, t_final, density, dynamic_viscosity, max_C, 0, 25, 25, 1, n_parallel_meshes, n_colors_x, n_colors_y);
  }
};

TEST_F(solver_test, Laplacian_values_test){
  // run solve routine until Laplacian matrix is generated
  this->solver->solve(Solver::stopping_point::AFTER_LAPLACIAN, Solver::linear_solver::NONE, Solver::adaptative_time_step::OFF);

  // remap CRS Laplacian into dense matrix for convenience
  int mn = 9;
  auto sparse_Laplacian = solver->get_Laplacian();
  double dense_Laplacian[mn][mn];
  auto n_cols = sparse_Laplacian.numCols();
  for(int j = 0; j < mn; j++){
    auto row = sparse_Laplacian.row(j);
    int i = 0;
    for(int k = 0; k < row.length; k++){
      auto val = row.value(k);
      auto col = row.colidx(k);
      while (i < col)
	dense_Laplacian[i++][j] = 0.;
      dense_Laplacian[i++][j] = val;
    }
    while (i < n_cols)
      dense_Laplacian[i++][j] = 0.;
  }

  // check if all values in Laplacian matrix are correct
  for(int j = 0; j < mn; j++){
    for(int i = 0; i < mn; i++){
      if(j == i){
	// check diagonal coefficient
	if(i < 2){
	  EXPECT_EQ(dense_Laplacian[2 * i][2 * i], -2);
	  EXPECT_EQ(dense_Laplacian[mn - 1 - 2 * i][mn - 1 - 2 * i], -2);
	}
	if(i < 4){
	  EXPECT_EQ(dense_Laplacian[2 * i + 1][2 * i + 1], -3);
	}
      } else if((j == i + 3 && i < 6) || (j == i - 3 && i > 2) || (j == i - 1 && i > 0 && i != 3 && i != 6) || (j == i + 1 && i < mn - 1 && i != 2 && i != 5)){
	// check non zero off-diagonal coefficients
	EXPECT_EQ(dense_Laplacian[i][j], 1);
      } else {
	// check zero off-diagonal coefficients
	EXPECT_EQ(dense_Laplacian[i][j], 0);
      }
    }
  }
  EXPECT_EQ(dense_Laplacian[4][4], -4);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  // handle Kokkos boilerplate
  Kokkos::ScopeGuard kokkos(argc, argv);

  return RUN_ALL_TESTS();
}
