#include <iostream>
#include <array>
#include <map>
#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include "mesh.cpp"
#include "boundaryconditions.cpp"
#include "solver.cpp"

struct solver_test : testing::Test{
  Solver* solver;

  solver_test()
  {
    // define input parameters
    double density = 1.225;
    double dynamic_viscosity = 0.3;
    double delta_t = 0.001;
    double t_final = 0.1;
    double max_C = 0.5;
    uint64_t n_cells = 3;

    // create mesh
    Mesh mesh(n_cells, n_cells, 1. / n_cells);

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

    solver = new Solver(mesh, b_c, delta_t, t_final, density, dynamic_viscosity, max_C, 0);
  }
};

TEST_F(solver_test, laplacian_values_test){
  // run solve routine until laplacian matrix is generated
  solver->solve(Solver::stopping_point::AFTER_LAPLACIAN, Solver::linear_solver::CONJUGATE_GRADIENT, Solver::adaptative_time_step::ON);

  // check if all values in laplacian matrix are correct
  for(int j = 0; j < 9; j++){
    for(int i = 0; i < 9; i++){
      if(j == i){
        // check diagonal coefficient
        if(i == 0 || i == 1){
          EXPECT_EQ(solver->get_laplacian()(2 * i, 2 * i), -2);
          EXPECT_EQ(solver->get_laplacian()(8 - 2 * i, 8 - 2 * i), -2);
        }
        if(i < 4){
          EXPECT_EQ(solver->get_laplacian()(2 * i + 1, 2 * i + 1), -3);
        }
      } else if((j == i + 3 && i < 6) || (j == i - 3 && i > 2) || (j == i - 1 && i > 0 && i != 3 && i != 6) || (j == i + 1 && i < 8 && i !=2 && i !=5)){
        // check non zero off-diagonal coefficients
        EXPECT_EQ(solver->get_laplacian()(i, j), 1);
      } else {
        // check zero off-diagonal coefficients
        EXPECT_EQ(solver->get_laplacian()(i, j), 0);
      }
    }
  }
  EXPECT_EQ(solver->get_laplacian()(4, 4), -4);
}


int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  // handle Kokkos boilerplate
  Kokkos::ScopeGuard kokkos(argc, argv);

  return RUN_ALL_TESTS();
}
