#include"solver.h"

Solver::Solver(Mesh m, double d_t, double t_f, double r, double d_v, double m_C, int v)
  : mesh(m)
  , delta_t(d_t)
  , t_final(t_f)
  , rho(r)
  , max_C(m_C)
  , verbosity(v)
  {
    // compute and store kinematic viscosity
    this->nu = d_v / r;


  }


void Solver::assemble_laplacian()
  {
    // initialize container for Laplacian
    const int mn = this->mesh.get_n_cells_x() * this->mesh.get_n_cells_y();
    this->laplacian = Kokkos::View<double**>("laplacian", mn, mn);

    for(int j = 1; j <= mn - 1; j++)
    {
      for(int i = 1; i <= mn - 1; i++)
      {
        int k = this->mesh.cartesian_to_index(i, j, mn, mn);
        // initialize diagonal value
        int v = -4;

        //detect corners and borders
        if(i == 0 || i == mn - 1 || j == 0 || j == mn - 1)
        {
          ++v;
        }

        // assign diagonal entry
        this->laplacian(k, k) = v;

        // assign unit entries
        if(i == 1)
        {
          this->laplacian(k, k-1) = 1;
        }
        if(i < mn - 1)
        {
          this->laplacian(k, k + 1) = 1;
        }
        if(j == 1)
        {
          this->laplacian(k, k - mn) = 1;
        }
        if(j < mn - 1)
        {
          this->laplacian(k, k + mn) = 1;
        }
      }
    }
  }
