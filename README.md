This is the repository for the Kokkos-based CFD mini app.

# Capabilities

* This CFD mini app is a solver for structured 2-dimensional grid meshes for incompressible fluid flow, using:
  * a finite difference method for the spatial discretization (with 5-point stencil for the Laplacian);
  * a conjugate gradient linear solve for the Poisson pressure equation;
  * explicit Euler time integration with adaptive time-stepping.
* The boundary conditions are as follows:
  * pressure (cell-centered): zero-flow von Neumann condition;
  * velocity (vertex-centered): Dirichlet conditions, with the boundary velocity values given to the solver through a map of values.
