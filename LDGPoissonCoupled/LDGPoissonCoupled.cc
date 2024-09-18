// https://www.dealii.org/developer/doxygen/deal.II/step_60.html#step_60-Runningwithspacedimequaltothree
// kozlow point
//  @sect3{LDGPoisson.cc}
//  The code begins as per usual with a long list of the the included
//  files from the deal.ii library.

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

#include <fstream>
#include <iostream>

// Here's where the classes for the DG methods begin.
// We can use either the Lagrange polynomials,
#include <deal.II/fe/fe_dgq.h>
// or the Legendre polynomials
#include <deal.II/fe/fe_dgp.h>
// as basis functions.  I'll be using the Lagrange polynomials.
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/lac/sparsity_tools.h>

#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
//#include <deal.II/non_matching/fe_values.h>

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
//#include <deal.II/fe/fe_coupling_values.h>

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "Functions.cc"

using namespace dealii;
#define USE_MPI 1
#define USE_LDG 0

constexpr unsigned int dimension_Omega{3};
const FEValuesExtractors::Vector VectorField_omega(0);
const FEValuesExtractors::Scalar Potential_omega(1);

const FEValuesExtractors::Vector VectorField(0);
const FEValuesExtractors::Scalar Potential(dimension_Omega + 1);

const unsigned int dimension_gap = 0;
const double extent = 1;
const double half_length = 0.5;
const double distance_tolerance = 10;
const unsigned int N_quad_points = 7;

struct Parameters {
  double radius;
  bool lumpedAverage;
};
template <int dim, int dim_omega> class LDGPoissonProblem {

public:
  LDGPoissonProblem(const unsigned int degree, const unsigned int n_refine,
                    Parameters parameters);

  ~LDGPoissonProblem();

  std::array<double, 4> run();

private:
  void make_grid();

  void make_dofs();

  void assemble_system();

  template <int _dim>
  void assemble_cell_terms(const FEValues<_dim> &cell_fe,
                           FullMatrix<double> &cell_matrix,
                           Vector<double> &cell_vector,
                           const TensorFunction<2, _dim> &K_inverse_function,
                           const Function<_dim> &_rhs_function,
                           const FEValuesExtractors::Vector &VectorField,
                           const FEValuesExtractors::Scalar &Potential);
  template <int _dim>
  void assemble_Neumann_boundary_terms(
      const FEFaceValues<_dim> &face_fe, FullMatrix<double> &local_matrix,
      Vector<double> &local_vector, const Function<_dim> &Neumann_bc_function);

  template <int _dim>
  void assemble_Dirichlet_boundary_terms(
      const FEFaceValues<_dim> &face_fe, FullMatrix<double> &local_matrix,
      Vector<double> &local_vector, const double &h,
      const Function<_dim> &Dirichlet_bc_function,
      const FEValuesExtractors::Vector VectorField,
      const FEValuesExtractors::Scalar Potential);
  template <int _dim>
  void assemble_flux_terms(
      const FEFaceValuesBase<_dim> &fe_face_values,
      const FEFaceValuesBase<_dim> &fe_neighbor_face_values,
      FullMatrix<double> &vi_ui_matrix, FullMatrix<double> &vi_ue_matrix,
      FullMatrix<double> &ve_ui_matrix, FullMatrix<double> &ve_ue_matrix,
      const double &h, const FEValuesExtractors::Vector VectorField,
      const FEValuesExtractors::Scalar Potential);

  void distribute_local_flux_to_global(
      FullMatrix<double> &vi_ui_matrix, FullMatrix<double> &vi_ue_matrix,
      FullMatrix<double> &ve_ui_matrix, FullMatrix<double> &ve_ue_matrix,
      const std::vector<types::global_dof_index> &local_dof_indices,
      const std::vector<types::global_dof_index> &local_neighbor_dof_indices);

  template <int _dim>
  void dof_omega_to_Omega(
      const DoFHandler<_dim> &dof_handler,
      std::vector<types::global_dof_index> &local_dof_indices_omega);

  void solve();

  std::array<double, 4> compute_errors() const;
  void output_results() const;

  const unsigned int degree;
  const unsigned int n_refine;
  double penalty;
  double h_max;
  double h_min;

  enum { Dirichlet, Neumann };

  // parameters
  double radius;
  double g;
  bool lumpedAverage;

#if USE_MPI
  parallel::distributed::Triangulation<dim> triangulation;

  TrilinosWrappers::SparseMatrix system_matrix;
  TrilinosWrappers::MPI::Vector solution;
  TrilinosWrappers::MPI::Vector system_rhs;

#else
  Triangulation<dim> triangulation;

  SparseMatrix<double> system_matrix;
  Vector<double> solution;
  Vector<double> system_rhs;

#endif
  FESystem<dim> fe;
  DoFHandler<dim> dof_handler;

  Triangulation<dim_omega> triangulation_omega;
  FESystem<dim_omega> fe_omega;
  DoFHandler<dim_omega> dof_handler_omega;
  Vector<double> solution_omega;

  AffineConstraints<double> constraints;

  SparsityPattern sparsity_pattern;

  ConditionalOStream pcout;
  TimerOutput computing_timer;

  SolverControl solver_control;
  TrilinosWrappers::SolverDirect solver;

  const RightHandSide<dim> rhs_function;
  const KInverse<dim> K_inverse_function;
  const DirichletBoundaryValues<dim> Dirichlet_bc_function;
  const NeumannBoundaryValues<dim> Neumann_bc_function;
  const TrueSolution<dim> true_solution;
  const TrueSolution_omega<dim_omega> true_solution_omega;

  const RightHandSide_omega<dim_omega> rhs_function_omega;
  const KInverse<dim_omega> k_inverse_function;
  const DirichletBoundaryValues_omega<dim_omega> Dirichlet_bc_function_omega;

  std::vector<Point<dim>> support_points;
  std::vector<Point<dim_omega>> support_points_omega;
  std::vector<Point<dim_omega>> unit_support_points_omega;
  unsigned int start_VectorField_omega;
  unsigned int start_Potential_omega;
  unsigned int start_Potential;

  const UpdateFlags update_flags = update_values | update_gradients |
                                   update_quadrature_points | update_JxW_values;
  const UpdateFlags update_flags_coupling = update_values | update_JxW_values;

  const UpdateFlags face_update_flags = update_values | update_normal_vectors |
                                        update_quadrature_points |
                                        update_JxW_values;
};

template <int dim, int dim_omega>
LDGPoissonProblem<dim, dim_omega>::LDGPoissonProblem(
    const unsigned int degree, const unsigned int n_refine,
    Parameters parameters)
    : degree(degree), n_refine(n_refine),
#if USE_MPI
      triangulation(MPI_COMM_WORLD),
#endif
      fe(FESystem<dim>(FE_DGP<dim>(degree), dim), FE_DGP<dim>(degree),
         FE_DGP<dim>(degree), FE_DGP<dim>(degree)),
      dof_handler(triangulation),
      fe_omega(FESystem<dim_omega>(FE_DGP<dim_omega>(degree), dim_omega),
               FE_DGP<dim_omega>(degree)),
      dof_handler_omega(triangulation_omega),
#if USE_MPI
      pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      computing_timer(MPI_COMM_WORLD, pcout, TimerOutput::summary,
                      TimerOutput::wall_times),
#else
      pcout(std::cout),
      computing_timer(pcout, TimerOutput::summary, TimerOutput::wall_times),
#endif
      solver_control(1), solver(solver_control), rhs_function(),
      Dirichlet_bc_function(), rhs_function_omega(),
      Dirichlet_bc_function_omega(), radius(parameters.radius),
      lumpedAverage(parameters.lumpedAverage) {

  g = constructed_solution == 3
          ? (2 * numbers::PI) / (2 * numbers::PI + std::log(radius))
          : 1;
}

template <int dim, int dim_omega>
LDGPoissonProblem<dim, dim_omega>::~LDGPoissonProblem() {
  dof_handler.clear();
  dof_handler_omega.clear();
}

template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::make_grid() {
  TimerOutput::Scope t(computing_timer, "make grid");
  double offset = 0.0;
#if 0
  if (constructed_solution == 3) {

    // Calculate the shift vector
    
    Point<dim> shift_vector;

    if (dim == 3) {
      GridGenerator::cylinder(triangulation, 1, half_length);
      shift_vector = Point<dim>(half_length, 0 + offset, 0 + offset);
      pcout << "Shift vector " << shift_vector << std::endl;
      // Shift the cylinder by the half-length along the z-axis
      GridTools::shift(shift_vector, triangulation);
    }
    if (dim == 2) {
      Point<dim> center(0, 0);
      GridGenerator::hyper_ball(triangulation, center, 1);
    }

  } else
    GridGenerator::hyper_cube(triangulation, -extent, extent);
#else

  // Point<dim> p1 = Point<dim>(1, -std::sqrt(0.5) +offset,
  // -std::sqrt(0.5)+offset); Point<dim> p2 = Point<dim>(0,
  // std::sqrt(0.5)+offset, std::sqrt(0.5)+offset);

  Point<dim> p1 =
      Point<dim>(1, -std::sqrt(0.4) + offset, -std::sqrt(0.4) + offset);
  Point<dim> p2 =
      Point<dim>(0, std::sqrt(0.4) + offset, std::sqrt(0.4) + offset);
  // std::cout<<"hyper_rectangle "<<p1 << " "<<p2<<std::endl;
  GridGenerator::hyper_rectangle(triangulation, p1, p2);
#endif

  triangulation.refine_global(n_refine);
  double max_diameter = 0.0;
  typename Triangulation<dim>::cell_iterator cell = triangulation.begin(),
                                             endc = triangulation.end();
  for (; cell != endc; ++cell) {
    double cell_diameter = cell->diameter(); // Get the diameter of the cell
    if (cell_diameter > max_diameter) {
      max_diameter = cell_diameter;
    }

    for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell;
         face_no++) {
      Point<dim> p = cell->face(face_no)->center();
      if (cell->face(face_no)->at_boundary() && (p[0] != 0 || p[0] != 1)) {
        cell->face(face_no)->set_boundary_id(Dirichlet);
        // if ((p[0] == 0 || p[0] == 1) && constructed_solution == 3 && dim ==
        // 3) cell->face(face_no)->set_boundary_id(Neumann);
      }
    }
  }
  if (radius > max_diameter && !lumpedAverage) {
    std::cout << "!!!!!!!!!!!!!! MAX DIAMETER > RADIUS !!!!!!!!!!!!!!!!"
              << max_diameter << radius << std::endl;
    throw std::invalid_argument("MAX DIAMETER > RADIUS");
  }

  if (constructed_solution == 3)
    GridGenerator::hyper_cube(triangulation_omega, 0, 2 * half_length);
  else
    GridGenerator::hyper_cube(triangulation_omega, -extent / 2, extent / 2);
  triangulation_omega.refine_global(n_refine);

  typename Triangulation<dim_omega>::cell_iterator
      cell_omega = triangulation_omega.begin(),
      endc_omega = triangulation_omega.end();
  for (; cell_omega != endc_omega; ++cell_omega) {
    for (unsigned int face_no = 0;
         face_no < GeometryInfo<dim_omega>::faces_per_cell; face_no++) {
      if (cell_omega->face(face_no)->at_boundary())
        cell_omega->face(face_no)->set_boundary_id(Dirichlet);
    }
  }
}

template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::make_dofs() {
  TimerOutput::Scope t(computing_timer, "setup");

  dof_handler.distribute_dofs(fe);
  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  pcout << "dofs_per_cell " << dofs_per_cell << std::endl;
  DoFRenumbering::component_wise(dof_handler);

  dof_handler_omega.distribute_dofs(fe_omega);
  const unsigned int dofs_per_cell_omega = fe_omega.dofs_per_cell;
  pcout << "dofs_per_cell_omega " << dofs_per_cell_omega << std::endl;
  DoFRenumbering::component_wise(dof_handler_omega);

#if USE_MPI
  IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
#endif
  const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);

  unsigned int n_dofs_Potential = dofs_per_component[dim + dim_omega];
  const unsigned int n_vector_field =
      dim * dofs_per_component[0] + dim_omega * dofs_per_component[dim];
  const unsigned int n_potential = dofs_per_component[dim + dim_omega] +
                                   dofs_per_component[dim + dim_omega + 1];

  for (unsigned int i = 0; i < dofs_per_component.size(); i++)
    pcout << "dofs_per_component " << dofs_per_component[i] << std::endl;

  pcout << "Number of global active cells Omega: "
        << triangulation.n_global_active_cells() << std::endl
        << "Number of degrees of freedom: " << dof_handler.n_dofs() << " ("
        << n_vector_field << " + " << n_potential << ")" << std::endl;
  pcout << "Number of active cells omega: "
        << triangulation_omega.n_global_active_cells() << std::endl;

  start_VectorField_omega = dim * dofs_per_component[0];
  start_Potential_omega = n_vector_field + dofs_per_component[dim + dim_omega];
  start_Potential = n_vector_field;
  pcout << " start_VectorField_omega " << start_VectorField_omega
        << " start_Potential " << start_Potential << " start_Potential_omega "
        << start_Potential_omega << std::endl;

  constraints.clear();
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof_handler, dsp);

  const std::vector<types::global_dof_index> dofs_per_component_omega =
      DoFTools::count_dofs_per_fe_component(dof_handler_omega);
  unsigned int n_dofs_VectorField_omega = dofs_per_component_omega[0];
  unsigned int n_dofs_Potential_omega = dofs_per_component_omega[1];

  pcout << "start - extra dof coupling" << std::endl;
  // DoFs for 1D Inclusion. DoFs are fully connected, so spatial arrangment does
  // not matter
  for (unsigned int i = start_VectorField_omega;
       i < start_VectorField_omega + n_dofs_VectorField_omega; i++) {
    for (unsigned int j = start_VectorField_omega;
         j < start_VectorField_omega + n_dofs_VectorField_omega; j++) {
      dsp.add(i, j);
    }
  }

  for (unsigned int i = start_Potential_omega;
       i < start_Potential_omega + n_dofs_Potential_omega; i++) {
    for (unsigned int j = start_Potential_omega;
         j < start_Potential_omega + n_dofs_Potential_omega; j++) {
      dsp.add(i, j);
    }
  }

  for (unsigned int i = start_Potential_omega;
       i < start_Potential_omega + n_dofs_Potential_omega; i++) {
    for (unsigned int j = start_VectorField_omega;
         j < start_VectorField_omega + n_dofs_VectorField_omega; j++) {
      dsp.add(i, j);
    }
  }
  for (unsigned int i = start_VectorField_omega;
       i < start_VectorField_omega + n_dofs_VectorField_omega; i++) {
    for (unsigned int j = start_Potential_omega;
         j < start_Potential_omega + n_dofs_Potential_omega; j++) {
      dsp.add(i, j);
    }
  }

#if COUPLED
  {

    // coupling

    QGauss<dim> quadrature_formula(fe.degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula, update_flags);
    const Mapping<dim> &mapping = fe_values.get_mapping();

    QGauss<dim_omega> quadrature_formula_omega(fe.degree + 1);
    FEValues<dim_omega> fe_values_omega(fe_omega, quadrature_formula_omega,
                                        update_flags);

    pcout << "setup dofs Coupling" << std::endl;
    std::vector<types::global_dof_index> local_dof_indices_test(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_trial(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_omega(
        dofs_per_cell_omega);
    unsigned int nof_quad_points;
    bool AVERAGE = radius != 0 && !lumpedAverage;
    pcout << "AVERAGE " << AVERAGE << std::endl;
    // weight
    if (AVERAGE) {
      nof_quad_points = N_quad_points;
    } else {
      nof_quad_points = 1;
    }

    typename DoFHandler<dim_omega>::active_cell_iterator
        cell_omega = dof_handler_omega.begin_active(),
        endc_omega = dof_handler_omega.end();

    for (; cell_omega != endc_omega; ++cell_omega) {
      fe_values_omega.reinit(cell_omega);
      cell_omega->get_dof_indices(local_dof_indices_omega);
      dof_omega_to_Omega(dof_handler_omega, local_dof_indices_omega);

      std::vector<Point<dim_omega>> quadrature_points_omega =
          fe_values_omega.get_quadrature_points();

      for (unsigned int p = 0; p < quadrature_points_omega.size(); p++) {
        Point<dim_omega> quadrature_point_omega = quadrature_points_omega[p];

        // TODO hier über kreis iterieren
        std::vector<Point<dim>> quadrature_points_circle;
        Point<dim> quadrature_point_coupling;

        Point<dim> quadrature_point_trial;
        Point<dim> quadrature_point_test;

        if (dim == 2)
          quadrature_point_coupling =
              Point<dim>(quadrature_point_omega[0], y_l);
        if (dim == 3)
          quadrature_point_coupling =
              Point<dim>(quadrature_point_omega[0], y_l, z_l);

        Point<dim> normal_vector_omega;
        if (dim == 3)
          normal_vector_omega = Point<dim>(1, 0, 0);
        else
          normal_vector_omega = Point<dim>(1, 0);

        quadrature_points_circle = equidistant_points_on_circle<dim>(
            quadrature_point_coupling, radius, normal_vector_omega,
            nof_quad_points);

        // test function
        std::vector<double> my_quadrature_weights = {1};
        quadrature_point_test = quadrature_point_coupling;

#if TEST
        auto cell_test_array = GridTools::find_all_active_cells_around_point(
            mapping, dof_handler, quadrature_point_test);
        // std::cout << "cell_test_array " << cell_test_array.size() <<
        // std::endl;

        for (auto cellpair : cell_test_array)
#else
        auto cell_test = GridTools::find_active_cell_around_point(
            dof_handler, quadrature_point_test);
#endif

        {
#if TEST
          auto cell_test = cellpair.first;
#endif

#if USE_MPI
          if (cell_test != dof_handler.end())
            if (cell_test->is_locally_owned())
#endif
            {
              //	std::cout<<cell_test<<" ";
              cell_test->get_dof_indices(local_dof_indices_test);

              Point<dim> quadrature_point_test_mapped_cell =
                  mapping.transform_real_to_unit_cell(cell_test,
                                                      quadrature_point_test);
              std::vector<Point<dim>> my_quadrature_points_test = {
                  quadrature_point_test_mapped_cell};
              const Quadrature<dim> my_quadrature_formula_test(
                  my_quadrature_points_test, my_quadrature_weights);
              FEValues<dim> fe_values_coupling_test(
                  fe, my_quadrature_formula_test, update_flags_coupling);
              fe_values_coupling_test.reinit(cell_test);

              // std::cout << "coupled " << std::endl;
              for (unsigned int q_avag = 0; q_avag < nof_quad_points;
                   q_avag++) {
                // Quadrature weights and points
                quadrature_point_trial = quadrature_points_circle[q_avag];

#if TEST
                auto cell_trial_array =
                    GridTools::find_all_active_cells_around_point(
                        mapping, dof_handler, quadrature_point_trial);
                //  std::cout << "cell_trial_array " << cell_trial_array.size()
                //  << std::endl;

                for (auto cellpair_trial : cell_trial_array)
#else
              auto cell_trial = GridTools::find_active_cell_around_point(
                  dof_handler, quadrature_point_trial);
#endif

                {
#if TEST
                  auto cell_trial = cellpair_trial.first;
#endif
                  if (cell_trial != dof_handler.end())
                    if (cell_trial->is_locally_owned() &&
                        cell_test->is_locally_owned()) {

                      cell_trial->get_dof_indices(local_dof_indices_trial);

                      for (unsigned int i = 0;
                           i < local_dof_indices_test.size(); i++) {
                        for (unsigned int j = 0;
                             j < local_dof_indices_trial.size(); j++) {
                          dsp.add(local_dof_indices_test[i],
                                  local_dof_indices_trial[j]);
                        }
                      }

                      for (unsigned int i = 0;
                           i < local_dof_indices_omega.size(); i++) {
                        for (unsigned int j = 0;
                             j < local_dof_indices_trial.size(); j++) {
                          dsp.add(local_dof_indices_omega[i],
                                  local_dof_indices_trial[j]);
                        }
                      }

                      for (unsigned int i = 0;
                           i < local_dof_indices_test.size(); i++) {
                        for (unsigned int j = 0;
                             j < local_dof_indices_omega.size(); j++) {
                          dsp.add(local_dof_indices_test[i],
                                  local_dof_indices_omega[j]);
                        }
                      }

                      for (unsigned int i = 0;
                           i < local_dof_indices_omega.size(); i++) {
                        for (unsigned int j = 0;
                             j < local_dof_indices_omega.size(); j++) {
                          dsp.add(local_dof_indices_omega[i],
                                  local_dof_indices_omega[j]);
                        }
                      }
                    }
                  //       else
                  // std::cout<<"düdüm"<<std::endl;
                }
              }
            }
          //   else
          // std::cout<<"düdüm"<<std::endl;
        }
        // std::cout<<std::endl;
      }
    }
  }
#endif

#if USE_MPI

  SparsityTools::distribute_sparsity_pattern(
      dsp, dof_handler.locally_owned_dofs(), MPI_COMM_WORLD,
      locally_relevant_dofs);

  system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp,
                       MPI_COMM_WORLD);

  solution.reinit(locally_relevant_dofs, MPI_COMM_WORLD);
  system_rhs.reinit(locally_owned_dofs, MPI_COMM_WORLD);

#else

  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

#endif
}

template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::assemble_system() {
  TimerOutput::Scope t(computing_timer, "assembly");
  pcout << "assemble_system" << std::endl;

  QGauss<dim> quadrature_formula(fe.degree + 1);
  QGauss<dim - 1> face_quadrature_formula(fe.degree + 1);

  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> local_neighbor_dof_indices(
      dofs_per_cell);

  const IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();

  FEValues<dim> fe_values(fe, quadrature_formula, update_flags);

  FEFaceValues<dim> fe_face_values(fe, face_quadrature_formula,
                                   face_update_flags);

  FEFaceValues<dim> fe_neighbor_face_values(fe, face_quadrature_formula,
                                            face_update_flags);

  FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> local_vector(dofs_per_cell);

  FullMatrix<double> vi_ui_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> vi_ue_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> ve_ui_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> ve_ue_matrix(dofs_per_cell, dofs_per_cell);

  const Mapping<dim> &mapping = fe_values.get_mapping();
#if 1
  {
    TimerOutput::Scope t(computing_timer, "assembly - Omega");
    pcout << "assemly - Omega" << std::endl;

    //
    // <code>vi_ui</code> - Taking the value of the test function from
    //         interior of this cell's face and the solution function
    //         from the interior of this cell.
    //
    // <code>vi_ue</code> - Taking the value of the test function from
    //         interior of this cell's face and the solution function
    //         from the exterior of this cell.
    //
    // <code>ve_ui</code> - Taking the value of the test function from
    //         exterior of this cell's face and the solution function
    //         from the interior of this cell.
    //
    // <code>ve_ue</code> - Taking the value of the test function from
    //         exterior of this cell's face and the solution function
    //         from the exterior of this cell.

    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler
                                                              .begin_active(),
                                                   endc = dof_handler.end();
    // unsigned int cell_number = 0;
    for (; cell != endc; ++cell) {
      // std::cout<<"cell_number "<<cell_number<<std::endl;
      // cell_number++;
      // unsigned int cell_id = cell->index();
      // std::cout<<cell_id<<std::endl;
#if USE_MPI
      if (cell->is_locally_owned())
#endif
      {

        local_matrix = 0;
        local_vector = 0;

        fe_values.reinit(cell);
        assemble_cell_terms(fe_values, local_matrix, local_vector,
                            K_inverse_function, rhs_function, VectorField,
                            Potential);

        cell->get_dof_indices(local_dof_indices);

        for (unsigned int face_no = 0;
             face_no < GeometryInfo<dim>::faces_per_cell; face_no++) {
          typename DoFHandler<dim>::face_iterator face = cell->face(face_no);

          if (face->at_boundary()) {
            fe_face_values.reinit(cell, face_no);

            if (face->boundary_id() == Dirichlet) {
              // std::cout<<"bound"<<std::endl;
              double h = cell->diameter();
              assemble_Dirichlet_boundary_terms(
                  fe_face_values, local_matrix, local_vector, h,
                  Dirichlet_bc_function, VectorField, Potential);
            } else if (face->boundary_id() == Neumann) {
              assemble_Neumann_boundary_terms(fe_face_values, local_matrix,
                                              local_vector,
                                              Neumann_bc_function);
            } else
              Assert(false, ExcNotImplemented());
          } else {

            Assert(cell->neighbor(face_no).state() == IteratorState::valid,
                   ExcInternalError());

            typename DoFHandler<dim>::cell_iterator neighbor =
                cell->neighbor(face_no);

            if (cell->id() < neighbor->id()) {

              const unsigned int neighbor_face_no =
                  cell->neighbor_of_neighbor(face_no);

              vi_ui_matrix = 0;
              vi_ue_matrix = 0;
              ve_ui_matrix = 0;
              ve_ue_matrix = 0;

              fe_face_values.reinit(cell, face_no);
              fe_neighbor_face_values.reinit(neighbor, neighbor_face_no);

              double h = std::min(cell->diameter(), neighbor->diameter());

              assemble_flux_terms(fe_face_values, fe_neighbor_face_values,
                                  vi_ui_matrix, vi_ue_matrix, ve_ui_matrix,
                                  ve_ue_matrix, h, VectorField, Potential);

              neighbor->get_dof_indices(local_neighbor_dof_indices);

              distribute_local_flux_to_global(
                  vi_ui_matrix, vi_ue_matrix, ve_ui_matrix, ve_ue_matrix,
                  local_dof_indices, local_neighbor_dof_indices);
            }
          }
        }

        constraints.distribute_local_to_global(local_matrix, local_dof_indices,
                                               system_matrix);

        constraints.distribute_local_to_global(local_vector, local_dof_indices,
                                               system_rhs);
      }
    }
  }
#endif
  // std::cout << "loop z uend" << std::endl;
#if USE_MPI
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
#endif

  // omega
  QGauss<dim_omega> quadrature_formula_omega(fe.degree + 1);
  QGauss<dim_omega - 1> face_quadrature_formula_omega(fe.degree + 1);

  FEValues<dim_omega> fe_values_omega(fe_omega, quadrature_formula_omega,
                                      update_flags);

  FEFaceValues<dim_omega> fe_face_values_omega(
      fe_omega, face_quadrature_formula_omega, face_update_flags);

  FEFaceValues<dim_omega> fe_neighbor_face_values_omega(
      fe_omega, face_quadrature_formula_omega, face_update_flags);

  const unsigned int dofs_per_cell_omega = fe_omega.dofs_per_cell;
  // std::cout<<"dofOmega "<<dofs_per_cell_omega<<std::endl;
  std::vector<types::global_dof_index> local_dof_indices_omega(
      dofs_per_cell_omega);
  std::vector<types::global_dof_index> local_neighbor_dof_indices_omega(
      dofs_per_cell_omega);

  std::vector<types::global_dof_index> local_dof_indices_omega_locally_owned;
  std::vector<types::global_dof_index>
      local_neighbor_dof_indices_omega_locally_owned;

  FullMatrix<double> local_matrix_omega(dofs_per_cell_omega,
                                        dofs_per_cell_omega);
  Vector<double> local_vector_omega(dofs_per_cell_omega);

  FullMatrix<double> vi_ui_matrix_omega(dofs_per_cell_omega,
                                        dofs_per_cell_omega);
  FullMatrix<double> vi_ue_matrix_omega(dofs_per_cell_omega,
                                        dofs_per_cell_omega);
  FullMatrix<double> ve_ui_matrix_omega(dofs_per_cell_omega,
                                        dofs_per_cell_omega);
  FullMatrix<double> ve_ue_matrix_omega(dofs_per_cell_omega,
                                        dofs_per_cell_omega);

  std::vector<unsigned int> indices_i;
  std::vector<unsigned int> indices_j;

  typename DoFHandler<dim_omega>::active_cell_iterator
      cell_omega = dof_handler_omega.begin_active(),
      endc_omega = dof_handler_omega.end();

#if USE_MPI
// if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
#endif
  {
    TimerOutput::Scope t(computing_timer, "assembly - omega");
    pcout << "assemly - omega" << std::endl;

    for (; cell_omega != endc_omega; ++cell_omega) {
      // unsigned int cell_id_omega = cell_omega->index();
      // std::cout<<cell_id_omega<<std::endl;

      local_matrix_omega = 0;
      local_vector_omega = 0;

      fe_values_omega.reinit(cell_omega);

      assemble_cell_terms(fe_values_omega, local_matrix_omega,
                          local_vector_omega, k_inverse_function,
                          rhs_function_omega, VectorField_omega,
                          Potential_omega);

      cell_omega->get_dof_indices(local_dof_indices_omega);

      dof_omega_to_Omega(dof_handler_omega, local_dof_indices_omega);

      indices_i.clear();
      local_dof_indices_omega_locally_owned.clear();
      for (unsigned int i = 0; i < local_dof_indices_omega.size(); i++) {
        types::global_dof_index dof_index = local_dof_indices_omega[i];
        if (locally_owned_dofs.is_element(dof_index)) {
          indices_i.push_back(dof_index);
          local_dof_indices_omega_locally_owned.push_back(dof_index);
        }
      }

      for (unsigned int face_no_omega = 0;
           face_no_omega < GeometryInfo<dim_omega>::faces_per_cell;
           face_no_omega++) {
        //   std::cout<<"face_no_omega "<<face_no_omega<<std::endl;
        typename DoFHandler<dim_omega>::face_iterator face_omega =
            cell_omega->face(face_no_omega);

        if (face_omega->at_boundary()) {
          fe_face_values_omega.reinit(cell_omega, face_no_omega);

          if (face_omega->boundary_id() == Dirichlet) {
            double h = cell_omega->diameter();
            assemble_Dirichlet_boundary_terms(
                fe_face_values_omega, local_matrix_omega, local_vector_omega, h,
                Dirichlet_bc_function_omega, VectorField_omega,
                Potential_omega);
          }
          /*else if (face_omega->boundary_id() == Neumann)
            {
              assemble_Neumann_boundary_terms(fe_face_values_omega,
                                              local_matrix_omega,
                                              local_vector_omega);
            }*/
          else
            Assert(false, ExcNotImplemented());
        } else {

          Assert(cell_omega->neighbor(face_no_omega).state() ==
                     IteratorState::valid,
                 ExcInternalError());

          typename DoFHandler<dim_omega>::cell_iterator neighbor_omega =
              cell_omega->neighbor(face_no_omega);

          if (cell_omega->id() < neighbor_omega->id()) {

            const unsigned int neighbor_face_no_omega =
                cell_omega->neighbor_of_neighbor(face_no_omega);

            vi_ui_matrix_omega = 0;
            vi_ue_matrix_omega = 0;
            ve_ui_matrix_omega = 0;
            ve_ue_matrix_omega = 0;

            fe_face_values_omega.reinit(cell_omega, face_no_omega);
            fe_neighbor_face_values_omega.reinit(neighbor_omega,
                                                 neighbor_face_no_omega);

            double h =
                std::min(cell_omega->diameter(), neighbor_omega->diameter());

            assemble_flux_terms(
                fe_face_values_omega, fe_neighbor_face_values_omega,
                vi_ui_matrix_omega, vi_ue_matrix_omega, ve_ui_matrix_omega,
                ve_ue_matrix_omega, h, VectorField_omega, Potential_omega);

            neighbor_omega->get_dof_indices(local_neighbor_dof_indices_omega);
            dof_omega_to_Omega(dof_handler_omega,
                               local_neighbor_dof_indices_omega);

            /*indices_j.clear();
            local_neighbor_dof_indices_omega_locally_owned.clear();
            for (unsigned int i = 0;
                 i < local_neighbor_dof_indices_omega.size(); i++) {
              types::global_dof_index dof_index =
                  local_neighbor_dof_indices_omega[i];
              if (locally_owned_dofs.is_element(dof_index)) {
                local_neighbor_dof_indices_omega_locally_owned.push_back(
                    dof_index);
                indices_j.push_back(dof_index);
              }
            }*/

            distribute_local_flux_to_global(
                vi_ui_matrix_omega, vi_ue_matrix_omega, ve_ui_matrix_omega,
                ve_ue_matrix_omega, local_dof_indices_omega,
                local_neighbor_dof_indices_omega);
          }
        }
      }

      constraints.distribute_local_to_global(
          local_matrix_omega, local_dof_indices_omega, system_matrix);

      constraints.distribute_local_to_global(
          local_vector_omega, local_dof_indices_omega, system_rhs);
      //#endif
    }
  }
#if USE_MPI
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
#endif
  // std::cout << "ende omega loop" << std::endl;

#if USE_MPI
// if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
#endif
  if (dim == 2 && constructed_solution == 3) {
    std::cout << "dim == 2 && constructed_solution == 3" << std::endl;
    Point<dim> quadrature_point_test(y_l, z_l);
    std::vector<types::global_dof_index> local_dof_indices_test(dofs_per_cell);
    // test function
    std::vector<double> my_quadrature_weights = {1};
    unsigned int n_te;
#if TEST
    auto cell_test_array = GridTools::find_all_active_cells_around_point(
        mapping, dof_handler, quadrature_point_test);
    n_te = cell_test_array.size();
    //   n_te = 1;
    pcout << "cell_test_array " << cell_test_array.size() << std::endl;

    for (auto cellpair : cell_test_array)
#else
    auto cell_test = GridTools::find_active_cell_around_point(
        dof_handler, quadrature_point_test);
    n_te = 1;
#endif

    {
#if TEST
      auto cell_test = cellpair.first;
#endif

#if USE_MPI
      if (cell_test != dof_handler.end())
        if (cell_test->is_locally_owned())
#endif
        {
          // std::cout << "cell_test->center() " << cell_test->center()
          //         << std::endl;
          cell_test->get_dof_indices(local_dof_indices_test);

          //-------------face -----------------
          unsigned int n_ftest = 0;
          for (unsigned int face_no = 0;
               face_no < GeometryInfo<dim>::faces_per_cell; face_no++) {
            typename DoFHandler<dim>::face_iterator face_test =
                cell_test->face(face_no);
            auto bounding_box = face_test->bounding_box();
            n_ftest += bounding_box.point_inside(quadrature_point_test) == true;
          }

          for (unsigned int face_no = 0;
               face_no < GeometryInfo<dim>::faces_per_cell; face_no++) {
            typename DoFHandler<dim>::face_iterator face_test =
                cell_test->face(face_no);

            Point<dim - 1> quadrature_point_test_mapped_face =
                mapping.project_real_point_to_unit_point_on_face(
                    cell_test, face_no, quadrature_point_test);

            auto bounding_box = face_test->bounding_box();

            if (bounding_box.point_inside(quadrature_point_test) == true) {
              /*    std::cout << "face_test->center() " << face_test->center()
                          << " quadrature_point_test_mapped_face "
                          << quadrature_point_test_mapped_face << " isinside "
                          << bounding_box.point_inside(quadrature_point_test)
                          << std::endl;*/
              std::vector<Point<dim - 1>> quadrature_point_test_face = {
                  quadrature_point_test_mapped_face};
              const Quadrature<dim - 1> my_quadrature_formula_test(
                  quadrature_point_test_face, my_quadrature_weights);
              FEFaceValues<dim> fe_values_coupling_test_face(
                  fe, my_quadrature_formula_test, update_flags_coupling);
              fe_values_coupling_test_face.reinit(cell_test, face_no);
              unsigned int n_face_points =
                  fe_values_coupling_test_face.n_quadrature_points;
              unsigned int dofs_this_cell =
                  fe_values_coupling_test_face.dofs_per_cell;
              // std::cout << "n_face_points " << n_face_points << std::endl;
              local_vector = 0;
              for (unsigned int q = 0; q < n_face_points; ++q) {
                for (unsigned int i = 0; i < dofs_this_cell; ++i) {
                  local_vector(i) +=
                      fe_values_coupling_test_face[Potential].value(i, q) * 1 /
                      (n_te * n_ftest);
                }
              }
              constraints.distribute_local_to_global(
                  local_vector, local_dof_indices_test, system_rhs);
            }
          }
          //-------------face ----------------- ende
          if (n_ftest == 0) {
            Point<dim> quadrature_point_test_mapped_cell =
                mapping.transform_real_to_unit_cell(cell_test,
                                                    quadrature_point_test);
            std::cout << "quadrature_point_test_mapped_cell "
                      << quadrature_point_test_mapped_cell << std::endl;
            std::vector<Point<dim>> my_quadrature_points_test = {
                quadrature_point_test_mapped_cell};
            const Quadrature<dim> my_quadrature_formula_test(
                my_quadrature_points_test, my_quadrature_weights);
            FEValues<dim> fe_values_coupling_test(
                fe, my_quadrature_formula_test,
                update_flags_coupling); // hier ist der fehler. wenn zweimal in
                                        // einer Cell integriert wird, stimmt es
                                        // nicht
            fe_values_coupling_test.reinit(cell_test);
            //  fe_values.reinit(cell_test);
            for (unsigned int i = 0; i < dofs_per_cell; i++) {
              //  std::cout<< fe_values_coupling_test[Potential].value(i, 0)<< "
              //  ";

              local_vector(i) +=
                  fe_values_coupling_test[Potential].value(i, 0); //
            }
            //  std::cout<<std::endl;
            constraints.distribute_local_to_global(
                local_vector, local_dof_indices_test, system_rhs);
          }
        }
    }
  }
  if (dim == 3 && constructed_solution == 3) {
    TimerOutput::Scope t(computing_timer, "assembly - coupling");
    // coupling
    pcout << "assemble Coupling" << std::endl;
    std::vector<types::global_dof_index> local_dof_indices_test(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_trial(dofs_per_cell);

    FullMatrix<double> V_U_matrix_coupling(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> v_U_matrix_coupling(dofs_per_cell_omega, dofs_per_cell);
    FullMatrix<double> V_u_matrix_coupling(dofs_per_cell, dofs_per_cell_omega);
    FullMatrix<double> v_u_matrix_coupling(dofs_per_cell_omega,
                                           dofs_per_cell_omega);

    bool insideCell_test = true;
    bool insideCell_trial = true;
    unsigned int nof_quad_points;
    bool AVERAGE = radius != 0 && !lumpedAverage;
    pcout << "AVERAGE " << AVERAGE << std::endl;

    // weight
    if (AVERAGE) {
      nof_quad_points = N_quad_points;
    } else {
      nof_quad_points = 1;
    }

    cell_omega = dof_handler_omega.begin_active();
    endc_omega = dof_handler_omega.end();

    for (; cell_omega != endc_omega; ++cell_omega) {
      fe_values_omega.reinit(cell_omega);
      cell_omega->get_dof_indices(local_dof_indices_omega);
      dof_omega_to_Omega(dof_handler_omega, local_dof_indices_omega);

      std::vector<Point<dim_omega>> quadrature_points_omega =
          fe_values_omega.get_quadrature_points();

      for (unsigned int p = 0; p < quadrature_points_omega.size(); p++) {
        Point<dim_omega> quadrature_point_omega = quadrature_points_omega[p];

        // TODO hier über kreis iterieren
        std::vector<Point<dim>> quadrature_points_circle;
        Point<dim> quadrature_point_coupling;

        Point<dim> quadrature_point_trial;
        Point<dim> quadrature_point_test;

        if (dim == 2)
          quadrature_point_coupling =
              Point<dim>(quadrature_point_omega[0], y_l);
        if (dim == 3)
          quadrature_point_coupling =
              Point<dim>(quadrature_point_omega[0], y_l, z_l);

        Point<dim> normal_vector_omega;
        if (dim == 3)
          normal_vector_omega = Point<dim>(1, 0, 0);
        else
          normal_vector_omega = Point<dim>(1, 0);

        quadrature_points_circle = equidistant_points_on_circle<dim>(
            quadrature_point_coupling, radius, normal_vector_omega,
            nof_quad_points);

        // test function
        std::vector<double> my_quadrature_weights = {1};
        quadrature_point_test = quadrature_point_coupling;

        // std::cout<< "------quadrature_point_test " <<  quadrature_point_test
        // << std::endl;
        unsigned int n_te;
#if TEST
        auto cell_test_array = GridTools::find_all_active_cells_around_point(
            mapping, dof_handler, quadrature_point_test);
        n_te = cell_test_array.size();
        //   n_te = 1;
        // pcout << "cell_test_array " << cell_test_array.size() << std::endl;
        for (auto cellpair : cell_test_array)
#else
        auto cell_test = GridTools::find_active_cell_around_point(
            dof_handler, quadrature_point_test);
        n_te = 1;
#endif

        {
#if TEST
          auto cell_test = cellpair.first;
#endif

#if USE_MPI
          if (cell_test != dof_handler.end())
            if (cell_test->is_locally_owned())
#endif
            {
              // std::cout<< "cell_test " << cell_test<< " "
              // <<cell_test->center() << std::endl;
              cell_test->get_dof_indices(local_dof_indices_test);

              std::vector<unsigned int> face_no_test;
              unsigned int n_ftest = 0;
              for (unsigned int face_no = 0;
                   face_no < GeometryInfo<dim>::faces_per_cell; face_no++) {
                typename DoFHandler<dim>::face_iterator face_test =
                    cell_test->face(face_no);
                auto bounding_box = face_test->bounding_box();
                n_ftest +=
                    bounding_box.point_inside(quadrature_point_test,
                                              distance_tolerance) == true;
                face_no_test.push_back(face_no);
              }
              if (n_ftest == 0) {
                insideCell_test = true;
                n_ftest = 1;
              } else {
                insideCell_test = false;
                // std::cout<<"insideCell_test = false"<<std::endl;
              }

              // std::cout << "n_ftest " << n_ftest << " insideCell_test "
              // <<insideCell_test<<std::endl;

              Point<dim> quadrature_point_test_mapped_cell =
                  mapping.transform_real_to_unit_cell(cell_test,
                                                      quadrature_point_test);

              std::vector<Point<dim>> my_quadrature_points_test = {
                  quadrature_point_test_mapped_cell};
              const Quadrature<dim> my_quadrature_formula_test(
                  my_quadrature_points_test, my_quadrature_weights);
              FEValues<dim> fe_values_coupling_test(
                  fe, my_quadrature_formula_test, update_flags_coupling);
              fe_values_coupling_test.reinit(cell_test);

#if !COUPLED
              std::cout << "not coupled" << std::endl;
              //-------------face -----------------
              // n_ftest = 0;
              if (!insideCell_test) {
                pcout << "Omega rhs face " << std::endl;
                for (unsigned int face_no = 0;
                     face_no < GeometryInfo<dim>::faces_per_cell; face_no++) {
                  typename DoFHandler<dim>::face_iterator face_test =
                      cell_test->face(face_no);

                  Point<dim - 1> quadrature_point_test_mapped_face =
                      mapping.project_real_point_to_unit_point_on_face(
                          cell_test, face_no, quadrature_point_test);

                  auto bounding_box = face_test->bounding_box();

                  if (bounding_box.point_inside(quadrature_point_test,
                                                distance_tolerance) == true) {
                    /*  std::cout << "face_test->center() " <<
                       face_test->center()
                          << " quadrature_point_test_mapped_face "
                          << quadrature_point_test_mapped_face << " isinside "
                          << bounding_box.point_inside(quadrature_point_test,
                                                       distance_tolerance)
                          << std::endl;*/

                    std::vector<Point<dim - 1>> quadrature_point_test_face = {
                        quadrature_point_test_mapped_face};
                    const Quadrature<dim - 1> my_quadrature_formula_test(
                        quadrature_point_test_face, my_quadrature_weights);
                    FEFaceValues<dim> fe_values_coupling_test_face(
                        fe, my_quadrature_formula_test, update_flags_coupling);
                    fe_values_coupling_test_face.reinit(cell_test, face_no);
                    unsigned int n_face_points =
                        fe_values_coupling_test_face.n_quadrature_points;
                    unsigned int dofs_this_cell =
                        fe_values_coupling_test_face.dofs_per_cell;

                    local_vector = 0;
                    for (unsigned int q = 0; q < n_face_points; ++q) {
                      for (unsigned int i = 0; i < dofs_this_cell; ++i) {
                        local_vector(i) +=
                            fe_values_coupling_test_face[Potential].value(i,
                                                                          q) *
                            1 / (n_te * n_ftest) *
                            (1 + quadrature_point_omega[0]) *
                            fe_values_omega.JxW(p);
                      }
                    }
                    constraints.distribute_local_to_global(
                        local_vector, local_dof_indices_test, system_rhs);
                  }
                }
              }
              //-------------face ----------------- ende

              if (insideCell_test) {

                pcout << "Omega rhs insideCell" << std::endl;
                local_vector = 0;
                const unsigned int n_q_points = fe_values.n_quadrature_points;
                // for (unsigned int q = 0; q < n_q_points; ++q) {
                for (unsigned int i = 0; i < dofs_per_cell; i++) {
                  local_vector(i) +=
                      fe_values_coupling_test[Potential].value(i, 0) *
                      (1 + quadrature_point_omega[0]) * fe_values_omega.JxW(p);
                }
                constraints.distribute_local_to_global(
                    local_vector, local_dof_indices_test, system_rhs);
              }
#endif

#if COUPLED
              // std::cout << "coupled " << std::endl;
              for (unsigned int q_avag = 0; q_avag < nof_quad_points;
                   q_avag++) {
                // Quadrature weights and points
                quadrature_point_trial = quadrature_points_circle[q_avag];
                //  std::cout<< "quadrature_point_trial " <<
                //  quadrature_point_trial << std::endl;
                double weight;
                double C_avag;
                if (AVERAGE) {
                  double perimeter = 2.0 * numbers::PI * radius;
                  double h_avag = perimeter / (nof_quad_points);

                  double weights_odd = 4.0 / 3.0 * h_avag;
                  double weights_even = 2.0 / 3.0 * h_avag;
                  double weights_first_last = h_avag / 3.0;

                  C_avag = 1.0 / (2.0 * numbers::PI);
                  // std::cout<< "h_avag " << h_avag <<" weights_odd "<<
                  // weights_odd<< " weights_even "<<weights_even<< " q_avag % 2
                  // == 0 " << (int)(q_avag % 2 == 0)<< std::endl;
                  if (q_avag == 0)
                    weight = 2 * weights_first_last;
                  else {

                    if (q_avag % 2 == 0)
                      weight = weights_even;
                    else
                      weight = weights_odd;
                  }
                  weight = ((2.0 * numbers::PI * radius) / (nof_quad_points));
                } else {
                  weight = 1;
                  C_avag = 1;
                }
                weight = 1.0 / nof_quad_points;
                C_avag = 1.0;
                //    std::cout<< "q_avag " << q_avag <<" weight "<< weight<<"
                //    C_avag "<<C_avag<<" nof_quad_points
                //    "<<nof_quad_points<<std::endl;
                unsigned int n_tr;
#if TEST
                auto cell_trial_array =
                    GridTools::find_all_active_cells_around_point(
                        mapping, dof_handler, quadrature_point_trial);
                // pcout<< "cell_trial_array " << cell_trial_array.size() <<
                // std::endl;
                n_tr = cell_trial_array.size();
                // n_tr  =1;
                for (auto cellpair_trial : cell_trial_array)
#else
                auto cell_trial = GridTools::find_active_cell_around_point(
                    dof_handler, quadrature_point_trial);
                n_tr = 1;
#endif

                {
#if TEST
                  auto cell_trial = cellpair_trial.first;
#endif
                  if (cell_trial != dof_handler.end())
                    if (cell_trial->is_locally_owned() &&
                        cell_test->is_locally_owned()) {
                      //  std::cout<< "cell_trial " << cell_trial<< " "
                      //  <<cell_trial->center() << std::endl;
                      cell_trial->get_dof_indices(local_dof_indices_trial);

                      //-----------------------cell--------------------------------------

                      Point<dim> quadrature_point_trial_mapped_cell =
                          mapping.transform_real_to_unit_cell(
                              cell_trial, quadrature_point_trial);

                      std::vector<Point<dim>> my_quadrature_points_trial = {
                          quadrature_point_trial_mapped_cell};

                      const Quadrature<dim> my_quadrature_formula_trial(
                          my_quadrature_points_trial, my_quadrature_weights);

                      FEValues<dim> fe_values_coupling_trial(
                          fe, my_quadrature_formula_trial,
                          update_flags_coupling);
                      fe_values_coupling_trial.reinit(cell_trial);

                      unsigned int n_ftrial = 0;
                      std::vector<unsigned int> face_no_trial;
                      for (unsigned int face_no = 0;
                           face_no < GeometryInfo<dim>::faces_per_cell;
                           face_no++) {
                        typename DoFHandler<dim>::face_iterator face_trial =
                            cell_trial->face(face_no);
                        auto bounding_box = face_trial->bounding_box();
                        n_ftrial += bounding_box.point_inside(
                                        quadrature_point_trial,
                                        distance_tolerance) == true;
                        face_no_trial.push_back(face_no);
                      }
                      if (n_ftrial == 0) {
                        insideCell_trial = true;
                        n_ftrial = 1;

                      } else {
                        insideCell_trial = false;
                        //   std::cout<<"insideCell_trial = false"<<std::endl;
                      }

                      //     std::cout << "n_ftrial " << n_ftrial << "
                      //     insideCell_trial " <<insideCell_trial<<std::endl;
                      //     // hier ist der Fehler: also unten, nicht jeder

                      for (unsigned int ftest = 0; ftest < n_ftest; ftest++) {
                        Point<dim - 1> quadrature_point_test_mapped_face =
                            mapping.project_real_point_to_unit_point_on_face(
                                cell_test, face_no_test[ftest],
                                quadrature_point_test);

                        std::vector<Point<dim - 1>> quadrature_point_test_face =
                            {quadrature_point_test_mapped_face};
                        const Quadrature<dim - 1> my_quadrature_formula_test(
                            quadrature_point_test_face, my_quadrature_weights);
                        FEFaceValues<dim> fe_values_coupling_test_face(
                            fe, my_quadrature_formula_test,
                            update_flags_coupling);
                        fe_values_coupling_test_face.reinit(
                            cell_test, face_no_test[ftest]);

                        for (unsigned int ftrial = 0; ftrial < n_ftrial;
                             ftrial++) {

                          Point<dim - 1> quadrature_point_trial_mapped_face =
                              mapping.project_real_point_to_unit_point_on_face(
                                  cell_trial, face_no_trial[ftrial],
                                  quadrature_point_trial);

                          std::vector<Point<dim - 1>>
                              quadrature_point_trial_face = {
                                  quadrature_point_trial_mapped_face};
                          const Quadrature<dim - 1> my_quadrature_formula_trial(
                              quadrature_point_trial_face,
                              my_quadrature_weights);
                          FEFaceValues<dim> fe_values_coupling_trial_face(
                              fe, my_quadrature_formula_trial,
                              update_flags_coupling);
                          fe_values_coupling_trial_face.reinit(
                              cell_trial, face_no_trial[ftest]);

                          V_U_matrix_coupling = 0;
                          v_U_matrix_coupling = 0;
                          V_u_matrix_coupling = 0;
                          v_u_matrix_coupling = 0;

                          double psi_potential_test;
                          double psi_potential_trial;

                          for (unsigned int i = 0; i < dofs_per_cell; i++) {
                            if (insideCell_test)
                              psi_potential_test =
                                  fe_values_coupling_test[Potential].value(i,
                                                                           0);
                            else
                              psi_potential_test =
                                  fe_values_coupling_test_face[Potential].value(
                                      i, 0);

                            for (unsigned int j = 0; j < dofs_per_cell; j++) {
                              if (insideCell_trial)
                                psi_potential_trial =
                                    fe_values_coupling_trial[Potential].value(
                                        j, 0);
                              else
                                psi_potential_trial =
                                    fe_values_coupling_trial_face[Potential]
                                        .value(j, 0);

                              V_U_matrix_coupling(i, j) +=
                                  g * psi_potential_test * psi_potential_trial *
                                  C_avag * weight * fe_values_omega.JxW(p) * 1 /
                                  (n_tr * n_ftrial) * 1 / (n_te * n_ftest);
                            }
                          }
                          constraints.distribute_local_to_global(
                              V_U_matrix_coupling, local_dof_indices_test,
                              local_dof_indices_trial, system_matrix);

                          for (unsigned int i = 0; i < dofs_per_cell_omega;
                               i++) {
                            for (unsigned int j = 0; j < dofs_per_cell; j++) {
                              if (insideCell_trial)
                                psi_potential_trial =
                                    fe_values_coupling_trial[Potential].value(
                                        j, 0);
                              else
                                psi_potential_trial =
                                    fe_values_coupling_trial_face[Potential]
                                        .value(j, 0);

                              v_U_matrix_coupling(i, j) +=
                                  -g *
                                  fe_values_omega[Potential_omega].value(i, p) *
                                  C_avag * weight * psi_potential_trial *
                                  fe_values_omega.JxW(p) * 1 /
                                  (n_tr * n_ftrial) * 1 / (n_te * n_ftest);
                            }
                          }
                          constraints.distribute_local_to_global(
                              v_U_matrix_coupling, local_dof_indices_omega,
                              local_dof_indices_trial, system_matrix);

                          for (unsigned int i = 0; i < dofs_per_cell; i++) {
                            if (insideCell_test)
                              psi_potential_test =
                                  fe_values_coupling_test[Potential].value(i,
                                                                           0);
                            else
                              psi_potential_test =
                                  fe_values_coupling_test_face[Potential].value(
                                      i, 0);

                            for (unsigned int j = 0; j < dofs_per_cell_omega;
                                 j++) {
                              V_u_matrix_coupling(i, j) +=
                                  -g * psi_potential_test *
                                  fe_values_omega[Potential_omega].value(j, p) *
                                  fe_values_omega.JxW(p) * 1 /
                                  (n_tr * n_ftrial) * 1 / (n_te * n_ftest) *
                                  C_avag * weight;
                            }
                          }
                          constraints.distribute_local_to_global(
                              V_u_matrix_coupling, local_dof_indices_test,
                              local_dof_indices_omega, system_matrix);

                          for (unsigned int i = 0; i < dofs_per_cell_omega;
                               i++) {
                            for (unsigned int j = 0; j < dofs_per_cell_omega;
                                 j++) {
                              v_u_matrix_coupling(i, j) +=
                                  g *
                                  fe_values_omega[Potential_omega].value(j, p) *
                                  fe_values_omega[Potential_omega].value(i, p) *
                                  fe_values_omega.JxW(p) * 1 /
                                  (n_tr * n_ftrial) * 1 / (n_te * n_ftest) *
                                  C_avag * weight;
                            }
                          }
                          constraints.distribute_local_to_global(
                              v_u_matrix_coupling, local_dof_indices_omega,
                              local_dof_indices_omega, system_matrix);

                          // --------------------------cell ende
                          // --------------------
                        }
                      }
                    }
                }
              }

#endif
            }
        }
        // std::cout<<std::endl;
      }
    }
  }

#if USE_MPI
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
#endif
  // std::cout << "ende coupling loop" << std::endl;

  // std::cout << "set ii " << std::endl;

  for (unsigned int i = 0; i < dof_handler.n_dofs(); i++) // dof_table.size()
  {
    // if(dof_table[i].first.first == 1 || dof_table[i].first.first == 3)
    {
      if (system_matrix.el(i, i) == 0) {

        system_matrix.set(i, i, 1);
      }
    }
  }
}

template <int dim, int dim_omega>
template <int _dim>
void LDGPoissonProblem<dim, dim_omega>::dof_omega_to_Omega(
    const DoFHandler<_dim> &dof_handler,
    std::vector<types::global_dof_index> &local_dof_indices_omega) {
  const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
  for (unsigned int i = 0; i < local_dof_indices_omega.size(); ++i) {
    const unsigned int base_i =
        dof_handler.get_fe().system_to_base_index(i).first.first;

    local_dof_indices_omega[i] =
        base_i == 0 ? local_dof_indices_omega[i] + start_VectorField_omega
                    : local_dof_indices_omega[i] - dofs_per_component[0] +
                          start_Potential_omega;
  }
}

template <int dim, int dim_omega>
template <int _dim>
void LDGPoissonProblem<dim, dim_omega>::assemble_cell_terms(
    const FEValues<_dim> &cell_fe, FullMatrix<double> &cell_matrix,
    Vector<double> &cell_vector,
    const TensorFunction<2, _dim> &_K_inverse_function,
    const Function<_dim> &_rhs_function,
    const FEValuesExtractors::Vector &VectorField,
    const FEValuesExtractors::Scalar &Potential) {
  const unsigned int dofs_per_cell = cell_fe.dofs_per_cell;
  const unsigned int n_q_points = cell_fe.n_quadrature_points;

  std::vector<double> rhs_values(n_q_points);
  std::vector<Tensor<2, _dim>> K_inverse_values(n_q_points);

  _rhs_function.value_list(cell_fe.get_quadrature_points(), rhs_values);
  _K_inverse_function.value_list(cell_fe.get_quadrature_points(),
                                 K_inverse_values);

  for (unsigned int q = 0; q < n_q_points; ++q) {
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      const Tensor<1, _dim> psi_i_field = cell_fe[VectorField].value(i, q);
      const double div_psi_i_field = cell_fe[VectorField].divergence(i, q);
      const double psi_i_potential = cell_fe[Potential].value(i, q);
      const Tensor<1, _dim> grad_psi_i_potential =
          cell_fe[Potential].gradient(i, q);

      for (unsigned int j = 0; j < dofs_per_cell; ++j) {
        const Tensor<1, _dim> psi_j_field = cell_fe[VectorField].value(j, q);
        const double psi_j_potential = cell_fe[Potential].value(j, q);

        cell_matrix(i, j) +=
            ((psi_i_field * K_inverse_values[q] * psi_j_field) -
             (div_psi_i_field * psi_j_potential) -
             (grad_psi_i_potential * psi_j_field)) *
            cell_fe.JxW(q);
      }

      cell_vector(i) += psi_i_potential * rhs_values[q] * cell_fe.JxW(q);
    }
  }
}

template <int dim, int dim_omega>
template <int _dim>
void LDGPoissonProblem<dim, dim_omega>::assemble_Dirichlet_boundary_terms(
    const FEFaceValues<_dim> &face_fe, FullMatrix<double> &local_matrix,
    Vector<double> &local_vector, const double &h,
    const Function<_dim> &Dirichlet_bc_function,
    const FEValuesExtractors::Vector VectorField,
    const FEValuesExtractors::Scalar Potential) {
  const unsigned int dofs_per_cell = face_fe.dofs_per_cell;
  const unsigned int n_q_points = face_fe.n_quadrature_points;

  std::vector<double> Dirichlet_bc_values(n_q_points);

  Dirichlet_bc_function.value_list(face_fe.get_quadrature_points(),
                                   Dirichlet_bc_values);

  for (unsigned int q = 0; q < n_q_points; ++q) {
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      const Tensor<1, _dim> psi_i_field = face_fe[VectorField].value(i, q);
      const double psi_i_potential = face_fe[Potential].value(i, q);

      for (unsigned int j = 0; j < dofs_per_cell; ++j) {
        const Tensor<1, _dim> psi_j_field = face_fe[VectorField].value(j, q);
        const double psi_j_potential = face_fe[Potential].value(j, q);

        local_matrix(i, j) += psi_i_potential *
                              (face_fe.normal_vector(q) * psi_j_field +
                               (penalty / h) * psi_j_potential) *
                              face_fe.JxW(q);
      }

      local_vector(i) += (-1.0 * psi_i_field * face_fe.normal_vector(q) +
                          (penalty / h) * psi_i_potential) *
                         Dirichlet_bc_values[q] * face_fe.JxW(q);
    }
  }
}

template <int dim, int dim_omega>
template <int _dim>
void LDGPoissonProblem<dim, dim_omega>::assemble_Neumann_boundary_terms(
    const FEFaceValues<_dim> &face_fe, FullMatrix<double> &local_matrix,
    Vector<double> &local_vector, const Function<_dim> &Neumann_bc_function) {
  const unsigned int dofs_per_cell = face_fe.dofs_per_cell;
  const unsigned int n_q_points = face_fe.n_quadrature_points;

  std::vector<double> Neumann_bc_values(n_q_points);

  Neumann_bc_function.value_list(face_fe.get_quadrature_points(),
                                 Neumann_bc_values);

  for (unsigned int q = 0; q < n_q_points; ++q) {
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      const Tensor<1, dim> psi_i_field = face_fe[VectorField].value(i, q);
      const double psi_i_potential = face_fe[Potential].value(i, q);

      for (unsigned int j = 0; j < dofs_per_cell; ++j) {

        const double psi_j_potential = face_fe[Potential].value(j, q);

        local_matrix(i, j) += psi_i_field * face_fe.normal_vector(q) *
                              psi_j_potential * face_fe.JxW(q);
      }

      local_vector(i) +=
          -psi_i_potential * Neumann_bc_values[q] * face_fe.JxW(q);
    }
  }
}

template <int dim, int dim_omega>
template <int _dim>
void LDGPoissonProblem<dim, dim_omega>::assemble_flux_terms(
    const FEFaceValuesBase<_dim> &fe_face_values,
    const FEFaceValuesBase<_dim> &fe_neighbor_face_values,
    FullMatrix<double> &vi_ui_matrix, FullMatrix<double> &vi_ue_matrix,
    FullMatrix<double> &ve_ui_matrix, FullMatrix<double> &ve_ue_matrix,
    const double &h, const FEValuesExtractors::Vector VectorField,
    const FEValuesExtractors::Scalar Potential) {
  const unsigned int n_face_points = fe_face_values.n_quadrature_points;
  const unsigned int dofs_this_cell = fe_face_values.dofs_per_cell;
  const unsigned int dofs_neighbor_cell = fe_neighbor_face_values.dofs_per_cell;

  for (unsigned int q = 0; q < n_face_points; ++q) {
    for (unsigned int i = 0; i < dofs_this_cell; ++i) {
      const Tensor<1, _dim> psi_i_field_minus =
          fe_face_values[VectorField].value(i, q);
      const double psi_i_potential_minus =
          fe_face_values[Potential].value(i, q);

      for (unsigned int j = 0; j < dofs_this_cell; ++j) {
        const Tensor<1, _dim> psi_j_field_minus =
            fe_face_values[VectorField].value(j, q);
        const double psi_j_potential_minus =
            fe_face_values[Potential].value(j, q);

        vi_ui_matrix(i, j) +=
            (0.5 * (psi_i_field_minus * fe_face_values.normal_vector(q) *
                        psi_j_potential_minus +
                    psi_i_potential_minus * fe_face_values.normal_vector(q) *
                        psi_j_field_minus) +
             (penalty / h) * psi_j_potential_minus * psi_i_potential_minus

             ) *
            fe_face_values.JxW(q);
      }

      for (unsigned int j = 0; j < dofs_neighbor_cell; ++j) {
        const Tensor<1, _dim> psi_j_field_plus =
            fe_neighbor_face_values[VectorField].value(j, q);
        const double psi_j_potential_plus =
            fe_neighbor_face_values[Potential].value(j, q);

        // We compute the flux matrix where the test function is
        // from the interior of this elements face and solution
        // function is taken from the exterior.
        vi_ue_matrix(i, j) +=
            (0.5 * (psi_i_field_minus * fe_face_values.normal_vector(q) *
                        psi_j_potential_plus +
                    psi_i_potential_minus * fe_face_values.normal_vector(q) *
                        psi_j_field_plus) -
             (penalty / h) * psi_i_potential_minus * psi_j_potential_plus) *
            fe_face_values.JxW(q);
      }
    }

    for (unsigned int i = 0; i < dofs_neighbor_cell; ++i) {
      const Tensor<1, _dim> psi_i_field_plus =
          fe_neighbor_face_values[VectorField].value(i, q);
      const double psi_i_potential_plus =
          fe_neighbor_face_values[Potential].value(i, q);

      for (unsigned int j = 0; j < dofs_this_cell; ++j) {
        const Tensor<1, _dim> psi_j_field_minus =
            fe_face_values[VectorField].value(j, q);
        const double psi_j_potential_minus =
            fe_face_values[Potential].value(j, q);

        // We compute the flux matrix where the test function is
        // from the exterior of this elements face and solution
        // function is taken from the interior.
        ve_ui_matrix(i, j) += 0;
        ve_ui_matrix(i, j) +=

            (-0.5 * (psi_i_field_plus * fe_face_values.normal_vector(q) *
                         psi_j_potential_minus +
                     psi_i_potential_plus * fe_face_values.normal_vector(q) *
                         psi_j_field_minus) -
             (penalty / h) * psi_i_potential_plus * psi_j_potential_minus) *
            fe_face_values.JxW(q);
      }

      for (unsigned int j = 0; j < dofs_neighbor_cell; ++j) {
        const Tensor<1, _dim> psi_j_field_plus =
            fe_neighbor_face_values[VectorField].value(j, q);
        const double psi_j_potential_plus =
            fe_neighbor_face_values[Potential].value(j, q);

        // And lastly we compute the flux matrix where the test
        // function and solution function are taken from the exterior
        // cell to this face.
        ve_ue_matrix(i, j) += 0;
        ve_ue_matrix(i, j) +=

            (-0.5 * (psi_i_field_plus * fe_face_values.normal_vector(q) *
                         psi_j_potential_plus +
                     psi_i_potential_plus * fe_face_values.normal_vector(q) *
                         psi_j_field_plus) +
             (penalty / h) * psi_i_potential_plus * psi_j_potential_plus) *
            fe_face_values.JxW(q);
      }
    }
  }
}

// @sect4{distribute_local_flux_to_global}
// In this function we use the ConstraintMatrix to distribute
// the local flux matrices to the global system matrix.
// Since I have to do this twice in assembling the
// system matrix, I made function to do it rather than have
// repeated code.
// We remark that the reader take special note of
// the which matrices we are distributing and the order
// in which we pass the dof indices vectors. In distributing
// the first matrix, i.e. <code>vi_ui_matrix</code>, we are
// taking the test function and solution function values from
// the interior of this cell and therefore only need the
// <code>local_dof_indices</code> since it contains the dof
// indices to this cell. When we distribute the second matrix,
// <code>vi_ue_matrix</code>, the test function is taken
// form the inteior of
// this cell while the solution function is taken from the
// exterior, i.e. the neighbor cell.  Notice that the order
// degrees of freedom index vectors matrch this pattern: first
// the <code>local_dof_indices</code> which is local to
// this cell and then
// the <code>local_neighbor_dof_indices</code> which is
// local to the neighbor's
// cell.  The order in which we pass the dof indices for the
// matrices is paramount to constructing our global system
// matrix properly.  The ordering of the last two matrices
// follow the same logic as the first two we discussed.
template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::distribute_local_flux_to_global(
    FullMatrix<double> &vi_ui_matrix, FullMatrix<double> &vi_ue_matrix,
    FullMatrix<double> &ve_ui_matrix, FullMatrix<double> &ve_ue_matrix,
    const std::vector<types::global_dof_index> &local_dof_indices,
    const std::vector<types::global_dof_index> &local_neighbor_dof_indices) {

  constraints.distribute_local_to_global(vi_ui_matrix, local_dof_indices,
                                         system_matrix);

  constraints.distribute_local_to_global(vi_ue_matrix, local_dof_indices,
                                         local_neighbor_dof_indices,
                                         system_matrix);

  constraints.distribute_local_to_global(ve_ui_matrix,
                                         local_neighbor_dof_indices,
                                         local_dof_indices, system_matrix);

  constraints.distribute_local_to_global(
      ve_ue_matrix, local_neighbor_dof_indices, system_matrix);
}

template <int dim, int dim_omega>
std::array<double, 4>
LDGPoissonProblem<dim, dim_omega>::compute_errors() const {
  /*std::cout << "compute_errors "
            << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << std::endl;*/
  double potential_l2_error, vectorfield_l2_error, potential_l2_error_omega,
      vectorfield_l2_error_omega;
  //  double global_potential_l2_error,global_vectorfield_l2_error,
  double global_potential_l2_error_omega, global_vectorfield_l2_error_omega;
  // if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
  {
    const ComponentSelectFunction<dim> potential_mask(dim + 1,
                                                      dim + dim_omega + 2);
    const ComponentSelectFunction<dim> vectorfield_mask(std::make_pair(0, dim),
                                                        dim + dim_omega + 2);
    double alpha = 0.5;
    const DistanceWeight<dim> distance_weight(alpha, radius); //, radius

    const ProductFunction<dim> connected_function_potential(potential_mask,
                                                            distance_weight);
    const ProductFunction<dim> connected_function_vectorfield(vectorfield_mask,
                                                              distance_weight);

    Vector<double> cellwise_errors(triangulation.n_active_cells());
    pcout << "triangulation.n_active_cells() " << triangulation.n_active_cells()
          << " dof_handler.n_dofs() " << dof_handler.n_dofs()
          << " dof_handler.n_locally_owned_dofs() "
          << dof_handler.n_locally_owned_dofs() << " solution size "
          << solution.size() << " mpi "
          << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << std::endl;

    const QTrapezoid<1> q_trapez;
    const QIterated<dim> quadrature(q_trapez, degree + 2);

    VectorTools::integrate_difference(
        dof_handler, solution, true_solution, cellwise_errors, quadrature,
        VectorTools::L2_norm, &connected_function_potential); //
    /*  std::cout<<"cellwise_error.size() "<<cellwise_errors.size()<<std::endl;
     for (unsigned int i = 0; i < cellwise_errors.size(); i++)
      std::cout << cellwise_errors[i] << " "<<std::endl;

      std::cout<<"--------------------"<<std::endl; */
    /*#if USE_MPI
        cellwise_errors.compress(VectorOperation::add); // TODO scauen was es
    noc  // fpr #endif
    */
    potential_l2_error = VectorTools::compute_global_error(
        triangulation, cellwise_errors, VectorTools::L2_norm);
    // std::cout<<"mpi  "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<< "
    // potential_l2_error "<<potential_l2_error<<std::endl;
    //  vectorfield Omega
    VectorTools::integrate_difference(
        dof_handler, solution, true_solution, cellwise_errors, quadrature,
        VectorTools::L2_norm, &connected_function_vectorfield);

    /*
    #if USE_MPI
        cellwise_errors.compress(VectorOperation::add); // TODO scauen was es
    noc #endif
    */
    vectorfield_l2_error = VectorTools::compute_global_error(
        triangulation, cellwise_errors, VectorTools::L2_norm);

    // std::cout<<"mpi  "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<< "
    // vectorfield_l2_error "<<vectorfield_l2_error<<std::endl;
    //-------------omega----------------------------------
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      // if (locally_owned_dofs.is_element(dof_index)) TODO allgemeine MPI
      // sachen, auch wenn alle andere 0 sind
      const ComponentSelectFunction<dim_omega> potential_mask_omega(
          dim_omega, dim_omega + 1);
      const ComponentSelectFunction<dim_omega> vectorfield_mask_omega(
          std::make_pair(0, dim_omega), dim_omega + 1);
      Vector<double> cellwise_errors_omega(
          triangulation_omega.n_active_cells());
      /*std::cout << "triangulation_omega.n_active_cells() " <<
         triangulation_omega.n_active_cells()
                << " dof_handler_omega.n_dofs() " << dof_handler_omega.n_dofs()
               << " dof_handler_omega.n_locally_owned_dofs()
         "<<dof_handler_omega.n_locally_owned_dofs()
                <<" solution_omega.size() "<<solution_omega.size()
                <<" mpi
         "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<std::endl;*/
      const QTrapezoid<1> q_trapez_omega;
      const QIterated<dim_omega> quadrature_omega(q_trapez_omega, degree + 2);

      for (unsigned int i = 0; i < solution_omega.size(); i++)
        std::cout << solution_omega[i] << " ";
      std::cout << std::endl;

      VectorTools::integrate_difference(
          dof_handler_omega, solution_omega, true_solution_omega,
          cellwise_errors_omega, quadrature_omega, VectorTools::L2_norm,
          &potential_mask_omega);

      potential_l2_error_omega = VectorTools::compute_global_error(
          triangulation_omega, cellwise_errors_omega, VectorTools::L2_norm);

      /*
       std::cout<<"cellwise_errors_omega_potential ";
         for (unsigned int i = 0; i < cellwise_errors_omega.size(); i++)
          std::cout << cellwise_errors_omega[i] << " ";
           std::cout << std::endl;
      */

      VectorTools::integrate_difference(
          dof_handler_omega, solution_omega, true_solution_omega,
          cellwise_errors_omega, quadrature_omega, VectorTools::L2_norm,
          &vectorfield_mask_omega);

      vectorfield_l2_error_omega = VectorTools::compute_global_error(
          triangulation_omega, cellwise_errors_omega, VectorTools::L2_norm);

    } else {
      potential_l2_error_omega = 0;
      vectorfield_l2_error_omega = 0;
    }
    std::cout << "mpi " << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)
              << " u " << potential_l2_error_omega << " q "
              << vectorfield_l2_error_omega << std::endl;
    global_potential_l2_error_omega = Utilities::MPI::sum(
        std::pow(potential_l2_error_omega, 2), MPI_COMM_WORLD);
    global_vectorfield_l2_error_omega = Utilities::MPI::sum(
        std::pow(vectorfield_l2_error_omega, 2), MPI_COMM_WORLD);
  }

  return std::array<double, 4>{{potential_l2_error, vectorfield_l2_error,
                                std::sqrt(global_potential_l2_error_omega),
                                std::sqrt(global_vectorfield_l2_error_omega)}};
}

template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::solve() {
  TimerOutput::Scope t(computing_timer, "solve");
  pcout << "Solving linear system... ";
#if USE_MPI

  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);
  const IndexSet &locally_owned_dofs = dof_handler.locally_owned_dofs();

  TrilinosWrappers::MPI::Vector completely_distributed_solution(
      locally_owned_dofs, MPI_COMM_WORLD);

  SolverControl solver_control(dof_handler.n_dofs(), 1e-12);
  TrilinosWrappers::SolverGMRES solver(solver_control);

  // TrilinosWrappers::PreconditionIdentity preconditioner;
  TrilinosWrappers::PreconditionILU preconditioner;
  // TrilinosWrappers::PreconditionBlockJacobi preconditioner;
  // TrilinosWrappers::PreconditionBlockSSOR preconditioner;
  // IdentityMatrix preconditioner;
  // TrilinosWrappers::PreconditionAMG preconditioner;
  // TrilinosWrappers::PreconditionAMG::AdditionalData data;
  // TrilinosWrappers::PreconditionIdentity::AdditionalData data;
  TrilinosWrappers::PreconditionILU::AdditionalData data;
  // TrilinosWrappers::PreconditionBlockJacobi::AdditionalData data;
  // TrilinosWrappers::PreconditionBlockSSOR::AdditionalData data;
  preconditioner.initialize(system_matrix, data);

  solver.solve(system_matrix, completely_distributed_solution, system_rhs,
               preconditioner);

  pcout << "   Solved in " << solver_control.last_step() << " iterations."
        << std::endl;

  constraints.distribute(completely_distributed_solution);

  solution = completely_distributed_solution;

  // solution = completely_distributed_solution;
#else
  Timer timer;

  SparseDirectUMFPACK A_direct;

  solution = system_rhs;
  A_direct.solve(system_matrix, solution);

  /*  const unsigned int max_iterations = solution.size();
    SolverControl      solver_control(max_iterations);
    SolverCG<>         solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs,
    PreconditionIdentity());*/

  timer.stop();
  std::cout << "done (" << timer.cpu_time() << "s)" << std::endl;

#endif

  const std::vector<types::global_dof_index> dofs_per_component_omega =
      DoFTools::count_dofs_per_fe_component(dof_handler_omega);
  // std::cout<<"nof compoent "<<dofs_per_component_omega.size()<<std::endl;

  solution_omega.reinit(dofs_per_component_omega[0] +
                        dofs_per_component_omega[1]);
  for (unsigned int i = 0; i < dofs_per_component_omega[0]; i++) {
    types::global_dof_index dof_index = start_VectorField_omega + i;
#if USE_MPI
    if (locally_owned_dofs.is_element(dof_index))
#endif
      solution_omega[i] = solution[dof_index];
  }

  for (unsigned int i = 0; i < dofs_per_component_omega[1]; i++) {
    types::global_dof_index dof_index = start_Potential_omega + i;
#if USE_MPI
    if (locally_owned_dofs.is_element(dof_index))
#endif
      solution_omega[dofs_per_component_omega[0] + i] = solution[dof_index];
  }

#if USE_MPI
  // solution_omega.compress(VectorOperation::add);//TODO scauen was es noc fpr
#endif
}

template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::output_results() const {
  std::cout << "Output_result" << std::endl;
  std::vector<std::string> solution_names;
  switch (dim) {
  case 1:
    solution_names.push_back("Q_x");
    solution_names.push_back("q");
    solution_names.push_back("U");
    solution_names.push_back("u");

    break;

  case 2:
    solution_names.push_back("Q_x");
    solution_names.push_back("Q_y");
    solution_names.push_back("q");
    solution_names.push_back("U");
    solution_names.push_back("u");
    break;

  case 3:
    solution_names.push_back("Q_x");
    solution_names.push_back("Q_y");
    solution_names.push_back("Q_z");
    solution_names.push_back("q");
    solution_names.push_back("U");
    solution_names.push_back("u");
    break;

  default:
    Assert(false, ExcNotImplemented());
  }

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution,
                           solution_names); //, DataOut<dim>::type_cell_data

  data_out.build_patches(degree);

  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);

  // ------analytical solution--------
  std::cout << "analytical solution" << std::endl;
  DoFHandler<dim> dof_handler_Lag(triangulation);
  FESystem<dim> fe_Lag(FESystem<dim>(FE_DGQ<dim>(degree), dim),
                       FE_DGQ<dim>(degree), FE_DGQ<dim>(degree),
                       FE_DGQ<dim>(degree));
  dof_handler_Lag.distribute_dofs(fe_Lag);
  TrilinosWrappers::MPI::Vector solution_const;
  solution_const.reinit(dof_handler_Lag.locally_owned_dofs(), MPI_COMM_WORLD);

  VectorTools::interpolate(dof_handler_Lag, true_solution, solution_const);

  DataOut<dim> data_out_const;
  data_out_const.attach_dof_handler(dof_handler_Lag);
  data_out_const.add_data_vector(solution_const, solution_names); //

  data_out_const.build_patches(degree);

  std::ofstream output_const("solution_const.vtu");
  data_out_const.write_vtu(output_const);

  //-----omega-----------
  std::cout << "omega solution" << std::endl;
  std::vector<std::string> solution_names_omega;
  solution_names_omega.emplace_back("q");
  solution_names_omega.emplace_back("u");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation_omega;
  interpretation_omega.push_back(
      DataComponentInterpretation::component_is_scalar);
  interpretation_omega.push_back(
      DataComponentInterpretation::component_is_scalar);

  DataOut<dim_omega> data_out_omega;
  data_out_omega.add_data_vector(dof_handler_omega, solution_omega,
                                 solution_names_omega, interpretation_omega);

  data_out_omega.build_patches(degree);

  std::ofstream output_omega("solution_omega.vtu");
  data_out_omega.write_vtu(output_omega);
}

template <int dim, int dim_omega>
std::array<double, 4> LDGPoissonProblem<dim, dim_omega>::run() {
  pcout << "n_refine " << n_refine << "  degree " << degree << std::endl;

  penalty = 5;
  make_grid();
  make_dofs();
  assemble_system();
  solve();
  //  output_results();
  std::array<double, 4> results_array = compute_errors();
  return results_array;
}

int main(int argc, char *argv[]) {
  std::cout << "USE_MPI " << USE_MPI << std::endl;
#if USE_MPI
  deallog.depth_console(0);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv,
      numbers::invalid_unsigned_int); //

  int num_processes = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  // Get the rank of the process
  int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  // Print the number of processes and the rank of the current process
  if (rank == 0) {
    std::cout << "Number of MPI processes: " << num_processes << std::endl;
  }

  std::cout << "This is MPI process " << rank << std::endl;

#endif

  // std::cout << "dimension_Omega " << dimension_Omega << " solution "
  //          << constructed_solution << std::endl;

  /*  LDGPoissonProblem<dimension_Omega, 1> LDGPoissonCoupled_s(1,3);
    std::array<double, 4> arr = LDGPoissonCoupled_s.run();
    std::cout << rank << " Result_ende: U " << arr[0] << " Q " << arr[1] << " u
    "
              << arr[2] << " q " << arr[3] << std::endl;
    return 0;
  */
  std::cout << "dimension_Omega " << dimension_Omega << std::endl;
  double radii[2] = { 0.1, 0.01 };
  bool lumpedAverages[2] = {true, false};
  std::vector<std::array<double, 4>> result_scenario;
  std::vector<std::string> scenario_names;
  for (unsigned int rad = 0; rad < 2; rad++) {
    for (unsigned int LA = 0; LA < 2; LA++) {
      
      std::string LA_string = lumpedAverages[LA] ? "true" : "false";
      std::string radius_string = std::to_string(radii[rad]);
      std::string name= "_LA_" + LA_string + "_rad_" + radius_string;
      scenario_names.push_back(name) ;


      Parameters parameters;
      parameters.radius =radii[rad];
      parameters.lumpedAverage = lumpedAverages[LA];
      const unsigned int p_degree[2] = {1,2};
      constexpr unsigned int p_degree_size =
          sizeof(p_degree) / sizeof(p_degree[0]);
      const unsigned int refinement[3] = {1,2,3};
      constexpr unsigned int refinement_size =
          sizeof(refinement) / sizeof(refinement[0]);

      std::array<double, 4> results[p_degree_size][refinement_size];

      std::vector<std::string> solution_names = {"U_Omega", "Q_Omega",
                                                 "u_omega", "q_omega"};
      for (unsigned int r = 0; r < refinement_size; r++) {
        for (unsigned int p = 0; p < p_degree_size; p++) {
          LDGPoissonProblem<dimension_Omega, 1> LDGPoissonCoupled =
              LDGPoissonProblem<dimension_Omega, 1>(p_degree[p], refinement[r],
                                                    parameters);
          std::array<double, 4> arr = LDGPoissonCoupled.run();
          std::cout << rank << " Result_ende: U " << arr[0] << " Q " << arr[1]
                    << " u " << arr[2] << " q " << arr[3] << std::endl;
          results[p][r] = arr;
        }
      }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    // std::cout << "--------" << std::endl;
    std::ofstream myfile;
    std::ofstream csvfile;
#if COUPLED
    std::string filename = "convergence_results_coupled" + name;
    myfile.open(filename +".txt");
    csvfile.open(filename +".csv");
#else
    std::string filename = "convergence_results_uncoupled" + name;
    myfile.open(filename +".txt");
    csvfile.open(filename +".csv");
#endif
    for (unsigned int f = 0; f < solution_names.size(); f++) {
      myfile << solution_names[f] << "\n";
      myfile << "refinement/p_degree, ";

      csvfile << solution_names[f] << "\n";
      csvfile << "refinement/p_degree;";

      std::cout << solution_names[f] << "\n";
      std::cout << "refinement/p_degree;";
      for (unsigned int p = 0; p < p_degree_size; p++) {
        myfile << p_degree[p] << ",";
        csvfile << p_degree[p] << ";";
        std::cout << p_degree[p] << ";";
      }
      myfile << "\n";
      csvfile << "\n";
      std::cout << "\n";
      for (unsigned int r = 0; r < refinement_size; r++) {
        myfile << refinement[r] << ",";
        csvfile << refinement[r] << ";";
        std::cout << refinement[r] << ";";
        for (unsigned int p = 0; p < p_degree_size; p++) {
          const double error = results[p][r][f];

          myfile << error;
          csvfile << error;
          std::cout << error;
          if (r != 0) {
            const double rate =
                std::log2(results[p][r - 1][f] / results[p][r][f]);
            myfile << " (" << rate << ")";
            csvfile << " (" << rate << ")";
            std::cout << " (" << rate << ")";
          }

          myfile << ",";
          if (p < p_degree_size - 1)
            csvfile << ";";
          std::cout << ";";
        }
        myfile << std::endl;
        csvfile << "\n";
        std::cout << std::endl;
      }
      myfile << std::endl << std::endl;
      csvfile << "\n\n";
      std::cout << std::endl << std::endl;
    }

    myfile.close();
    csvfile.close();
  }
    }
  }
  return 0;
}
