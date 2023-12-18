#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <cmath>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/matrix_out.h>
#include <deal.II/numerics/data_out.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#include <deal.II/base/logstream.h>

using namespace dealii;

constexpr unsigned int nof_scalar_fields{2};
constexpr unsigned int dimension_omega{3};
constexpr unsigned int refinement{4};
constexpr unsigned int p_degree{1};

template <int dim_omega, int dim_sigma> class CouplingLaplace {
public:
  CouplingLaplace();
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<dim_omega> triangulation;
  FESystem<dim_omega> fe;
  DoFHandler<dim_omega> dof_handler;

  Triangulation<dim_sigma> triangulation_sigma;
  FESystem<dim_sigma> fe_sigma; // FE_Q
  DoFHandler<dim_sigma> dof_handler_sigma;

  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
  double distance_between_grid_points;

  double g = 1.0;
};
namespace PrescribedSolution {
constexpr double boundary1 = -0.5;
constexpr double boundary2 = 0.5;

template <int dim_omega> bool isOnSigma(Point<dim_omega> p) {
  bool return_value = true;
  if (p[0] >= boundary1 && p[0] <= boundary2)
    return_value = return_value && true;
  else
    return_value = return_value && false;
  for (unsigned int i = 1; i < dim_omega; i++) {
    if (p[i] == 0)
      return_value = return_value && true;
    else
      return_value = return_value && false;
  }
  return return_value;
}
template <int dim_omega> bool isOnSigma_boundary(Point<dim_omega> p) {
  bool return_value = true;
  if (p[0] == boundary1 || p[0] == boundary2)
    return_value = return_value && true;
  else
    return_value = return_value && false;
  for (unsigned int i = 1; i < dim_omega; i++) {
    if (p[i] == 0)
      return_value = return_value && true;
    else
      return_value = return_value && false;
  }
  return return_value;
}
template <int dim_omega> bool isMidPoint(Point<dim_omega> p) {
  bool return_value = true;
  for (unsigned int i = 0; i < dim_omega; i++) {
    if (p[i] == 0)
      return_value = return_value && true;
    else
      return_value = return_value && false;
  }
  return return_value;
}
template <int dim_omega> class F_Omega : public Function<dim_omega> {
public:
  F_Omega() : Function<dim_omega>(nof_scalar_fields) {}

  virtual double value(const Point<dim_omega> &p,
                       const unsigned int component = 0) const override;
};

template <int dim_sigma> class F_Sigma : public Function<dim_sigma> {
public:
  F_Sigma() : Function<dim_sigma>() {}

  virtual double value(const Point<dim_sigma> &p,
                       const unsigned int component = 0) const override;
};

template <int dim_omega>
class BoundaryValues_Omega : public Function<dim_omega> {
public:
  BoundaryValues_Omega() : Function<dim_omega>(nof_scalar_fields) {}

  virtual double value(const Point<dim_omega> &p,
                       const unsigned int component = 0) const override;
};

template <int dim_omega>
double F_Omega<dim_omega>::value(const Point<dim_omega> & /*p*/,
                                 const unsigned int /*component*/) const {
  return 0;
}

template <int dim_sigma>
double F_Sigma<dim_sigma>::value(const Point<dim_sigma> & /*p*/,
                                 const unsigned int /*component*/) const {
  double return_value = 0.0;
  /*for (unsigned int i = 0; i < dim_sigma; ++i)
    return_value += 4.0 * std::pow(p(i), 4.0);*/

  return return_value;
}

template <int dim_omega>
double
BoundaryValues_Omega<dim_omega>::value(const Point<dim_omega> & /*p*/,
                                       const unsigned int component) const {
  if (component == 0)
    return 0.0;
  else
    return 0.0; //
}
} // namespace PrescribedSolution

template <int dim_omega, int dim_sigma>
CouplingLaplace<dim_omega, dim_sigma>::CouplingLaplace()
    : fe(FE_Q<dim_omega>(p_degree) ^ nof_scalar_fields),
      dof_handler(triangulation),
      fe_sigma(FE_Q<dim_sigma>(p_degree) ^ nof_scalar_fields),
      dof_handler_sigma(triangulation_sigma) {}

template <int dim_omega, int dim_sigma>
void CouplingLaplace<dim_omega, dim_sigma>::make_grid() {
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(refinement);

  if (dim_omega == 2) {
    std::ofstream out("grid-1.svg");
    GridOut grid_out;
    grid_out.write_svg(triangulation, out);
    std::cout << "Grid written to grid-1.svg" << std::endl;
  }

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;

  distance_between_grid_points = 2.0 / std::pow(2, 4);
  std::cout << "distance_between_grid_points " << distance_between_grid_points
            << std::endl;
}

template <int dim_omega, int dim_sigma>
void CouplingLaplace<dim_omega, dim_sigma>::setup_system() {
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}

template <int dim_omega, int dim_sigma>
void CouplingLaplace<dim_omega, dim_sigma>::assemble_system() {
  QGauss<dim_omega> quadrature_formula(fe.degree + 1);

  PrescribedSolution::F_Sigma<dim_sigma> f_sigma;
  PrescribedSolution::F_Omega<dim_omega> f_omega;

  FEValues<dim_omega> fe_values(fe, quadrature_formula,
                                update_values | update_gradients |
                                    update_quadrature_points |
                                    update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  std::cout << "fe.n_dofs_per_face() " << fe.n_dofs_per_face()
            << " dofs_per_cell  " << dofs_per_cell << " fe.n_dofs_per_line() "
            << fe.n_dofs_per_line() << " fe.n_dofs_per_vertex() "
            << fe.n_dofs_per_vertex() << std::endl;

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::array<std::vector<types::global_dof_index>, nof_scalar_fields>
      line_dof_indices;

  const FEValuesExtractors::Scalar concentration_u(0);
  const FEValuesExtractors::Scalar concentration_mu(1);

  std::vector<Point<dim_omega>> support_points;
  support_points.resize(dof_handler.n_dofs());
  DoFTools::map_dofs_to_support_points(fe_values.get_mapping(), dof_handler,
                                       support_points);

  std::vector<
      std::array<std::vector<types::global_dof_index>, nof_scalar_fields>>
      dof_indices_sigma_per_cells_comp;
  std::vector<std::vector<types::global_dof_index>> dof_indices_sigma_per_cells;
  std::array<std::set<types::global_dof_index>, nof_scalar_fields>
      dof_indices_boundary_sigma;
  std::array<std::set<types::global_dof_index>, nof_scalar_fields>
      dof_indices_per_component;
  std::array<std::set<types::global_dof_index>, nof_scalar_fields>
      dof_indices_sigma;
  std::array<types::global_dof_index, nof_scalar_fields> dof_index_midpoint;

  std::vector<typename DoFHandler<dim_omega>::face_iterator> faces;
  std::vector<typename DoFHandler<dim_omega>::active_line_iterator> lines;


  for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);

    /*for (const typename DoFHandler<dim_omega>::face_iterator &face :
         cell->face_iterators()) {
      if (std::find(faces.begin(), faces.end(), face) == faces.end())
        faces.push_back(face);
      else
        continue;

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_face);
      face->get_dof_indices(local_dof_indices);
*/

     for (unsigned int l = 0; l < cell->n_lines(); l++) {

      const typename DoFHandler<dim_omega>::active_line_iterator line =
          cell->line(l);

      if (std::find(lines.begin(), lines.end(), line) == lines.end())
        lines.push_back(line);
      else
        continue;  



      std::vector<types::global_dof_index> local_dof_indices(
          fe.n_dofs_per_line() + fe.n_dofs_per_vertex() * 2);
      line->get_dof_indices(local_dof_indices);




      std::array<std::vector<types::global_dof_index>, nof_scalar_fields>
          dof_indices_sigma_cell;
      std::vector<types::global_dof_index> dof_indices_sigma_cell_v2;

      bool push = true;
      for (unsigned int i = 0; i < local_dof_indices.size(); i++) {
        unsigned int index = local_dof_indices[i];
        Point<dim_omega> p = support_points[index];
        const unsigned int component_i = fe.system_to_component_index(i).first;

        dof_indices_per_component[component_i].insert(local_dof_indices[i]);
        if (PrescribedSolution::isOnSigma<dim_omega>(p)) {
          dof_indices_sigma_cell[component_i].push_back(local_dof_indices[i]);
          dof_indices_sigma_cell_v2.push_back(local_dof_indices[i]);

          dof_indices_sigma[component_i].insert(local_dof_indices[i]);
          push = push && true;
        } else
          push = push && false;
        if (PrescribedSolution::isOnSigma_boundary<dim_omega>(p)) {
          dof_indices_boundary_sigma[component_i].insert(local_dof_indices[i]);
        }
        if (PrescribedSolution::isMidPoint<dim_omega>(p)) {
          dof_index_midpoint[component_i] = local_dof_indices[i];
        }
      }
      if (push) {
        dof_indices_sigma_per_cells_comp.push_back(dof_indices_sigma_cell);
        dof_indices_sigma_per_cells.push_back(dof_indices_sigma_cell_v2);
      }
    }
  }

  std::cout << "----------start loops-----------------" << std::endl;
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs = 0;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    cell->get_dof_indices(local_dof_indices);

    for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
      for (const unsigned int i : fe_values.dof_indices()) {
        // const unsigned int component_i =
        // fe.system_to_component_index(i).first;
        for (const unsigned int j : fe_values.dof_indices()) {
          // const unsigned int component_j =
          //   fe.system_to_component_index(j).first;

          cell_matrix(i, j) +=
              fe_values[concentration_u].gradient(i, q_index) // grad phi_i(x_q)
              *
              fe_values[concentration_u].gradient(j, q_index) // grad phi_j(x_q)
              * fe_values.JxW(q_index);                       // dx

          double value = -g *
                         (fe_values[concentration_mu].value(j, q_index) -
                          fe_values[concentration_u].value(j, q_index)) *
                         fe_values[concentration_u].value(i, q_index) *
                         fe_values.JxW(q_index);
          cell_matrix(i, j) += value;
        }
        const auto &x_q = fe_values.quadrature_point(q_index);

        cell_rhs(i) += fe_values.shape_value(i, q_index) *
                       f_omega.value(x_q) *    // f(x_q)
                       fe_values.JxW(q_index); // dx;
      }

      for (const unsigned int i : fe_values.dof_indices()) {
        // const unsigned int component_i =
        // fe.system_to_component_index(i).first;
        for (const unsigned int j : fe_values.dof_indices()) {
          // const unsigned int component_j =
          // fe.system_to_component_index(j).first;

          system_matrix.add(local_dof_indices[i], local_dof_indices[j],
                            cell_matrix(i, j));
        }
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }
    }
  }
  std::cout << "Start sigma ----------------" << std::endl;

  for (std::array<std::vector<types::global_dof_index>, nof_scalar_fields>
           cell : dof_indices_sigma_per_cells_comp) {
    std::cout << "cell: ";
    for (types::global_dof_index i : cell[concentration_mu.component]) {
      std::cout << "[dof: " << i << ", point: " << support_points[i] << "] ";
    }

    std::cout << std::endl;
  }
  std::cout << "----------------" << std::endl;

  for (std::vector<types::global_dof_index> cell_sigma :
       dof_indices_sigma_per_cells) {
    triangulation_sigma.clear();
    GridGenerator::hyper_cube(
        triangulation_sigma, 0,
        distance_between_grid_points); // TODO abstand zwischen vertices
    triangulation_sigma.refine_global(0);

    dof_handler_sigma.distribute_dofs(fe_sigma);

    QGauss<dim_sigma> quadrature_formula_sigma(fe_sigma.degree + 1);

    FEValues<dim_sigma> fe_values_sigma(fe_sigma, quadrature_formula_sigma,
                                        update_values | update_gradients |
                                            update_quadrature_points |
                                            update_JxW_values);

    std::vector<types::global_dof_index> local_dof_indices = cell_sigma;

    for (const auto &cell : dof_handler_sigma.active_cell_iterators()) {
      fe_values_sigma.reinit(cell);
      cell_matrix = 0;
      cell_rhs = 0;

      for (const unsigned int q_index :
           fe_values_sigma.quadrature_point_indices()) {
        for (const unsigned int i : fe_values_sigma.dof_indices()) {
          // const unsigned int component_i =
          //    fe_sigma.system_to_component_index(i).first;
          for (const unsigned int j : fe_values_sigma.dof_indices()) {
            //  const unsigned int component_j =
            //    fe_sigma.system_to_component_index(j).first;
            cell_matrix(i, j) +=
                ((fe_values_sigma[concentration_mu].gradient(i, q_index) *
                  fe_values_sigma[concentration_mu].gradient(j, q_index))) *
                fe_values_sigma.JxW(q_index);

            double value =
                -g *
                (fe_values_sigma[concentration_u].value(j, q_index) -
                 fe_values_sigma[concentration_mu].value(j, q_index)) *
                fe_values[concentration_mu].value(i, q_index) *
                fe_values.JxW(q_index);
            cell_matrix(i, j) += value;
          }
          const auto &x_q = fe_values_sigma.quadrature_point(q_index);
          cell_rhs(i) +=
              (fe_values_sigma.shape_value(i, q_index) * // phi_i(x_q)
               f_sigma.value(x_q) *                      // f(x_q)
               fe_values_sigma.JxW(q_index));            // dx
        }
      }

      for (const unsigned int i : fe_values_sigma.dof_indices()) {
        for (const unsigned int j : fe_values_sigma.dof_indices()) {
          system_matrix.add(local_dof_indices[i], local_dof_indices[j],
                            cell_matrix(i, j));
        }

        system_rhs(local_dof_indices[i]) += cell_rhs(i);
      }
    }
  }

  std::map<types::global_dof_index, double> boundary_values_sigma;
  for (types::global_dof_index i :
       dof_indices_boundary_sigma[concentration_mu.component]) {
    boundary_values_sigma.insert(std::make_pair(i, 0.0));
  }

  std::cout << "midpoint: ";
  for (types::global_dof_index i : dof_index_midpoint)
    std::cout << i << " ";
  std::cout << std::endl;

  std::map<types::global_dof_index, double> midpoint_value_sigma;
  midpoint_value_sigma.insert(
      std::make_pair(dof_index_midpoint[concentration_mu.component], 1));

  std::map<types::global_dof_index, double> dirac_extend_zero;
  for (types::global_dof_index i :
       dof_indices_per_component[concentration_mu.component]) {
    if (dof_indices_sigma[concentration_mu.component].find(i) ==
        dof_indices_sigma[concentration_mu.component].end())
      dirac_extend_zero.insert(std::make_pair(i, 0.0));
  }

  std::map<types::global_dof_index, double> boundary_values_omega;
  VectorTools::interpolate_boundary_values(
      dof_handler, 0,
      PrescribedSolution::BoundaryValues_Omega<
          dim_omega>(), // Functions::ZeroFunction<dim_omega>(dim_omega),//
      boundary_values_omega);
  MatrixTools::apply_boundary_values(boundary_values_omega, system_matrix,
                                     solution, system_rhs);
  MatrixTools::apply_boundary_values(boundary_values_sigma, system_matrix,
                                     solution, system_rhs);
  MatrixTools::apply_boundary_values(dirac_extend_zero, system_matrix, solution,
                                     system_rhs);
  MatrixTools::apply_boundary_values(midpoint_value_sigma, system_matrix,
                                     solution, system_rhs);
}
template <int dim_omega, int dim_sigma>
void CouplingLaplace<dim_omega, dim_sigma>::solve() {
  SolverControl solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
}

template <int dim_omega, int dim_sigma>
void CouplingLaplace<dim_omega, dim_sigma>::output_results() const {
  DataOut<dim_omega> data_out;

  data_out.attach_dof_handler(dof_handler);

  std::vector<std::string> solution_names;
  solution_names.emplace_back("concentration_u");
  if (nof_scalar_fields == 2)
    solution_names.emplace_back("concentration_mu");

  data_out.add_data_vector(solution, solution_names);

  data_out.build_patches();

  std::ofstream output("couplingLaplace_solution.vtk");
  data_out.write_vtk(output);

  /* MatrixOut matrix_out;
   std::ofstream out("system.vtk");
   matrix_out.build_patches(system_matrix, "system");
   matrix_out.write_vtk(out);*/
}

template <int dim_omega, int dim_sigma>
void CouplingLaplace<dim_omega, dim_sigma>::run() {
  std::cout << "Solving problem in " << dim_omega << " space dimensions."
            << std::endl;

  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}

int main() {

  CouplingLaplace<dimension_omega, 1> laplace_problem_2d;
  laplace_problem_2d.run();

  return 0;
}
