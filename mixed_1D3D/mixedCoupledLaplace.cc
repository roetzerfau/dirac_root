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
//new
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

// The only two new header files that deserve some attention are those for
// the LinearOperator and PackagedOperation classes:
#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
//new end

#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <deal.II/base/logstream.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/base/tensor_function.h>

using namespace dealii;
double g = 1;
double w = numbers::PI * 3 / 2;

constexpr unsigned int nof_scalar_fields{2};
constexpr unsigned int dimension_omega{2};
constexpr unsigned int dimension_sigma{1};
constexpr unsigned int constructed_solution{2};
constexpr unsigned int concentration_base{0};

const FEValuesExtractors::Scalar concentration_u(0);
const FEValuesExtractors::Scalar concentration_mu(1);
const FEValuesExtractors::Vector velocity_P(nof_scalar_fields);
const FEValuesExtractors::Vector velocity_p(nof_scalar_fields+dimension_omega);




template <int dim_omega, int dim_sigma> class MixedCoupledLaplace {
public:
  MixedCoupledLaplace();
  MixedCoupledLaplace(unsigned int _p_degree);
  std::array<double, 2> run(unsigned int _refinement);

  MixedCoupledLaplace(const MixedCoupledLaplace &);
  MixedCoupledLaplace &operator=(const MixedCoupledLaplace &);

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;
  std::array<double, 2> compute_errors();

  unsigned int refinement{1};
  unsigned int p_degree{1};

  Triangulation<dim_omega> triangulation;
  FESystem<dim_omega> fe;
  DoFHandler<dim_omega> dof_handler;

  Triangulation<dim_sigma> triangulation_sigma;
  FESystem<dim_sigma> fe_sigma;
  DoFHandler<dim_sigma> dof_handler_sigma;

/*
  SparsityPattern sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
*/


  BlockSparsityPattern sparsity_pattern;
  BlockSparseMatrix<double> system_matrix;

  BlockVector<double> solution;
  BlockVector<double> system_rhs;

  double distance_between_grid_points;

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

  std::vector<Point<dim_omega>> support_points;
  std::vector<bool> DoF_has_support_point;
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
// for error integration
template <int dim_omega> class Mask_Sigma : public Function<dim_omega> {
public:
  Mask_Sigma() : Function<dim_omega>(nof_scalar_fields) {}
  virtual double value(const Point<dim_omega> &p,
                       const unsigned int component) const override;
};
//RightHandSide
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
class PressureBoundaryValues_Omega : public Function<dim_omega>
{
public:
  PressureBoundaryValues_Omega(): Function<dim_omega>(1){}
  virtual double value(const Point<dim_omega>  &p,
                        const unsigned int component = 0) const override;
};

template <int dim_sigma>
class PressureBoundaryValues_Sigma : public Function<dim_sigma>
{
public:
  PressureBoundaryValues_Sigma() : Function<dim_sigma>(1) {}
  virtual double value(const Point<dim_sigma>  &p,
                        const unsigned int component = 0) const override;
};



template <int dim_omega> class ExactSolution : public Function<dim_omega> {
public:
  ExactSolution() : Function<dim_omega>(nof_scalar_fields) {}

  virtual void vector_value(const Point<dim_omega> &p,
                            Vector<double> &value) const override;
};

template <int dim_omega>
class BoundaryValues_Omega : public Function<dim_omega> {
public:
  BoundaryValues_Omega() : Function<dim_omega>(nof_scalar_fields) {}

  virtual double value(const Point<dim_omega> &p,
                       const unsigned int component = 0) const override;
};
template <int dim_omega>
double Mask_Sigma<dim_omega>::value(const Point<dim_omega> &p,
                                    const unsigned int component) const {
  if (component == 1 && isOnSigma(p)) {
    return 1;
  } else
    return 0;
}

template <int dim_omega>
double F_Omega<dim_omega>::value(const Point<dim_omega> &p,
                                 const unsigned int component) const {
  if (component == 0) {
    switch (constructed_solution) {
    case 1:
      return 0;
    case 2:
      return -2;
    case 3:
      return (2 * std::pow(w, 2)) * std::sin(w * p[0] + numbers::PI / 2) *
             std::cos(w * p[1]);
    default:
      return 0;
    }
  } else {
    return 0;
  }
}

template <int dim_sigma>
double F_Sigma<dim_sigma>::value(const Point<dim_sigma> &p,
                                 const unsigned int component) const {
  if (component == 1) {
    switch (constructed_solution) {
    case 1:
      return 0;
    case 2:
      return -2;
    case 3:
      return (std::pow(w, 2)) * std::sin(w * p[0] + numbers::PI / 2);
    default:
      return 0;
    }
  } else {
    return 0;
  }
}

template <int dim_omega>
double
BoundaryValues_Omega<dim_omega>::value(const Point<dim_omega> &p,
                                       const unsigned int component) const {
  if (component == 0) {

    switch (constructed_solution) {
    case 1:
      return p[0];
    case 2:
      return std::pow(p[0], 2);
    case 3:
      return 0;
    default:
      return 0;
    }
  } else
    return 0.0;
}

template <int dim_omega>
double PressureBoundaryValues_Omega<dim_omega>::value(const Point<dim_omega> &p,
                                    const unsigned int /*component*/) const
{
  return 0;
  //return -(alpha * p[0] * p[1] * p[1] / 2 + beta * p[0] -
    //        alpha * p[0] * p[0] * p[0] / 6);
}

template <int dim_sigma>
double PressureBoundaryValues_Sigma<dim_sigma>::value(const Point<dim_sigma> &p,
                                    const unsigned int /*component*/) const
{
  return 0;
  //return -(alpha * p[0] * p[1] * p[1] / 2 + beta * p[0] -
    //        alpha * p[0] * p[0] * p[0] / 6);
}

template <int dim_omega>
void ExactSolution<dim_omega>::vector_value(const Point<dim_omega> &p,
                                            Vector<double> &values) const {
  AssertDimension(values.size(), nof_scalar_fields);
  // value(0)
  switch (constructed_solution) {
  case 1:
    values(0) = p[0];
    break;
  case 2:
    values(0) = std::pow(p[0], 2);
    break;
  case 3:
    values(0) = std::sin(w * p[0] + numbers::PI / 2) * std::cos(w * p[1]);
    break;
  default:
    values(0) = 0;
    break;
  }

  if (isOnSigma(p)) {
    switch (constructed_solution) {
    case 1:
      values(1) = p[0];
      break;
    case 2:
      values(1) = std::pow(p[0], 2);
      break;
    case 3:
      values(1) = std::sin(w * p[0] + numbers::PI / 2);
      break;
    default:
      values(1) = 0;
      break;
    }
  } else
    values(1) = 0;
}



//KInverse
  template <int dim_omega>
  class KInverse : public TensorFunction<2, dim_omega>
  {
  public:
    KInverse()
      : TensorFunction<2, dim_omega>()
    {}

    virtual void
    value_list(const std::vector<Point<dim_omega>> &points,
                std::vector<Tensor<2, dim_omega>>   &values) const override;
  };
      template <int dim_omega>
  void KInverse<dim_omega>::value_list(const std::vector<Point<dim_omega>> &points,
                                  std::vector<Tensor<2, dim_omega>>   &values) const
  {
    (void)points;
    AssertDimension(points.size(), values.size());

    for (auto &value : values)
      value = unit_symmetric_tensor<dim_omega>();
  }


  //kInverse
  template <int dim_sigma>
  class kInverse : public TensorFunction<2, dim_sigma>
  {
  public:
    kInverse()
      : TensorFunction<2, dim_sigma>()
    {}

    virtual void
    value_list(const std::vector<Point<dim_sigma>> &points,
                std::vector<Tensor<2, dim_sigma>>   &values) const override;
  };
      template <int dim_sigma>
  void kInverse<dim_sigma>::value_list(const std::vector<Point<dim_sigma>> &points,
                                  std::vector<Tensor<2, dim_sigma>>   &values) const
  {
    (void)points;
    AssertDimension(points.size(), values.size());

    for (auto &value : values)
      value = unit_symmetric_tensor<dim_sigma>();
  }


} // namespace PrescribedSolution
//DG
template <int dim_omega, int dim_sigma>
MixedCoupledLaplace<dim_omega, dim_sigma>::MixedCoupledLaplace(unsigned int _p_degree)
    : p_degree(_p_degree), fe(FE_Q<dim_omega>(p_degree + 1)^nof_scalar_fields, FE_RaviartThomas<dim_omega>(p_degree)),//fe(FESystem<dim_omega>(FE_Q<dim_omega>(p_degree + 1), nof_scalar_fields), FESystem<dim_omega>(FE_RaviartThomas<dim_omega>(p_degree))),
      dof_handler(triangulation),
      fe_sigma(FE_Q<dim_sigma>(p_degree+1) ^ nof_scalar_fields),
      dof_handler_sigma(triangulation_sigma) {}

template <int dim_omega, int dim_sigma>
void MixedCoupledLaplace<dim_omega, dim_sigma>::make_grid() {
  triangulation.clear();
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(refinement);

  if (dim_omega == 2 && false) {
    std::ofstream out("grid-1.svg");
    GridOut grid_out;
    grid_out.write_svg(triangulation, out);
    std::cout << "Grid written to grid-1.svg" << std::endl;
  }
}

template <int dim_omega, int dim_sigma>
void MixedCoupledLaplace<dim_omega, dim_sigma>::setup_system() {
  dof_handler.distribute_dofs(fe);

  DoFRenumbering::component_wise(dof_handler); //new

const std::vector<types::global_dof_index> dofs_per_component =
    DoFTools::count_dofs_per_fe_component(dof_handler);
  
  for(unsigned int i = 0; i < dofs_per_component.size(); i++)
  std::cout<<"dofs_per_component " <<dofs_per_component[i]<<std::endl;
  const unsigned int n_u = dofs_per_component[0] * nof_scalar_fields,
                      n_p = dofs_per_component[dofs_per_component.size()-1] ;//* nof_scalar_fields

  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Total number of cells: " << triangulation.n_cells()
            << std::endl
            << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << " (" << n_u << '+' << n_p << ')' << std::endl;

  const std::vector<types::global_dof_index> block_sizes = {n_p, n_u};
  BlockDynamicSparsityPattern dsp(block_sizes, block_sizes);
  DoFTools::make_sparsity_pattern(dof_handler, dsp);


  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);


  solution.reinit(block_sizes);
  system_rhs.reinit(block_sizes);

/*
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
  */
}

template <int dim_omega, int dim_sigma>
void MixedCoupledLaplace<dim_omega, dim_sigma>::assemble_system() {
  QGauss<dim_omega> quadrature_formula(fe.degree + 2);
  QGauss<dim_omega - 1> face_quadrature_formula(fe.degree + 2);
  Quadrature<dim_omega> dummy_quadrature(fe.get_unit_support_points());
  PrescribedSolution::F_Sigma<dim_sigma> f_sigma;
  PrescribedSolution::F_Omega<dim_omega> f_omega;

  const std::vector<types::global_dof_index> dofs_per_component =
    DoFTools::count_dofs_per_fe_component(dof_handler);
      const unsigned int n_u = dofs_per_component[0] + dofs_per_component[1],
                      n_p = dofs_per_component[2] * 2;


  FEValues<dim_omega> fe_values(fe, quadrature_formula,
                                update_values | update_gradients |
                                    update_quadrature_points |
                                    update_JxW_values);
  FEFaceValues<dim_omega> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  const unsigned int n_q_points      = quadrature_formula.size();
  const unsigned int n_face_q_points = face_quadrature_formula.size();


    //const PrescribedSolution::RightHandSide<dim_omega> right_hand_side;
    const PrescribedSolution::PressureBoundaryValues_Omega<dim_omega> pressure_boundary_values;
    const PrescribedSolution::KInverse<dim_omega> k_inverse;

    std::vector<double>         rhs_values(n_q_points);
    std::vector<double>         boundary_values(n_face_q_points);
    std::vector<Tensor<2, dim_omega>> k_inverse_values(n_q_points);

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);


  std::array<std::vector<types::global_dof_index>, nof_scalar_fields>
      line_dof_indices;


  //const ComponentSelectFunction<dim_omega> u_mask(std::make_pair(0, nof_scalar_fields), nof_scalar_fields* dim + nof_scalar_fields);
  ComponentMask u_mask = fe.component_mask (concentration_u);
  
  support_points.resize(dof_handler.n_dofs());

  //DoFTools::map_dofs_to_support_points(fe_values.get_mapping(), dof_handler, support_points);
  //fe_values.generalized_support_points();
 /* for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);
    for (unsigned int l = 0; l < cell->n_lines(); l++) {

      const typename DoFHandler<dim_omega>::active_line_iterator line =
          cell->line(l);

      std::vector<types::global_dof_index> local_dof_indices(
          fe.n_dofs_per_line() + fe.n_dofs_per_vertex() * 2);

      line->get_dof_indices(local_dof_indices);
      std::cout<<"fe.n_dofs_per_cell() "<<fe.n_dofs_per_cell()<<
      "fe.n_dofs_per_line() "<<fe.n_dofs_per_line()<< 
      "  fe.n_dofs_per_vertex() "<< fe.n_dofs_per_vertex()<<std::endl;
      for(types::global_dof_index ind: local_dof_indices)
        std::cout<<ind<<std::endl;
    }*/

    std::vector<Point<dim_omega>> unit_support_points_FE_Q(fe.n_dofs_per_cell());
    unit_support_points_FE_Q =  dof_handler.get_fe().base_element(concentration_base).get_unit_support_points();
    std::vector<std::pair< std::pair< unsigned int, unsigned int >, unsigned int >> dof_table(dof_handler.n_dofs());
    const Mapping<dim_omega> &mapping = fe_values.get_mapping();
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      //fe_values.reinit(cell);
       std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
        std::vector<Point<dim_omega>> cell_support_points(fe.n_dofs_per_cell());
        cell->get_dof_indices( local_dof_indices);


           std::cout<<"fe.n_dofs_per_cell() "<< fe.n_dofs_per_cell() << " unit_support_points_FE_Q.size() "<<unit_support_points_FE_Q.size()<<std::endl;
        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        {
            dof_table[local_dof_indices[i]] = dof_handler.get_fe().system_to_base_index(i);
                const unsigned int base =
            dof_table[local_dof_indices[i]].first.first;
            const unsigned int multiplicity =
            dof_table[local_dof_indices[i]].first.second;
            const unsigned int within_base_  =
            dof_table[local_dof_indices[i]].second; // same as above
                


                
            for(unsigned int i = 0; i < unit_support_points_FE_Q.size(); i++)
                cell_support_points[i] = mapping.transform_unit_to_real_cell(cell, unit_support_points_FE_Q[i]);
            unsigned int component_i = 2;
            if(base == concentration_base)
            {
              /*for(unsigned int j = 0; j < unit_support_points_FE_Q.size();j++)
              {
                cell_support_points[j] = fe_values.get_mapping().transform_unit_to_real_cell(cell, unit_support_points_FE_Q[j]);
                std::cout<<"points "<<unit_support_points_FE_Q[j] <<" cell "<<cell_support_points[j]<<std::endl;
              }*/
             support_points[local_dof_indices[i]] = cell_support_points[within_base_];
            component_i = dof_handler.get_fe().base_element(base).system_to_component_index(within_base_).first;
            }
            
          std::cout<<local_dof_indices[i]<< " "<< i <<" point "<<support_points[local_dof_indices[i]]<<" base "<<base <<" "<< multiplicity<<" "<< within_base_ <<" comp "<<component_i<<std::endl;
        }
       std::cout<<"----------"<<std::endl;
    }


  //boost::container::small_vector< Point< dim_omega >, GeometryInfo< dim_omega>::vertices_per_cell >
 // auto points_vertex = fe_values.get_mapping().get_vertices(cell);
 // std::cout<<points_vertex<<std::endl;
  
  

   std::cout<<"------------Start dof support points"<<std::endl;
  std::vector<typename DoFHandler<dim_omega>::face_iterator> faces;
  std::vector<typename DoFHandler<dim_omega>::active_line_iterator> lines;

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);

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
     // std::cout<<"fe.n_dofs_per_line() "<<fe.n_dofs_per_line()<< "  fe.n_dofs_per_vertex() "<< fe.n_dofs_per_vertex()<<std::endl;
     // for(types::global_dof_index ind: local_dof_indices)
     // std::cout<<ind<<std::endl;
     
      std::array<std::vector<types::global_dof_index>, nof_scalar_fields>
          dof_indices_sigma_cell;
      std::vector<types::global_dof_index> dof_indices_sigma_cell_v2;

      bool push = true;
     // std::cout<<"local_dof_indices.size() "<<local_dof_indices.size()<<std::endl;
      for (unsigned int i = 0; i < local_dof_indices.size(); i++) {
        unsigned int index = local_dof_indices[i];
        Point<dim_omega> p = support_points[index];
        const unsigned int base = dof_table[index].first.first;
        const unsigned int within_base_ = dof_table[index].second; //fe.system_to_component_index(i).first;  dof_handler.get_fe().system_to_base_index(i)

        
        if(base == concentration_base)
        {
          unsigned int component_i = dof_handler.get_fe().base_element(base).system_to_component_index(within_base_).first;
          std::cout<<"dof "<<i<<" point "<<p <<" base "<<base<<" "<<" within_base_ "<<within_base_<<" "<<component_i<<std::endl;
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
      }
      if (push) {
        dof_indices_sigma_per_cells_comp.push_back(dof_indices_sigma_cell);
        dof_indices_sigma_per_cells.push_back(dof_indices_sigma_cell_v2);
      }
    }
  }
  std::cout<<"Loop ende.--------------------"<<std::endl;
  /*FEPointEvaluation<nof_scalar_fields, dim_omega> fe_point_eval(
      fe_values.get_mapping(), fe,
      update_values | update_gradients | update_quadrature_points |
          update_JxW_values);*/

  //"----------start loops-----------------"
  for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs = 0;

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    cell->get_dof_indices(local_dof_indices);

    for (const unsigned int q_index : fe_values.quadrature_point_indices()) {
      for (const unsigned int i :
           fe_values.dof_indices()) { // testfunction phi_j
      //const unsigned int component_i = fe.system_to_component_index(i).first;
        //unsigned int component_i = dof_handler.get_fe().base_element(concentration_base).system_to_component_index(within_base_).first;
        
        const Tensor<1, dim_omega> phi_i_u = fe_values[velocity_P].value(i, q_index);
        const double div_phi_i_u = fe_values[velocity_P].divergence(i, q_index);
        const double phi_i_p     = fe_values[concentration_u].value(i, q_index);
       
        for (const unsigned int j :
             fe_values.dof_indices()) { // j kommt von u_h sum over u_j phi_j

          /*cell_matrix(i, j) +=
              fe_values[concentration_u].gradient(i, q_index) // grad phi_i(x_q)
              *
              fe_values[concentration_u].gradient(j, q_index) // grad phi_j(x_q)
              * fe_values.JxW(q_index);                       // dx
              */
          const Tensor<1, dim_omega> phi_j_u =
                    fe_values[velocity_P].value(j, q_index);
                  const double div_phi_j_u =
                    fe_values[velocity_P].divergence(j, q_index);
                  const double phi_j_p = fe_values[concentration_u].value(j, q_index);

                cell_matrix(i, j) +=
                    (phi_i_u * phi_j_u //* k_inverse_values[q] 
                     - phi_i_p * div_phi_j_u                 //
                     - div_phi_i_u * phi_j_p)                //
                    * fe_values.JxW(q_index);
                


        }
        const auto &x_q = fe_values.quadrature_point(q_index);

        /*cell_rhs(i) += fe_values.shape_value(i, q_index) *
                       f_omega.value(x_q, 0) * // f(x_q)
                       fe_values.JxW(q_index);           // dx;
                       */
        cell_rhs(i) += -phi_i_p * f_omega.value(x_q, 0) * fe_values.JxW(q_index);
      }
    }



    for (const auto &face : cell->face_iterators())
      if (face->at_boundary())
        {
          fe_face_values.reinit(cell, face);

          pressure_boundary_values.value_list(
            fe_face_values.get_quadrature_points(), boundary_values);

          for (unsigned int q = 0; q < n_face_q_points; ++q)
            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              cell_rhs(i) += -(fe_face_values[velocity_P].value(i, q) * //
                                fe_face_values.normal_vector(q) *        //
                                boundary_values[q] *                     //
                                fe_face_values.JxW(q));
        }



    for (const unsigned int i : fe_values.dof_indices()) {

      for (const unsigned int j : fe_values.dof_indices()) {

       system_matrix.add(local_dof_indices[i], local_dof_indices[j],
                        cell_matrix(i, j));
      }

      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
  }



  /*for (std::vector<types::global_dof_index> cell_sigma :
       dof_indices_sigma_per_cells) {
    triangulation_sigma.clear();

    GridGenerator::hyper_cube(triangulation_sigma,
                              support_points[cell_sigma[0]][0],
                              support_points[cell_sigma[2]][0]);
    triangulation_sigma.refine_global(0);

    dof_handler_sigma.distribute_dofs(fe_sigma);

    QGauss<dim_sigma> quadrature_formula_sigma(fe_sigma.degree + 1);

    FEValues<dim_sigma> fe_values_sigma(fe_sigma, quadrature_formula_sigma,
                                        update_values | update_gradients |
                                            update_quadrature_points |
                                            update_JxW_values);

    std::vector<types::global_dof_index> local_dof_indices = cell_sigma;

    for (const auto &cell_sigma : dof_handler_sigma.active_cell_iterators()) {
      fe_values_sigma.reinit(cell_sigma);
      cell_matrix = 0;
      cell_rhs = 0;

      for (const unsigned int q_index :
           fe_values_sigma.quadrature_point_indices()) {
        for (const unsigned int i : fe_values_sigma.dof_indices()) {
          const unsigned int component_i =
              fe_sigma.system_to_component_index(i).first;
          for (const unsigned int j : fe_values_sigma.dof_indices()) {

            // mu gradient in Sigma
            cell_matrix(i, j) +=
                ((fe_values_sigma[concentration_mu].gradient(i, q_index) *
                  fe_values_sigma[concentration_mu].gradient(j, q_index))) *
                fe_values_sigma.JxW(q_index);

            // u in Omega
            cell_matrix(i, j) +=
                g * (fe_values[concentration_u].value(j, q_index) *
                     fe_values[concentration_u].value(i, q_index) *
                     fe_values.JxW(q_index));
            // mu in Sigma
            cell_matrix(i, j) +=
                g * (((fe_values_sigma[concentration_mu].value(j, q_index) *
                       fe_values_sigma[concentration_mu].value(i, q_index))) *
                     fe_values_sigma.JxW(q_index));

            // Cross
            // u in Sigma
            cell_matrix(i, j) +=
                -g * (((fe_values_sigma[concentration_u].value(j, q_index) *
                        fe_values_sigma[concentration_mu].value(i, q_index))) *
                      fe_values_sigma.JxW(q_index));

            // mu in Omega
            cell_matrix(i, j) +=
                -g * (fe_values[concentration_mu].value(j, q_index) *
                      fe_values[concentration_u].value(i, q_index) *
                      fe_values.JxW(q_index));
          }
          const auto &x_q = fe_values_sigma.quadrature_point(q_index);
          cell_rhs(i) +=
              (fe_values_sigma.shape_value(i, q_index) * // phi_i(x_q)
               f_sigma.value(x_q, component_i) *         // f(x_q)
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
*/

  // Boundary Sigma
 /* std::map<types::global_dof_index, double> boundary_values_sigma;
  for (types::global_dof_index i :
       dof_indices_boundary_sigma[concentration_mu.component]) {
    switch (constructed_solution) {
    case 1:
      boundary_values_sigma.insert(std::make_pair(i, support_points[i][0]));
      break;
    case 2:
      boundary_values_sigma.insert(
          std::make_pair(i, std::pow(support_points[i][0], 2)));
      break;
    case 3:
      boundary_values_sigma.insert(std::make_pair(
          i, std::sin(w * support_points[i][0] + numbers::PI / 2)));
      break;
    default:
      boundary_values_sigma.insert(std::make_pair(i, 0));
      break;
    }
  }
  */
  /*MatrixTools::apply_boundary_values(boundary_values_sigma, system_matrix,
                                     solution, system_rhs, true);

  std::map<types::global_dof_index, double> midpoint_value_sigma;
  midpoint_value_sigma.insert(
      std::make_pair(dof_index_midpoint[concentration_mu.component], 1));

  // Boundary Omega
  std::map<types::global_dof_index, double> boundary_values_omega;
  VectorTools::interpolate_boundary_values(
      fe_values.get_mapping(), dof_handler, 0,
      PrescribedSolution::BoundaryValues_Omega<dim_omega>(),
      boundary_values_omega);
  MatrixTools::apply_boundary_values(boundary_values_omega, system_matrix,
                                     solution, system_rhs, true);
*/
  // Extend Sigma
  /* std::map<types::global_dof_index, double> dirac_extend_zero;
    for (types::global_dof_index i :
         dof_indices_per_component[concentration_mu.component]) {
      if (dof_indices_sigma[concentration_mu.component].find(i) ==
          dof_indices_sigma[concentration_mu.component].end()) {
        dirac_extend_zero.insert(std::make_pair(i, 0.0));
      }
    }
    MatrixTools::apply_boundary_values(dirac_extend_zero, system_matrix,
    solution, system_rhs, false);*/

  /*for(types::global_dof_index i :
   dof_indices_per_component[concentration_mu.component])
   {
     if (dof_indices_sigma[concentration_mu.component].find(i) ==
          dof_indices_sigma[concentration_mu.component].end())
          {
           system_matrix.set(i,i,1);
           system_rhs[i] = 0;
          }

   }
 */
}
template <int dim_omega, int dim_sigma>
void MixedCoupledLaplace<dim_omega, dim_sigma>::solve() {
  std::cout<<"solve........."<<std::endl;
      // As a first step we declare references to all block components of the
    // matrix, the right hand side and the solution vector that we will
    // need.
    const auto &M = system_matrix.block(0, 0);
    const auto &B = system_matrix.block(0, 1);

    const auto &F = system_rhs.block(0);
    const auto &G = system_rhs.block(1);

    auto &U = solution.block(0);
    auto &P = solution.block(1);

    // Then, we will create corresponding LinearOperator objects and create
    // the <code>op_M_inv</code> operator:
    const auto op_M = linear_operator(M);
    const auto op_B = linear_operator(B);

    ReductionControl         reduction_control_M(2000, 1.0e-18, 1.0e-10);
    SolverCG<Vector<double>> solver_M(reduction_control_M);
    PreconditionJacobi<SparseMatrix<double>> preconditioner_M;

    preconditioner_M.initialize(M);

    const auto op_M_inv = inverse_operator(op_M, solver_M, preconditioner_M);

    // This allows us to declare the Schur complement <code>op_S</code> and
    // the approximate Schur complement <code>op_aS</code>:
    const auto op_S = transpose_operator(op_B) * op_M_inv * op_B;
    const auto op_aS =
      transpose_operator(op_B) * linear_operator(preconditioner_M) * op_B;

    // We now create a preconditioner out of <code>op_aS</code> that
    // applies a fixed number of 30 (inexpensive) CG iterations:
    IterationNumberControl   iteration_number_control_aS(30, 1.e-18);
    SolverCG<Vector<double>> solver_aS(iteration_number_control_aS);

    const auto preconditioner_S =
      inverse_operator(op_aS, solver_aS, PreconditionIdentity());

    // Now on to the first equation. The right hand side of it is
    // $B^TM^{-1}F-G$, which is what we compute in the first few lines. We
    // then solve the first equation with a CG solver and the
    // preconditioner we just declared.
    const auto schur_rhs = transpose_operator(op_B) * op_M_inv * F - G;

    SolverControl            solver_control_S(2000, 1.e-12);
    SolverCG<Vector<double>> solver_S(solver_control_S);

    const auto op_S_inv = inverse_operator(op_S, solver_S, preconditioner_S);

    P = op_S_inv * schur_rhs;

    std::cout << solver_control_S.last_step()
              << " CG Schur complement iterations to obtain convergence."
              << std::endl;

    // After we have the pressure, we can compute the velocity. The equation
    // reads $MU=-BP+F$, and we solve it by first computing the right hand
    // side, and then multiplying it with the object that represents the
    // inverse of the @ref GlossMassMatrix "mass matrix":
    U = op_M_inv * (F - op_B * P);
}
template <int dim_omega, int dim_sigma>
std::array<double, 2> MixedCoupledLaplace<dim_omega, dim_sigma>::compute_errors() {
  const ComponentSelectFunction<dim_omega> u_mask(
      0, nof_scalar_fields); // mu_mask(1, nof_scalar_fields);
  const PrescribedSolution::Mask_Sigma<dim_omega> mu_mask;

  PrescribedSolution::ExactSolution<dim_omega> exact_solution;
  Vector<double> cellwise_errors(triangulation.n_active_cells());

  QTrapezoid<1> q_trapez;
  QIterated<dim_omega> quadrature(q_trapez, fe.degree + 2);

  VectorTools::integrate_difference(dof_handler, solution, exact_solution,
                                    cellwise_errors, quadrature,
                                    VectorTools::L2_norm, &u_mask);
  const double u_l2_error = VectorTools::compute_global_error(
      triangulation, cellwise_errors, VectorTools::L2_norm);

  Vector<double> cellwise_errors_sigma(triangulation.n_active_cells());
  VectorTools::integrate_difference(dof_handler, solution, exact_solution,
                                    cellwise_errors_sigma, quadrature,
                                    VectorTools::L2_norm, &mu_mask);
  const double mu_l2_error = VectorTools::compute_global_error(
      triangulation, cellwise_errors_sigma, VectorTools::L2_norm);
  std::cout << "Errors: ||e_u||_L2 = " << u_l2_error
            << ",   ||e_mu||_L2 = " << mu_l2_error << std::endl;

  return std::array<double, 2>{{u_l2_error, mu_l2_error}};
}

template <int dim_omega, int dim_sigma>
void MixedCoupledLaplace<dim_omega, dim_sigma>::output_results() const {


  std::vector<std::string> solution_names(1, "concentration_u");
  solution_names.emplace_back("concentration_mu");
  solution_names.emplace_back( "concentration_P");
  solution_names.emplace_back( "concentration_P");
  //if ( nof_scalar_fields== 2)
    //solution_names.emplace_back("concentration_mu");


   std::vector<DataComponentInterpretation::DataComponentInterpretation> 
   interpretation(2, DataComponentInterpretation::component_is_scalar);
  //interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  //interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
   interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);

  DataOut<dim_omega> data_out;
  data_out.add_data_vector(dof_handler,
                            solution,
                            solution_names,
                            interpretation);

  data_out.build_patches(p_degree + 1);

  std::ofstream output("couplingLaplace_solution_" +
                       std::to_string(refinement) + "_" +
                       std::to_string(p_degree) + ".vtk");
  data_out.write_vtk(output);
}

template <int dim_omega, int dim_sigma>
std::array<double, 2>
MixedCoupledLaplace<dim_omega, dim_sigma>::run(unsigned int _refinement) {
  refinement = _refinement;
  std::cout << "---------------refinement: " << refinement
            << " p_degree: " << p_degree << " ------------------" << std::endl;

  make_grid();
  setup_system();
  assemble_system();
  
  MatrixOut matrix_out;
  std::ofstream out("system.vtk");
  matrix_out.build_patches(system_matrix, "system");
  matrix_out.write_vtk(out);


  /*std::ofstream fout("filename.txt");
  system_rhs.print(fout, 3, true, false);
  system_rhs.print(std::cout, 3, true, false);*/
  solve();
  std::array<double, 2> arr;// = compute_errors();
  output_results();
  return arr;
}

int main() {

  MixedCoupledLaplace<dimension_omega, dimension_sigma> *laplace_problem_2d;

  const unsigned int p_degree[1] = {0};
  constexpr unsigned int p_degree_size = sizeof(p_degree) / sizeof(p_degree[0]);
  const unsigned int refinement[1] = {1};
  constexpr unsigned int refinement_size =
      sizeof(refinement) / sizeof(refinement[0]);

  std::array<double, 2> results[p_degree_size][refinement_size];
  for (unsigned int r = 0; r < refinement_size; r++) {

    for (unsigned int p = 0; p < p_degree_size; p++) {
      laplace_problem_2d =
          new MixedCoupledLaplace<dimension_omega, dimension_sigma>(p_degree[p]);

      std::array<double, 2> arr = laplace_problem_2d->run(refinement[r]);
      results[p][r] = arr;

      delete laplace_problem_2d;
    }
  }
  std::cout << "--------" << std::endl;
  std::ofstream myfile;
  myfile.open("convergence_results.txt");
  for (unsigned int f = 0; f < nof_scalar_fields; f++) {
    myfile << "refinement/p_degree, ";
    for (unsigned int p = 0; p < p_degree_size; p++) {
      myfile << p_degree[p] << ",";
    }
    myfile << "\n";
    for (unsigned int r = 0; r < refinement_size; r++) {
      myfile << refinement[r] << ",";
      for (unsigned int p = 0; p < p_degree_size; p++) {
        const double error = results[p][r][f];

        myfile << error;
        std::cout << error;
        if (r != 0) {
          const double rate =
              std::log2(results[p][r - 1][f] / results[p][r][f]);
          myfile << " (" << rate << ")";
          std::cout << " (" << rate << ")";
        }

        myfile << ",";
        std::cout << ",";
      }
      myfile << std::endl;
      std::cout << std::endl;
    }
    myfile << std::endl << std::endl;
    std::cout << std::endl << std::endl;
  }

  myfile.close();

  return 0;
}
