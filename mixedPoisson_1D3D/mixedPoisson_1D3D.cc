#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>

#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

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
#include <deal.II/numerics/data_out.h>
#include <deal.II/fe/fe_q.h>

#include <fstream>
#include <iostream>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/base/tensor_function.h>


namespace Step20
{
  using namespace dealii;


  double g = 1;
  double w = numbers::PI * 3 / 2;

  constexpr unsigned int nof_scalar_fields{2};
  constexpr unsigned int dimension_Omega{2};
  constexpr unsigned int dimension_omega{1};
  constexpr unsigned int constructed_solution{2};
  constexpr unsigned int concentration_base{1};
  constexpr unsigned int p_degree{1};
  
  //constexpr unsigned int p_degree_size = sizeof(p_degree) / sizeof(p_degree[0]);
  constexpr unsigned int refinement{3};


  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dimension_Omega);
  const FEValuesExtractors::Scalar pressure_omega(dimension_Omega + 1);



  template <int dim, int dim_omega>
  class MixedLaplaceProblem
  {
  public:
    //MixedLaplaceProblem();
    MixedLaplaceProblem(const unsigned int degree);

    //MixedLaplaceProblem(const MixedLaplaceProblem &);
    //MixedLaplaceProblem &operator=(const MixedLaplaceProblem &);

    void run();

  private:
    void make_grid_and_dofs();
    void assemble_system();
    void solve();
    void compute_errors() const;
    void output_results() const;

    const unsigned int degree; 
    //unsigned int refinement{1};

    Triangulation<dim>  triangulation;
    const FESystem<dim> fe;
    DoFHandler<dim>     dof_handler;
/*
    Triangulation<dim_omega> triangulation_omega;
    FESystem<dim_omega> fe_omega;
    DoFHandler<dim_omega> dof_handler_omega;
*/

    BlockSparsityPattern      sparsity_pattern;
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

    std::vector<Point<dim>> support_points;
  std::vector<bool> DoF_has_support_point;
  };


  namespace PrescribedSolution
  {
    constexpr double alpha = 0.3;
    constexpr double beta  = 1;


    template <int dim>
    class RightHandSide : public Function<dim>
    {
    public:
      RightHandSide()
        : Function<dim>(1)
      {}

      virtual double value(const Point<dim>  &p,
                           const unsigned int component = 0) const override;
    };


    template <int dim>
    class RightHandSide_omega : public Function<dim>
    {
    public:
      RightHandSide_omega()
        : Function<dim>(1)
      {}

      virtual double value(const Point<dim>  &p,
                           const unsigned int component = 0) const override;
    };



    template <int dim>
    class PressureBoundaryValues : public Function<dim>
    {
    public:
      PressureBoundaryValues()
        : Function<dim>(1)
      {}

      virtual double value(const Point<dim>  &p,
                           const unsigned int component = 0) const override;
    };


    template <int dim>
    class ExactSolution : public Function<dim>
    {
    public:
      ExactSolution()
        : Function<dim>(dim + nof_scalar_fields)
      {}

      virtual void vector_value(const Point<dim> &p,
                                Vector<double>   &value) const override;
    };

    template <int dim>
    double RightHandSide<dim>::value(const Point<dim> & /*p*/,
                                     const unsigned int /*component*/) const
    {
      return 0;
    }

    template <int dim>
    double RightHandSide_omega<dim>::value(const Point<dim> & /*p*/,
                                     const unsigned int /*component*/) const
    {
      return -2;
    }

    template <int dim>
    double
    PressureBoundaryValues<dim>::value(const Point<dim> &p,
                                       const unsigned int /*component*/) const
    {
      return -(alpha * p[0] * p[1] * p[1] / 2 + beta * p[0] -
               alpha * p[0] * p[0] * p[0] / 6);
    }



    template <int dim>
    void ExactSolution<dim>::vector_value(const Point<dim> &p,
                                          Vector<double>   &values) const
    {
      AssertDimension(values.size(), dim + nof_scalar_fields);

      values(0) = alpha * p[1] * p[1] / 2 + beta - alpha * p[0] * p[0] / 2;
      values(1) = alpha * p[0] * p[1];
      values(2) = -(alpha * p[0] * p[1] * p[1] / 2 + beta * p[0] -
                    alpha * p[0] * p[0] * p[0] / 6);
      values(3) = 0;
    }

    template <int dim>
    class KInverse : public TensorFunction<2, dim>
    {
    public:
      KInverse()
        : TensorFunction<2, dim>()
      {}

      virtual void
      value_list(const std::vector<Point<dim>> &points,
                 std::vector<Tensor<2, dim>>   &values) const override;
    };

    template <int dim>
    void KInverse<dim>::value_list(const std::vector<Point<dim>> &points,
                                   std::vector<Tensor<2, dim>>   &values) const
    {
      (void)points;
      AssertDimension(points.size(), values.size());

      for (auto &value : values)
        value = unit_symmetric_tensor<dim>();
    }








constexpr double boundary1 = -0.5;
constexpr double boundary2 = 0.5;

template <int dim> bool isOnSigma(Point<dim> p) {
  bool return_value = true;
  if (p[0] >= boundary1 && p[0] <= boundary2)
    return_value = return_value && true;
  else
    return_value = return_value && false;
  for (unsigned int i = 1; i < dim; i++) {
    if (p[i] == 0)
      return_value = return_value && true;
    else
      return_value = return_value && false;
  }
  return return_value;
}
template <int dim> bool isOnSigma_boundary(Point<dim> p) {
  bool return_value = true;
  if (p[0] == boundary1 || p[0] == boundary2)
    return_value = return_value && true;
  else
    return_value = return_value && false;
  for (unsigned int i = 1; i < dim; i++) {
    if (p[i] == 0)
      return_value = return_value && true;
    else
      return_value = return_value && false;
  }
  return return_value;
}
template <int dim> bool isMidPoint(Point<dim> p) {
  bool return_value = true;
  for (unsigned int i = 0; i < dim; i++) {
    if (p[i] == 0)
      return_value = return_value && true;
    else
      return_value = return_value && false;
  }
  return return_value;
}
















  } // namespace PrescribedSolution
//fe(FESystem<dim>(FE_Q<dim>(degree ),2), FE_Q<dim>(degree )
//
  template <int dim, int dim_omega>
  MixedLaplaceProblem<dim, dim_omega>::MixedLaplaceProblem(const unsigned int _degree)
    : degree(_degree)
    , fe(FE_RaviartThomas<dim>(degree), FE_Q<dim>(degree)^nof_scalar_fields)
    , dof_handler(triangulation) {}


  template <int dim, int dim_omega>
  void MixedLaplaceProblem<dim, dim_omega>::make_grid_and_dofs()
  {
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(4);

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise(dof_handler);


    const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
  for(unsigned int i = 0; i < dofs_per_component.size(); i++)
  std::cout<<"dofs_per_component " <<dofs_per_component[i]<<std::endl;
    const unsigned int n_u = dofs_per_component[0],
                       n_p = dofs_per_component[dim]*nof_scalar_fields;

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: " << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_p << ')' << std::endl;

   
    const std::vector<types::global_dof_index> block_sizes = {n_u, n_p};
    BlockDynamicSparsityPattern                dsp(block_sizes, block_sizes);
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(block_sizes);
    system_rhs.reinit(block_sizes);
  }

  template <int dim, int dim_omega>
  void MixedLaplaceProblem<dim, dim_omega>::assemble_system()
  {
    const QGauss<dim>     quadrature_formula(degree + 2);
    const QGauss<dim - 1> face_quadrature_formula(degree + 2);

    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();



    support_points.resize(dof_handler.n_dofs());
    std::vector<Point<dim>> unit_support_points_FE_Q(fe.n_dofs_per_cell());
    unit_support_points_FE_Q =  dof_handler.get_fe().base_element(concentration_base).get_unit_support_points();
    std::vector<std::pair< std::pair< unsigned int, unsigned int >, unsigned int >> dof_table(dof_handler.n_dofs());
    const Mapping<dim> &mapping = fe_values.get_mapping();
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
       std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
        std::vector<Point<dim>> cell_support_points(fe.n_dofs_per_cell());
        cell->get_dof_indices( local_dof_indices);
          // std::cout<<"fe.n_dofs_per_cell() "<< fe.n_dofs_per_cell() << " unit_support_points_FE_Q.size() "<<unit_support_points_FE_Q.size()<<std::endl;
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
             support_points[local_dof_indices[i]] = cell_support_points[within_base_];
             component_i = dof_handler.get_fe().base_element(base).system_to_component_index(within_base_).first;
            }
            
          //std::cout<<local_dof_indices[i]<< " "<< i <<" point "<<support_points[local_dof_indices[i]]<<" base "<<base <<" "<< multiplicity<<" "<< within_base_ <<" comp "<<component_i<<std::endl;
        }
       //std::cout<<"----------"<<std::endl;
    }











  std::cout<<"------------Start dof support points"<<std::endl;
  std::vector<typename DoFHandler<dim>::face_iterator> faces;
  std::vector<typename DoFHandler<dim>::active_line_iterator> lines;

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    fe_values.reinit(cell);

    for (unsigned int l = 0; l < cell->n_lines(); l++) {

      const typename DoFHandler<dim>::active_line_iterator line =
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
        Point<dim> p = support_points[index];
        const unsigned int base = dof_table[index].first.first;
        const unsigned int within_base_ = dof_table[index].second; //fe.system_to_component_index(i).first;  dof_handler.get_fe().system_to_base_index(i)

        
        if(base == concentration_base)
        {
          unsigned int component_i = dof_handler.get_fe().base_element(base).system_to_component_index(within_base_).first;
          //std::cout<<"dof "<<i<<" point "<<p <<" base "<<base<<" "<<" within_base_ "<<within_base_<<" "<<component_i<<std::endl;
          dof_indices_per_component[component_i].insert(local_dof_indices[i]);
          if (PrescribedSolution::isOnSigma<dim>(p)) {
            dof_indices_sigma_cell[component_i].push_back(local_dof_indices[i]);
            dof_indices_sigma_cell_v2.push_back(local_dof_indices[i]);

            dof_indices_sigma[component_i].insert(local_dof_indices[i]);
            push = push && true;
          } else
            push = push && false;
          if (PrescribedSolution::isOnSigma_boundary<dim>(p)) {
            dof_indices_boundary_sigma[component_i].insert(local_dof_indices[i]);
          }
          if (PrescribedSolution::isMidPoint<dim>(p)) {
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


















    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


    const PrescribedSolution::RightHandSide<dim> right_hand_side;
    const PrescribedSolution::RightHandSide_omega<dim_omega> right_hand_side_omega;
    const PrescribedSolution::PressureBoundaryValues<dim>
                                            pressure_boundary_values;
    const PrescribedSolution::KInverse<dim> k_inverse;

    std::vector<double>         rhs_values(n_q_points);
    std::vector<double>         boundary_values(n_face_q_points);
    std::vector<Tensor<2, dim>> k_inverse_values(n_q_points);


    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        fe_values.reinit(cell);

        local_matrix = 0;
        local_rhs    = 0;

        right_hand_side.value_list(fe_values.get_quadrature_points(),
                                   rhs_values);
        k_inverse.value_list(fe_values.get_quadrature_points(),
                             k_inverse_values);

        for (unsigned int q = 0; q < n_q_points; ++q)
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const Tensor<1, dim> phi_i_u = fe_values[velocities].value(i, q);
              const double div_phi_i_u = fe_values[velocities].divergence(i, q);
              const double phi_i_p     = fe_values[pressure].value(i, q);
              const double phi_i_p_omega     = fe_values[pressure_omega].value(i, q);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const Tensor<1, dim> phi_j_u =
                    fe_values[velocities].value(j, q);
                  const double div_phi_j_u =
                    fe_values[velocities].divergence(j, q);
                  const double phi_j_p = fe_values[pressure].value(j, q);

                  local_matrix(i, j) +=
                    (phi_i_u * k_inverse_values[q] * phi_j_u //
                     - phi_i_p * div_phi_j_u                 //
                     - div_phi_i_u * phi_j_p                //
                     + phi_i_p_omega )
                    * fe_values.JxW(q);
                }

              local_rhs(i) += -phi_i_p * rhs_values[q] * fe_values.JxW(q);
            }

        for (const auto &face : cell->face_iterators())
          if (face->at_boundary())
            {
              fe_face_values.reinit(cell, face);

              pressure_boundary_values.value_list(
                fe_face_values.get_quadrature_points(), boundary_values);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                  local_rhs(i) += -(fe_face_values[velocities].value(i, q) * //
                                    fe_face_values.normal_vector(q) *        //
                                    boundary_values[q] *                     //
                                    fe_face_values.JxW(q));
            }


        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          for (unsigned int j = 0; j < dofs_per_cell; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              local_matrix(i, j));
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          system_rhs(local_dof_indices[i]) += local_rhs(i);
      }




/*

for(std::vector<types::global_dof_index> cell_omega :
       dof_indices_sigma_per_cells) {
    triangulation_omega.clear();

    GridGenerator::hyper_cube(triangulation_omega,
                              support_points[cell_omega[0]][0],
                              support_points[cell_omega[2]][0]);
    triangulation_omega.refine_global(0);

    dof_handler_omega.distribute_dofs(fe_omega);

    QGauss<dim_omega> quadrature_formula_sigma(fe_omega.degree + 1);

    FEValues<dim_omega> fe_values_omega(fe_omega, quadrature_formula_sigma,
                                        update_values | update_gradients |
                                            update_quadrature_points |
                                            update_JxW_values);

    std::vector<types::global_dof_index> local_dof_indices = cell_omega;

    for (const auto &cell_omega : dof_handler_omega.active_cell_iterators()) {
      fe_values_omega.reinit(cell_omega);
      local_matrix = 0;
      local_rhs = 0;

      for (const unsigned int q_index :
           fe_values_omega.quadrature_point_indices()) {
        for (const unsigned int i : fe_values_omega.dof_indices()) {
          const unsigned int component_i =
              fe_omega.system_to_component_index(i).first;
          for (const unsigned int j : fe_values_omega.dof_indices()) {

            // mu gradient in Sigma
            local_matrix(i, j) +=
                ((fe_values_omega[pressure_omega].gradient(i, q_index) *
                  fe_values_omega[pressure_omega].gradient(j, q_index))) *
                fe_values_omega.JxW(q_index);

            // u in Omega
            local_matrix(i, j) +=
                g * (fe_values[pressure].value(j, q_index) *
                     fe_values[pressure].value(i, q_index) *
                     fe_values.JxW(q_index));
            // mu in Sigma
            local_matrix(i, j) +=
                g * (((fe_values_omega[pressure_omega].value(j, q_index) *
                       fe_values_omega[pressure_omega].value(i, q_index))) *
                     fe_values_omega.JxW(q_index));

            // Cross
            // u in Sigma
            local_matrix(i, j) +=
                -g * (((fe_values_omega[pressure].value(j, q_index) *
                        fe_values_omega[pressure_omega].value(i, q_index))) *
                      fe_values_omega.JxW(q_index));

            // mu in Omega
            local_matrix(i, j) +=
                -g * (fe_values[pressure_omega].value(j, q_index) *
                      fe_values[pressure].value(i, q_index) *
                      fe_values.JxW(q_index));
          }
          const auto &x_q = fe_values_omega.quadrature_point(q_index);
          local_rhs(i) +=
              (fe_values_omega.shape_value(i, q_index) * // phi_i(x_q)
               right_hand_side_omega.value(x_q, component_i) *         // f(x_q)
               fe_values_omega.JxW(q_index));            // dx
        }
      }

      for (const unsigned int i : fe_values_omega.dof_indices()) {
        for (const unsigned int j : fe_values_omega.dof_indices()) {
          system_matrix.add(local_dof_indices[i], local_dof_indices[j],
                            local_matrix(i, j));
        }

        system_rhs(local_dof_indices[i]) += local_rhs(i);
      }
    }
  }

*/














  }













  template <int dim, int dim_omega>
  void MixedLaplaceProblem<dim, dim_omega>::solve()
  {

    const auto &M = system_matrix.block(0, 0);
    const auto &B = system_matrix.block(0, 1);

    const auto &F = system_rhs.block(0);
    const auto &G = system_rhs.block(1);

    auto &U = solution.block(0);
    auto &P = solution.block(1);

    const auto op_M = linear_operator(M);
    const auto op_B = linear_operator(B);

    ReductionControl         reduction_control_M(2000, 1.0e-18, 1.0e-10);
    SolverCG<Vector<double>> solver_M(reduction_control_M);
    PreconditionJacobi<SparseMatrix<double>> preconditioner_M;

    preconditioner_M.initialize(M);

    const auto op_M_inv = inverse_operator(op_M, solver_M, preconditioner_M);

    const auto op_S = transpose_operator(op_B) * op_M_inv * op_B;
    const auto op_aS =
      transpose_operator(op_B) * linear_operator(preconditioner_M) * op_B;


    IterationNumberControl   iteration_number_control_aS(30, 1.e-18);
    SolverCG<Vector<double>> solver_aS(iteration_number_control_aS);

    const auto preconditioner_S =
      inverse_operator(op_aS, solver_aS, PreconditionIdentity());


    const auto schur_rhs = transpose_operator(op_B) * op_M_inv * F - G;

    SolverControl            solver_control_S(2000, 1.e-12);
    SolverCG<Vector<double>> solver_S(solver_control_S);

    const auto op_S_inv = inverse_operator(op_S, solver_S, preconditioner_S);

    P = op_S_inv * schur_rhs;

    std::cout << solver_control_S.last_step()
              << " CG Schur complement iterations to obtain convergence."
              << std::endl;

    U = op_M_inv * (F - op_B * P);
  }


  template <int dim, int dim_omega>
  void MixedLaplaceProblem<dim, dim_omega>::compute_errors() const
  {
    const ComponentSelectFunction<dim> pressure_mask(dim, dim + nof_scalar_fields);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + nof_scalar_fields);


    PrescribedSolution::ExactSolution<dim> exact_solution;
    Vector<double> cellwise_errors(triangulation.n_active_cells());

    const QTrapezoid<1>  q_trapez;
    const QIterated<dim> quadrature(q_trapez, degree + 2);


    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &pressure_mask);
    const double p_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      exact_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &velocity_mask);
    const double u_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    std::cout << "Errors: ||e_p||_L2 = " << p_l2_error
              << ",   ||e_u||_L2 = " << u_l2_error << std::endl;
  }

  template <int dim, int dim_omega>
  void MixedLaplaceProblem<dim, dim_omega>::output_results() const
  {
    std::vector<std::string> solution_names(dim, "Q");
    solution_names.emplace_back("U");
    if(nof_scalar_fields == 2 )
    solution_names.emplace_back("u");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    if(nof_scalar_fields == 2 )
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler,
                             solution,
                             solution_names,
                             interpretation);

    data_out.build_patches(degree + 1);

    std::ofstream output("solution.vtu");
    data_out.write_vtu(output);
  }


  template <int dim, int dim_omega>
  void MixedLaplaceProblem<dim, dim_omega>::run()
  {
    make_grid_and_dofs();
    assemble_system();
    solve();
    compute_errors();
    output_results();
  }
} // namespace Step20

int main()
{
  try
    {
      using namespace Step20;

      const unsigned int     fe_degree = 1;
      MixedLaplaceProblem<dimension_Omega, dimension_omega> mixed_laplace_problem(fe_degree);
      //MixedLaplaceProblem<2, 1> mixed_laplace_problem(fe_degree);
      mixed_laplace_problem.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}