//https://www.dealii.org/current/doxygen/deal.II/code_gallery_Distributed_LDG_Method.html

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
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/meshworker/mesh_loop.h>
#include <deal.II/lac/trilinos_solver.h>

namespace Step20
{
  using namespace dealii;


  double g = 1;
  double w = numbers::PI * 3 / 2;

  constexpr unsigned int nof_scalar_fields{3};
  constexpr unsigned int dimension_Omega{2};
  constexpr unsigned int dimension_omega{1};
  constexpr unsigned int constructed_solution{2};
  constexpr unsigned int concentration_base{2}; //1
  constexpr unsigned int p_degree{0};
  
  //constexpr unsigned int p_degree_size = sizeof(p_degree) / sizeof(p_degree[0]);
  constexpr unsigned int refinement{4};

  constexpr unsigned int eta = 5;


  const FEValuesExtractors::Vector velocities(0);
  // const FEValuesExtractors::Scalar velocity_omega(dimension_Omega);
  const FEValuesExtractors::Scalar pressure(dimension_Omega);
  // const FEValuesExtractors::Scalar pressure_omega(dimension_Omega + 2);

 const FEValuesExtractors::Scalar velocity_omega(0);
 const FEValuesExtractors::Scalar pressure_omega(dimension_omega);

  template <int dim, int dim_omega>
  class MixedLaplaceProblem
  {
  public:
    //MixedLaplaceProblem();
    MixedLaplaceProblem(const unsigned int degree);

    //MixedLaplaceProblem(const MixedLaplaceProblem &);
    //MixedLaplaceProblem &operator=(const MixedLaplaceProblem &);

    void run(unsigned int _refinement);

  private:
    void make_grid_and_dofs();
    void assemble_system();
    void solve();
    void compute_errors() const;
    void output_results() const;

    const unsigned int degree; 
    unsigned int refinement{1};

    Triangulation<dim>  triangulation;
    const FESystem<dim> fe;
    DoFHandler<dim>     dof_handler;

    Triangulation<dim_omega> triangulation_omega;
    FESystem<dim_omega> fe_omega;
    DoFHandler<dim_omega> dof_handler_omega;


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

  unsigned int start_velocity_omega;
  unsigned int start_pressure_omega;
  };


  template <int dim>
  struct ScratchData
  {
    ScratchData(const Mapping<dim>        &mapping,
                const FiniteElement<dim>  &fe,
                const Quadrature<dim>     &quadrature,
                const Quadrature<dim - 1> &quadrature_face,
                const UpdateFlags          update_flags = update_values |
                                                 update_gradients |
                                                 update_quadrature_points |
                                                 update_JxW_values,
                const UpdateFlags interface_update_flags =
                  update_values | update_gradients | update_quadrature_points |
                  update_JxW_values | update_normal_vectors)
      : fe_values(mapping, fe, quadrature, update_flags)
      , fe_interface_values(mapping,
                            fe,
                            quadrature_face,
                            interface_update_flags)
    {}


    ScratchData(const ScratchData<dim> &scratch_data)
      : fe_values(scratch_data.fe_values.get_mapping(),
                  scratch_data.fe_values.get_fe(),
                  scratch_data.fe_values.get_quadrature(),
                  scratch_data.fe_values.get_update_flags())
      , fe_interface_values(scratch_data.fe_interface_values.get_mapping(),
                            scratch_data.fe_interface_values.get_fe(),
                            scratch_data.fe_interface_values.get_quadrature(),
                            scratch_data.fe_interface_values.get_update_flags())
    {}

    FEValues<dim>          fe_values;
    FEInterfaceValues<dim> fe_interface_values;
  };



  struct CopyDataFace
  {
    FullMatrix<double>                   cell_matrix;
    std::vector<types::global_dof_index> joint_dof_indices;
  };



  struct CopyData
  {
    FullMatrix<double>                   cell_matrix;
    Vector<double>                       cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;
    std::vector<CopyDataFace>            face_data;

    template <class Iterator>
    void reinit(const Iterator &cell, unsigned int dofs_per_cell)
    {
      cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
      cell_rhs.reinit(dofs_per_cell);

      local_dof_indices.resize(dofs_per_cell);
      cell->get_dof_indices(local_dof_indices);
    }
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
    class PressureBoundaryValues_omega : public Function<dim>
    {
    public:
      PressureBoundaryValues_omega()
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
        //: Function<dim>(dim + nof_scalar_fields)
        : Function<dim>(dim + 1)
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
      return 10; //-2
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
    double
    PressureBoundaryValues_omega<dim>::value(const Point<dim> &p,
                                       const unsigned int /*component*/) const
    {
      return -alpha * p[0];
    }

    template <int dim>
    void ExactSolution<dim>::vector_value(const Point<dim> &p,
                                          Vector<double>   &values) const
    {
      //AssertDimension(values.size(), dim + nof_scalar_fields);

      values(0) = alpha * p[1] * p[1] / 2 + beta - alpha * p[0] * p[0] / 2;
      values(1) = alpha * p[0] * p[1];
     // values(2) = 0;
      values(2) = -(alpha * p[0] * p[1] * p[1] / 2 + beta * p[0] -
                    alpha * p[0] * p[0] * p[0] / 6);
     // values(4) = 0;
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
    , fe(FESystem<dim>(FE_DGQ<dim>(degree), dim), FE_DGQ<dim>(degree))// FE_RaviartThomas<dim>(degree)    FESystem<dim>(FE_DGQ<dim>(degree), dim)
    , dof_handler(triangulation) 
    , fe_omega(FE_DGQ<dim_omega>(degree), FE_DGQ<dim_omega>(degree))
    , dof_handler_omega(triangulation_omega){}


  template <int dim, int dim_omega>
  void MixedLaplaceProblem<dim, dim_omega>::make_grid_and_dofs()
  {
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(refinement);

    dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise(dof_handler);


    const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
  for(unsigned int i = 0; i < dofs_per_component.size(); i++)
  std::cout<<"dofs_per_component " <<dofs_per_component[i]<<std::endl;
    const unsigned int n_u = dofs_per_component[0] * dim ,// +  dofs_per_component[dim]
                       n_p = dofs_per_component[dim] ;//+ dofs_per_component[dim +2]

    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Total number of cells: " << triangulation.n_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << " (" << n_u << '+' << n_p << ')' << std::endl;

    start_velocity_omega = dofs_per_component[0];
    start_pressure_omega = n_u + dofs_per_component[dim +1];



    const std::vector<types::global_dof_index> block_sizes = {n_u, n_p};
  
    BlockDynamicSparsityPattern dsp(block_sizes, block_sizes);
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    sparsity_pattern.copy_from(dsp);
    system_matrix.reinit(sparsity_pattern);

    solution.reinit(block_sizes);
    system_rhs.reinit(block_sizes);
  }

  template <int dim, int dim_omega>
  void MixedLaplaceProblem<dim, dim_omega>::assemble_system()
  {
   const QGauss<dim>     quadrature_formula(degree + 1);
   const QGauss<dim - 1> face_quadrature_formula(degree + 1);

      // Define custom quadrature points and weights
    std::vector<Point<dim>> my_quadrature_points = {Point<dim>(0.0,0.0), Point<dim>(0.5,0.5), Point<dim>(1,1)};
    std::vector<double> my_quadrature_weights = {1,2,1};

    // Create custom quadrature rule
   //const Quadrature<dim> quadrature_formula(my_quadrature_points, my_quadrature_weights);

    FEValues<dim>     fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    
    FEFaceValues<dim> fe_face_values(fe,
                                     face_quadrature_formula,
                                     update_values |  update_gradients | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);
    const Mapping<dim> &mapping = fe_values.get_mapping();    
    //const UpdateFlags interface_update_flags=              ;
    FEInterfaceValues<dim>  fe_interface_values(mapping,
                            fe,
                            face_quadrature_formula,
                            update_values | update_gradients | update_quadrature_points |
                    update_JxW_values | update_normal_vectors);

    const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();
    const unsigned int n_q_points      = quadrature_formula.size();
    std::cout<<"n_q_points "<<n_q_points <<std::endl;

    for(unsigned int i = 0; i < quadrature_formula.get_points().size(); i++)
    {
      std::cout<<quadrature_formula.get_points()[i]<<" | "<<quadrature_formula.get_weights()[i]<<std::endl;
    }
    std::cout<<".--------"<<std::endl;
    const unsigned int n_face_q_points = face_quadrature_formula.size();



    support_points.resize(dof_handler.n_dofs());

    std::vector<Point<dim>> unit_support_points_FE_Q(fe.n_dofs_per_cell());

    std::vector<std::pair< std::pair< unsigned int, unsigned int >, unsigned int >> dof_table(dof_handler.n_dofs());
    
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
                

            unit_support_points_FE_Q =  dof_handler.get_fe().base_element(base).get_unit_support_points();
            for(unsigned int j = 0; j < unit_support_points_FE_Q.size(); j++)
               cell_support_points[j] = mapping.transform_unit_to_real_cell(cell, unit_support_points_FE_Q[j]);
            
            unsigned int component_i = 11;
          //  if(base == concentration_base)
          //if(base > 1)
            {
              support_points[local_dof_indices[i]] = cell_support_points[within_base_];
              //component_i = dof_handler.get_fe().base_element(base).system_to_component_index(within_base_).first;
            }
            
          //std::cout<<local_dof_indices[i]<< " "<< i <<" point "<<support_points[local_dof_indices[i]]<<" base "<<base <<" "<< multiplicity<<" "<< within_base_ <<" comp "<<component_i<<std::endl;
        }
       //std::cout<<"----------"<<std::endl;
    }



    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


    const PrescribedSolution::RightHandSide<dim> right_hand_side;
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
        {
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const Tensor<1, dim> phi_i_u = fe_values[velocities].value(i, q);
              const double div_phi_i_u = fe_values[velocities].divergence(i, q);
              const double phi_i_p     = fe_values[pressure].value(i, q);
              const Tensor<1, dim> grad_phi_i_p = fe_values[pressure].gradient(i, q);
            /// const double phi_i_p_omega = fe_values[pressure_omega].value(i, q);
            //  const double phi_i_u_omega = fe_values[velocity_omega].value(i, q);

              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                {
                  const Tensor<1, dim> phi_j_u = fe_values[velocities].value(j, q);
                  const double div_phi_j_u =  fe_values[velocities].divergence(j, q);
                  const double phi_j_p = fe_values[pressure].value(j, q);
                  const Tensor<1, dim> grad_phi_j_p = fe_values[pressure].gradient(j, q);
               //   const double phi_j_p_omega  =  fe_values[pressure_omega].value(j, q);
                //  const double phi_j_u_omega  =  fe_values[velocity_omega].value(j, q);

                  local_matrix(i, j) +=
                    (phi_j_u * k_inverse_values[q] * phi_i_u //
                     - phi_j_p * div_phi_i_u                 //
                    // - div_phi_i_u * phi_j_p                //
                    -  phi_j_u * grad_phi_i_p


                    // + phi_i_p_omega * phi_j_p_omega 
                   //  + phi_i_u_omega * phi_j_u_omega
                   )
                    * fe_values.JxW(q);
                }

              local_rhs(i) += phi_i_p * rhs_values[q] * fe_values.JxW(q);
            }
         }




        for (const auto &face : cell->face_iterators())
        {
          unsigned int face_index = cell->face_iterator_to_index(face);
          fe_face_values.reinit(cell, face);
          fe_interface_values.reinit(cell, face_index);
          
           
          if (face->at_boundary())
          {
           // std::cout<<face_index<<std::endl;
              pressure_boundary_values.value_list(
                fe_face_values.get_quadrature_points(), boundary_values);

              for (unsigned int q = 0; q < n_face_q_points; ++q)
              {
                for (unsigned int i = 0; i < dofs_per_cell; ++i)
                {
                 local_rhs(i) +=  -fe_face_values[velocities].value(i, q) * 
                                    fe_face_values.normal_vector(q) *        
                                    boundary_values[q] *                     
                                    fe_face_values.JxW(q)  

                                    + eta/cell->diameter() * boundary_values[q] * 
                                            fe_face_values[pressure].value(i,q) * fe_face_values.JxW(q);

                  for(unsigned int j = 0; j < dofs_per_cell; j++)
                  {
                    local_matrix(i, j) += (fe_values[velocities].value(j,q) * fe_face_values.normal_vector(q) 
                                          + (eta/cell->diameter()) * fe_values[pressure].value(j,q))
                                         * fe_values[pressure].value(i,q)  * fe_face_values.JxW(q);                       
                  }
                }
              }

            }
            else
            {         
              const unsigned int n_dofs  = fe_interface_values.n_current_interface_dofs();
              for (unsigned int q = 0; q < fe_interface_values.get_quadrature_points().size(); ++q)
              {
                for (unsigned int i = 0; i < n_dofs; ++i)
                {
                   //std::cout<<fe_interface_values[pressure].value(1,i,q)<<" | "<<fe_face_values[pressure].value(i,q)<<std::endl;
                  for (unsigned int j = 0; j < n_dofs; ++j)
                  {
                   
                    local_matrix(i, j) += fe_interface_values[pressure].average_of_values(j,q) *  
                                          fe_interface_values[velocities].value(1,i,q) * fe_face_values.normal_vector(q)  * fe_face_values.JxW(q)
                                          +
                                          (fe_interface_values[velocities].average_of_values(j,q) * fe_face_values.normal_vector(q) 
                                          + (eta/cell->diameter()) * fe_interface_values[pressure].jump_in_values(j,q)) 
                                          * fe_interface_values[pressure].value(1,i,q)  * fe_face_values.JxW(q);                                                                                      
                  }
                }
              }
            }

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


//for()
//system_matrix.add(local_dof_indices[i],
  //                            local_dof_indices[j],1)
const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
const std::vector<types::global_dof_index> dofs_per_block =
      DoFTools::count_dofs_per_fe_block(dof_handler);


/*
std::cout<<pressure_omega.component<<" "<<dofs_per_component[pressure_omega.component]<<std::endl;
std::cout<<pressure.component<<" "<<dofs_per_component[pressure.component]<<std::endl;


std::cout<<"perblock"<<std::endl;
for(unsigned int i = 0; i< dofs_per_block.size(); i++)
  std::cout<<dofs_per_block[i]<<std::endl;


std::cout<<"percomp"<<std::endl;
for(unsigned int i = 0; i< dofs_per_component.size(); i++)
  std::cout<<dofs_per_component[i]<<std::endl;
*/



    GridGenerator::hyper_cube(triangulation_omega, -1, 1);
    triangulation_omega.refine_global(refinement);

    dof_handler_omega.distribute_dofs(fe_omega);
    DoFRenumbering::component_wise(dof_handler_omega);

    const std::vector<types::global_dof_index> dofs_per_component_omega =
      DoFTools::count_dofs_per_fe_component(dof_handler_omega);
  for(unsigned int i = 0; i < dofs_per_component_omega.size(); i++)
  std::cout<<"dofs_per_component_omega " <<dofs_per_component_omega[i]<<std::endl;


 std::cout << "Number of degrees of freedom omega: " << dof_handler_omega.n_dofs()<<std::endl;

  std::cout << "Number of active cells omega: " << triangulation_omega.n_active_cells()
              << std::endl
              << "Total number of cells omega: " << triangulation_omega.n_cells()
              << std::endl;

   const QGauss<dim_omega>     quadrature_formula_omega(degree + 2);
    const QGauss<dim_omega - 1> face_quadrature_formula_omega(degree + 2);

    FEValues<dim_omega> fe_values_omega(fe_omega,
                            quadrature_formula_omega,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim_omega> fe_face_values_omega(fe_omega,
                                     face_quadrature_formula_omega,
                                     update_values | update_normal_vectors |
                                       update_quadrature_points |
                                       update_JxW_values);

    const unsigned int dofs_per_cell_omega   = fe_omega.n_dofs_per_cell();
    const unsigned int n_q_points_omega      = quadrature_formula_omega.size();
    const unsigned int n_face_q_points_omega = face_quadrature_formula_omega.size();

     std::cout << "Number of degrees of freedom omega cell: " <<dofs_per_cell_omega<<std::endl;

    FullMatrix<double> local_matrix_omega(dofs_per_cell_omega, dofs_per_cell_omega);
    Vector<double>     local_rhs_omega(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices_omega(dofs_per_cell_omega);


    const PrescribedSolution::RightHandSide_omega<dim_omega> right_hand_side_omega;
    const PrescribedSolution::PressureBoundaryValues_omega<dim_omega>
                                            pressure_boundary_values_omega;
    const PrescribedSolution::KInverse<dim_omega> k_inverse_omega;

    std::vector<double>         rhs_values_omega(n_q_points_omega);
    std::vector<double>         boundary_values_omega(n_face_q_points_omega);
   std::vector<Tensor<2, dim_omega>> k_inverse_values_omega(n_q_points_omega);


std::vector<std::pair< std::pair< unsigned int, unsigned int >, unsigned int >> dof_table_omega(dof_handler_omega.n_dofs());
 for (const auto &cell : dof_handler_omega.active_cell_iterators())
    {
       std::vector<types::global_dof_index> local_dof_indices(fe_omega.n_dofs_per_cell());
        //std::vector<Point<dim>> cell_support_points(fe.n_dofs_per_cell());
        cell->get_dof_indices( local_dof_indices);
        for (unsigned int i = 0; i < fe_omega.n_dofs_per_cell(); ++i)
        {
                    dof_table_omega[local_dof_indices[i]] = dof_handler_omega.get_fe().system_to_base_index(i);
                        const unsigned int base =
                    dof_table_omega[local_dof_indices[i]].first.first;
                    const unsigned int multiplicity =
                    dof_table_omega[local_dof_indices[i]].first.second;
                    const unsigned int within_base_  =
                    dof_table_omega[local_dof_indices[i]].second; // same as above
        }
    }
 /* std::cout<<"table omega "<<std::endl;
  for(unsigned int i = 0; i < dof_table_omega.size(); i++)
    {
      std::cout<<i<<" "<<dof_table_omega[i].first.first << " "<<dof_table_omega[i].second<<std::endl;
    }
std::cout<<"------"<<std::endl;
  */





 for (const auto &cell : dof_handler_omega.active_cell_iterators())
      {
        fe_values_omega.reinit(cell);

        local_matrix_omega = 0;
        local_rhs_omega    = 0;

        right_hand_side_omega.value_list(fe_values_omega.get_quadrature_points(),
                                   rhs_values_omega);
        k_inverse_omega.value_list(fe_values_omega.get_quadrature_points(),
                             k_inverse_values_omega);

        for (unsigned int q = 0; q < n_q_points_omega; ++q)
          for (unsigned int i = 0; i < dofs_per_cell_omega; ++i)
            {
              const double phi_i_u = fe_values_omega[velocity_omega].value(i, q);
              const double div_phi_i_u = fe_values_omega[velocity_omega].gradient(i, q)[0];
              const double phi_i_p     = fe_values_omega[pressure_omega].value(i, q);
              const double grad_phi_i_p     = fe_values_omega[pressure_omega].gradient(i, q)[0];

              for (unsigned int j = 0; j < dofs_per_cell_omega; ++j)
                {
                  const double phi_j_u = fe_values_omega[velocity_omega].value(j, q);
                  const double div_phi_j_u =  fe_values_omega[velocity_omega].gradient(j, q)[0];
                  const double phi_j_p = fe_values_omega[pressure_omega].value(j, q);
                  const double grad_phi_j_p = fe_values_omega[pressure_omega].gradient(j, q)[0];


                  /*local_matrix_omega(i, j) +=
                    (phi_i_u * phi_j_u //* k_inverse_values_omega[q]
                     - phi_i_p * div_phi_j_u                 //
                     - div_phi_i_u * phi_j_p                //
                      )
                    * fe_values_omega.JxW(q);*/
                    

                /*   local_matrix_omega(i, j) +=
                    (grad_phi_i_p * grad_phi_j_p //* k_inverse_values_omega[q]
                      )
                    * fe_values_omega.JxW(q);*/



                   local_matrix_omega(i, j) +=
                    (phi_i_p * phi_j_p + //* k_inverse_values_omega[q]
                    phi_i_u * phi_j_u)
                    * fe_values_omega.JxW(q);
                   
                }

             // local_rhs_omega(i) += -phi_i_p * rhs_values_omega[q] * fe_values_omega.JxW(q);
             local_rhs_omega(i) += phi_i_p * rhs_values_omega[q] * fe_values_omega.JxW(q);
            }



      /* for (const auto &face : cell->face_iterators())
          if (face->at_boundary())
            {
              fe_face_values_omega.reinit(cell, face);

              pressure_boundary_values_omega.value_list(
                fe_face_values_omega.get_quadrature_points(), boundary_values_omega);

              for (unsigned int q = 0; q < n_face_q_points_omega; ++q)
                for (unsigned int i = 0; i < dofs_per_cell_omega; ++i)
                  local_rhs_omega(i) += -(fe_face_values_omega[velocity_omega].value(i, q)* //
                                    fe_face_values_omega.normal_vector(q)[0] *        //
                                    boundary_values_omega[q] *                     //
                                    fe_face_values_omega.JxW(q));
            }
*/



        //std::cout<<"---"<<std::endl;
        cell->get_dof_indices(local_dof_indices_omega);
       /*for(unsigned int l = 0; l < local_dof_indices_omega.size(); l++)
          std::cout<<local_dof_indices_omega[l]<<std::endl;*/

        //std::cout<<"start "<<start_velocity_omega<< " "<<start_pressure_omega<<std::endl;
       for (unsigned int i = 0; i < dofs_per_cell_omega; ++i)
        {
            const unsigned int base_i =
            dof_table_omega[local_dof_indices_omega[i]].first.first;
            const unsigned int within_base_i  =
            dof_table_omega[local_dof_indices_omega[i]].second;

          types::global_dof_index i_global = base_i == 0 ? local_dof_indices_omega[i] + start_velocity_omega : local_dof_indices_omega[i] - dofs_per_component_omega[0] + start_pressure_omega;
          
          for (unsigned int j = 0; j < dofs_per_cell_omega; ++j)
          {
            const unsigned int base_j =
            dof_table_omega[local_dof_indices_omega[j]].first.first;
            const unsigned int within_base_j  =
            dof_table_omega[local_dof_indices_omega[j]].second;
            types::global_dof_index j_global = base_j == 0 ? local_dof_indices_omega[j] + start_velocity_omega : local_dof_indices_omega[j] - dofs_per_component_omega[0]  + start_pressure_omega;
            
           // std::cout<<base_i<<" "<<local_dof_indices_omega[i]<< " "<<i_global<< " -- " <<base_j<<" "<<local_dof_indices_omega[j]<< " "<<j_global<<" value "<<local_matrix_omega(i, j)<< std::endl;
            
            /*system_matrix.add(i_global,
                              j_global,
                              local_matrix_omega(i, j));*/
          }
          //std::cout<<"rhs "<<local_rhs_omega(i)<<std::endl;
         // system_rhs(i_global) += local_rhs_omega(i);
        }         
    }


/*
std::cout<<"set ii "<<std::endl;

for(unsigned int i = 0; i < dof_table.size(); i++)
{
  //std::cout<<i<<" "<<dof_table[i].first.first<<std::endl;
 if(dof_table[i].first.first == 1 || dof_table[i].first.first == 3)
  {

     if(system_matrix.el(i,i) == 0 )
      {
       // std::cout<<system_matrix.el(i,i);
        system_matrix.set(i,i,1);
        //system_rhs(i) = 1;
       // std::cout<<system_matrix.el(i,i)<<std::endl;
      }
      
   // std::cout<<system_matrix.block(dof_table[i].first.first,dof_table[i].first.first)(0,0)<<std::endl;
  }

}
*/
//system_matrix.print(std::cout);






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
      std::cout << "Solving linear system... ";
      Timer timer;
      SparseDirectUMFPACK A_direct;
  
      solution = system_rhs;
      A_direct.solve(system_matrix, solution);
  
      timer.stop();
      std::cout << "done (" << timer.cpu_time() << "s)" << std::endl;

/*
    TrilinosWrappers::SolverDirect solver;
        TrilinosWrappers::MPI::Vector completely_distributed_solution(system_rhs);
        solver.solve(system_matrix,
                 completely_distributed_solution,
                 system_rhs);

   // constraints.distribute(completely_distributed_solution);
    solution = completely_distributed_solution;
    */


/*
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
    */
  }


  template <int dim, int dim_omega>
  void MixedLaplaceProblem<dim, dim_omega>::compute_errors() const
  {
/*     const ComponentSelectFunction<dim> pressure_mask(dim, dim + nof_scalar_fields);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + nof_scalar_fields); */
  /*  const ComponentSelectFunction<dim> pressure_mask(dim + 1, dim + nof_scalar_fields);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + nof_scalar_fields);*/

   const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + 1);
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
    //solution_names.emplace_back("q");
   // solution_names.emplace_back("U");    
    //solution_names.emplace_back("u");   
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation(dim,
                     DataComponentInterpretation::component_is_part_of_vector);
    interpretation.push_back(DataComponentInterpretation::component_is_scalar);
   // interpretation.push_back(DataComponentInterpretation::component_is_scalar);
    //interpretation.push_back(DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler,
                             solution,
                             solution_names,
                             interpretation);

    data_out.build_patches(degree + 1);

    std::ofstream output("solution.vtu");
    data_out.write_vtu(output);

//---------------omega------------------------------------
   /* BlockVector<double> solution_omega;
    const std::vector<types::global_dof_index> dofs_per_component_omega =
      DoFTools::count_dofs_per_fe_component(dof_handler_omega);
    for(unsigned int i = 0; i < dofs_per_component_omega.size(); i++)
    std::cout<<"dofs_per_component_omega " <<dofs_per_component_omega[i]<<std::endl;
    const std::vector<types::global_dof_index> block_sizes_omega = {dofs_per_component_omega[0], dofs_per_component_omega[1]};
    solution_omega.reinit(block_sizes_omega);
    for(unsigned int i = 0; i < dofs_per_component_omega[0]; i++)
      solution_omega[i] = solution[start_velocity_omega + i];

    for(unsigned int i = 0; i < dofs_per_component_omega[1]; i++)
      solution_omega[dofs_per_component_omega[0]+ i] = solution[start_pressure_omega + i];
    std::vector<std::string> solution_names_omega;
    solution_names_omega.emplace_back("q");
    solution_names_omega.emplace_back("u");
 
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      interpretation_omega;
    interpretation_omega.push_back(DataComponentInterpretation::component_is_scalar);
    interpretation_omega.push_back(DataComponentInterpretation::component_is_scalar);
   
    DataOut<dim_omega> data_out_omega;
    data_out_omega.add_data_vector(dof_handler_omega,
                             solution_omega,
                             solution_names_omega,
                             interpretation_omega);

    data_out_omega.build_patches(degree + 1);

    std::ofstream output_omega("solution_omega.vtu");
    data_out_omega.write_vtu(output_omega);
    */
  }


  template <int dim, int dim_omega>
  void MixedLaplaceProblem<dim, dim_omega>::run(unsigned int _refinement)
  {
      refinement = _refinement;
  std::cout << "---------------refinement: " << refinement
            << " degree: " << degree << " ------------------" << std::endl;
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

      const unsigned int     fe_degree = p_degree;
      MixedLaplaceProblem<dimension_Omega, dimension_omega> mixed_laplace_problem(fe_degree);
      //MixedLaplaceProblem<2, 1> mixed_laplace_problem(fe_degree);
      mixed_laplace_problem.run(refinement);
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