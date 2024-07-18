// @sect3{LDGPoisson.cc}
// The code begins as per usual with a long list of the the included
// files from the deal.ii library.
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>


#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
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

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>


#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/distributed/tria.h>



#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/non_matching/fe_values.h>







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











#include "Functions.cc"

using namespace dealii;

constexpr unsigned int dimension_Omega{2};
const FEValuesExtractors::Vector VectorField_omega(0);
const FEValuesExtractors::Scalar Potential_omega(1);

const FEValuesExtractors::Vector VectorField(0);
const FEValuesExtractors::Scalar Potential(dimension_Omega + 1);

const unsigned int dimension_gap = 1;


template <int dim, int dim_omega>
class LDGPoissonProblem
{

public:
  LDGPoissonProblem(const unsigned int degree,
                    const unsigned int n_refine);

  ~LDGPoissonProblem();

  std::array<double,4> run();



private:
  void make_grid();

  void make_dofs();

  void assemble_system();

  template<int _dim>
  void assemble_cell_terms(const FEValues<_dim>   &cell_fe,
                           FullMatrix<double>     &cell_matrix,
                           Vector<double>         &cell_vector, 
                           const TensorFunction<2, _dim>   &K_inverse_function,
                           const Function<_dim>   &_rhs_function,
                           const FEValuesExtractors::Vector VectorField, 
                           const FEValuesExtractors::Scalar Potential);

  void assemble_Neumann_boundary_terms(const FEFaceValues<dim>    &face_fe,
                                       FullMatrix<double>         &local_matrix,
                                       Vector<double>             &local_vector);
 
  template<int _dim>
  void assemble_Dirichlet_boundary_terms(const FEFaceValues<_dim>  &face_fe,
                                         FullMatrix<double>       &local_matrix,
                                         Vector<double>           &local_vector,
                                         const double              &h,
                                          const Function<_dim>        &Dirichlet_bc_function,
                                          const FEValuesExtractors::Vector VectorField, 
                                          const FEValuesExtractors::Scalar Potential);
  template<int _dim>
  void assemble_flux_terms(const FEFaceValuesBase<_dim>  &fe_face_values,
                           const FEFaceValuesBase<_dim>  &fe_neighbor_face_values,
                           FullMatrix<double>           &vi_ui_matrix,
                           FullMatrix<double>           &vi_ue_matrix,
                           FullMatrix<double>           &ve_ui_matrix,
                           FullMatrix<double>           &ve_ue_matrix,
                           const double                  &h,
                           const FEValuesExtractors::Vector VectorField, 
                           const FEValuesExtractors::Scalar Potential);

  void distribute_local_flux_to_global(
     FullMatrix<double> &vi_ui_matrix,
     FullMatrix<double> &vi_ue_matrix,
     FullMatrix<double> &ve_ui_matrix,
     FullMatrix<double> &ve_ue_matrix,
    const std::vector<types::global_dof_index> &local_dof_indices,
    const std::vector<types::global_dof_index> &local_neighbor_dof_indices);

template<int _dim>
void dof_omega_to_Omega(const DoFHandler<_dim>  &dof_handler,
                        std::vector<types::global_dof_index> &local_dof_indices_omega);


  void solve();

  std::array<double, 4> compute_errors() const;
  void output_results() const;






  const unsigned int degree;
  const unsigned int n_refine;
  double penalty;
  double h_max;
  double h_min;

  enum
  {
    Dirichlet,
    Neumann
  };

  Triangulation<dim>                              triangulation;
  FESystem<dim>                                   fe;
  DoFHandler<dim>                                 dof_handler;

  Triangulation<dim_omega>                        triangulation_omega;
  FESystem<dim_omega>                             fe_omega;
  DoFHandler<dim_omega>                           dof_handler_omega;

  AffineConstraints<double>                       constraints;

  /*  SparsityPattern                                 sparsity_pattern;

    SparseMatrix                  system_matrix;
   Vector                   solution;
   Vector                   system_rhs;*/
    /*BlockSparsityPattern      sparsity_pattern;
    BlockSparseMatrix<double> system_matrix;

    BlockVector<double> solution;
    BlockVector<double> system_rhs;*/

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
 
  Vector<double> solution;
   Vector<double> solution_omega;
  Vector<double> system_rhs;

  ConditionalOStream                              pcout;
  TimerOutput                                     computing_timer;

  SolverControl                                   solver_control;
  TrilinosWrappers::SolverDirect                  solver;

  const RightHandSide<dim>              rhs_function;
  const KInverse<dim>                   K_inverse_function;
  const DirichletBoundaryValues<dim>    Dirichlet_bc_function;
  const TrueSolution<dim>               true_solution;
  const TrueSolution_omega<dim_omega>   true_solution_omega;

  const RightHandSide_omega<dim_omega>              rhs_function_omega;
  const KInverse<dim_omega>                       	k_inverse_function;
  const DirichletBoundaryValues_omega<dim_omega>    Dirichlet_bc_function_omega;


  std::vector<Point<dim>> support_points;
  std::vector<Point<dim_omega>> support_points_omega;
  std::vector<Point<dim_omega>> unit_support_points_omega;
  unsigned int start_VectorField_omega;
  unsigned int start_Potential_omega;
  unsigned int start_Potential;

  const double g = 1;
};


template <int dim, int dim_omega>
LDGPoissonProblem<dim, dim_omega>::
LDGPoissonProblem(const unsigned int degree,
                  const unsigned int n_refine)
  :
  degree(degree),   
  n_refine(n_refine),
  fe( FESystem<dim>(FE_DGQ<dim>(degree), dim), FE_DGQ<dim>(degree),  FE_DGQ<dim>(degree), FE_DGQ<dim>(degree)),
  fe_omega( FESystem<dim_omega>(FE_DGQ<dim_omega>(degree), dim_omega), FE_DGQ<dim_omega>(degree)),
  dof_handler(triangulation),
  dof_handler_omega(triangulation_omega),
  pcout(std::cout),
  computing_timer(
                 pcout,
                  TimerOutput::summary,
                  TimerOutput::wall_times),
  solver_control(1),
  solver(solver_control),
  rhs_function(),
  Dirichlet_bc_function(),
  rhs_function_omega(),
  Dirichlet_bc_function_omega()
{
}

template <int dim, int dim_omega>
LDGPoissonProblem<dim, dim_omega>::
~LDGPoissonProblem()
{
  dof_handler.clear();
  dof_handler_omega.clear();
}

template <int dim, int dim_omega>
void
LDGPoissonProblem<dim, dim_omega>::
make_grid()
{
  TimerOutput::Scope t(computing_timer, "make grid");

  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(n_refine);

  typename Triangulation<dim>::cell_iterator
  cell = triangulation.begin(),
  endc = triangulation.end();
  for (; cell != endc; ++cell)
    {
      for (unsigned int face_no=0;
           face_no < GeometryInfo<dim>::faces_per_cell;
           face_no++)
        {
          if (cell->face(face_no)->at_boundary() )
            cell->face(face_no)->set_boundary_id(Dirichlet);
        }
    }
  GridGenerator::hyper_cube(triangulation_omega, -0.5, 0.5);
  triangulation_omega.refine_global(n_refine);

  typename Triangulation<dim_omega>::cell_iterator
  cell_omega = triangulation_omega.begin(),
  endc_omega = triangulation_omega.end();
  for (; cell_omega != endc_omega; ++cell_omega)
    {
      for (unsigned int face_no=0;
           face_no < GeometryInfo<dim_omega>::faces_per_cell;
           face_no++)
        {
          if (cell_omega->face(face_no)->at_boundary() )
            cell_omega->face(face_no)->set_boundary_id(Dirichlet);
        }
    }
}


template <int dim, int dim_omega>
void
LDGPoissonProblem<dim, dim_omega>::
make_dofs()
{
  TimerOutput::Scope t(computing_timer, "setup");

  dof_handler.distribute_dofs(fe);
  DoFRenumbering::component_wise(dof_handler);

  dof_handler_omega.distribute_dofs(fe_omega);
  DoFRenumbering::component_wise(dof_handler_omega);


/*  IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();


  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler,
                                          locally_relevant_dofs);
*/
  const std::vector<types::global_dof_index> dofs_per_component =
    DoFTools::count_dofs_per_fe_component(dof_handler);
    
unsigned int n_dofs_Potential =  dofs_per_component[dim + dim_omega];
     
          
/*
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof_handler,
                                       dsp);

  SparsityTools::distribute_sparsity_pattern(dsp,
                                            // dof_handler.locally_owned_dofs(),
                                            // MPI_COMM_WORLD,
                                             locally_relevant_dofs);


  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp);
                      // MPI_COMM_WORLD);


  solution.reinit(locally_relevant_dofs);
                                  // MPI_COMM_WORLD);

  system_rhs.reinit(locally_owned_dofs,
                    locally_relevant_dofs);
                    //MPI_COMM_WORLD,
                    //true);
                    */

  const unsigned int n_vector_field = dim * dofs_per_component[0] + dim_omega * dofs_per_component[dim];
  const unsigned int n_potential = dofs_per_component[dim + dim_omega] + dofs_per_component[dim + dim_omega + 1];

   /* for(unsigned int i = 0; i < dofs_per_component.size(); i++)
  std::cout<<"dofs_per_component " <<dofs_per_component[i]<<std::endl;*/

  cout << "Number of active cells : "
        << triangulation.n_global_active_cells()
        << std::endl
        << "Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << " (" << n_vector_field << " + " << n_potential << ")"
        << std::endl;

  start_VectorField_omega = dim * dofs_per_component[0];
  start_Potential_omega = n_vector_field + dofs_per_component[dim + dim_omega];
  start_Potential = n_vector_field;
  //std::cout<<" start_VectorField_omega "<< start_VectorField_omega<<" start_Potential_omega "<<start_Potential_omega<<" start_Potential "<<start_Potential<<std::endl;

  constraints.clear();
  constraints.close();

//  const std::vector<types::global_dof_index> block_sizes = {n_vector_field, n_potential};
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
  //BlockDynamicSparsityPattern dsp(block_sizes, block_sizes);
    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                       dsp);   
  //std::cout<<"dof_handler_omega.n_dofs() "<<dof_handler_omega.n_dofs()<<std::endl;
    const std::vector<types::global_dof_index> dofs_per_component_omega =
    DoFTools::count_dofs_per_fe_component(dof_handler_omega);
  unsigned int n_dofs_VectorField_omega = dofs_per_component_omega[0];
  unsigned int n_dofs_Potential_omega = dofs_per_component_omega[1];
 for(unsigned int i = start_VectorField_omega; i < start_VectorField_omega +n_dofs_VectorField_omega ; i++)
  {
     for(unsigned int j = start_VectorField_omega; j < start_VectorField_omega + n_dofs_VectorField_omega ; j++)
      {
        dsp.add(i,j);
      }
  }

    for(unsigned int i = start_Potential_omega; i < start_Potential_omega + n_dofs_Potential_omega ; i++)
  {
     for(unsigned int j = start_Potential_omega; j < start_Potential_omega + n_dofs_Potential_omega ; j++)
      {
        dsp.add(i,j);
      }
  }

  for(unsigned int i = start_Potential_omega; i < start_Potential_omega +n_dofs_Potential_omega ; i++)
  {
      for(unsigned int j = start_VectorField_omega; j < start_VectorField_omega + n_dofs_VectorField_omega ; j++)
      {
        dsp.add(i,j);
      }
  }
   for(unsigned int i = start_VectorField_omega; i < start_VectorField_omega + n_dofs_VectorField_omega ; i++)
  {
      for(unsigned int j = start_Potential_omega; j < start_Potential_omega + n_dofs_Potential_omega ; j++)
      {
        dsp.add(i,j);
      }
  }

//COUPLING
for(unsigned int i = start_Potential_omega; i < start_Potential_omega +n_dofs_Potential_omega ; i++)
  {
      for(unsigned int j = start_Potential; j < start_Potential + n_dofs_Potential ; j++)
      {
        dsp.add(i,j);
      }
  }

  for(unsigned int i = start_Potential; i < start_Potential + n_dofs_Potential ; i++)
  {
      for(unsigned int j = start_Potential_omega; j < start_Potential_omega + n_dofs_Potential_omega ; j++)
      {
        dsp.add(i,j);
      }
  }
        
                                  
  //DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints,false);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());


}

template <int dim, int dim_omega>
void
LDGPoissonProblem<dim, dim_omega>::
assemble_system()
{
  TimerOutput::Scope t(computing_timer, "assembly");

  QGauss<dim>         quadrature_formula(fe.degree+2);
  QGauss<dim-1>       face_quadrature_formula(fe.degree+2);


  const UpdateFlags update_flags  = update_values
                                    | update_gradients
                                    | update_quadrature_points
                                    | update_JxW_values;
const UpdateFlags update_flags_coupling  = update_values;

  const UpdateFlags face_update_flags =   update_values
                                          | update_normal_vectors
                                          | update_quadrature_points
                                          | update_JxW_values;



  const unsigned int dofs_per_cell = fe.dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index>
  local_neighbor_dof_indices(dofs_per_cell);


  FEValues<dim>           fe_values(fe, quadrature_formula, update_flags);

  FEFaceValues<dim>       fe_face_values(fe,face_quadrature_formula,
                                         face_update_flags);

  FEFaceValues<dim>       fe_neighbor_face_values(fe,
                                                  face_quadrature_formula,
                                                  face_update_flags);
  
  

  FullMatrix<double>      local_matrix(dofs_per_cell,dofs_per_cell);
  Vector<double>          local_vector(dofs_per_cell);


  FullMatrix<double>      vi_ui_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double>      vi_ue_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double>      ve_ui_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double>      ve_ue_matrix(dofs_per_cell, dofs_per_cell);

  const Mapping<dim> &mapping = fe_values.get_mapping();  
  {
    TimerOutput::Scope t(computing_timer, "assembly - Omega");
   
    /*support_points.resize(dof_handler.n_dofs());

    std::vector<Point<dim>> unit_support_points_FE_Q(fe.n_dofs_per_cell());

    std::vector<std::pair< std::pair< unsigned int, unsigned int >, unsigned int >> dof_table(dof_handler.n_dofs());
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
       std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
        std::vector<Point<dim>> cell_support_points(fe.n_dofs_per_cell());
        cell->get_dof_indices( local_dof_indices);
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

            support_points[local_dof_indices[i]] = cell_support_points[within_base_];
        }
    
    }*/

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


  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    {
      
        {

          local_matrix = 0;
          local_vector = 0;

          fe_values.reinit(cell);
          assemble_cell_terms(fe_values,
                              local_matrix,
                              local_vector,
                              K_inverse_function, 
                              rhs_function,
                              VectorField,
                              Potential);


          cell->get_dof_indices(local_dof_indices);

          for (unsigned int face_no=0;
               face_no< GeometryInfo<dim>::faces_per_cell;
               face_no++)
            {
              typename DoFHandler<dim>::face_iterator  face =
                cell->face(face_no);

              if (face->at_boundary() )
                {
                  fe_face_values.reinit(cell, face_no);

                  if (face->boundary_id() == Dirichlet)
                    {
                     // std::cout<<"bound"<<std::endl;
                      double h = cell->diameter();
                      assemble_Dirichlet_boundary_terms(fe_face_values,
                                                        local_matrix,
                                                        local_vector,
                                                        h,
                                                        Dirichlet_bc_function,
                                                        VectorField,
                                                        Potential);
                    }
                  else if (face->boundary_id() == Neumann)
                    {
                      assemble_Neumann_boundary_terms(fe_face_values,
                                                      local_matrix,
                                                      local_vector);
                    }
                  else
                    Assert(false, ExcNotImplemented() );
                }
              else
                {

                  Assert(cell->neighbor(face_no).state() ==
                         IteratorState::valid,
                         ExcInternalError());

                  typename DoFHandler<dim>::cell_iterator neighbor =
                    cell->neighbor(face_no);


              
                      if (cell->id() < neighbor->id())
                        {
                 
                          const unsigned int neighbor_face_no =
                            cell->neighbor_of_neighbor(face_no);

                          vi_ui_matrix = 0;
                          vi_ue_matrix = 0;
                          ve_ui_matrix = 0;
                          ve_ue_matrix = 0;

                          fe_face_values.reinit(cell, face_no);
                          fe_neighbor_face_values.reinit(neighbor,
                                                         neighbor_face_no);

                          double h = std::min(cell->diameter(),
                                              neighbor->diameter());

                        
                          assemble_flux_terms(fe_face_values,
                                              fe_neighbor_face_values,
                                              vi_ui_matrix,
                                              vi_ue_matrix,
                                              ve_ui_matrix,
                                              ve_ue_matrix,
                                              h,
                                              VectorField, 
                                              Potential);

                          neighbor->get_dof_indices(local_neighbor_dof_indices);

                          distribute_local_flux_to_global(
                            vi_ui_matrix,
                            vi_ue_matrix,
                            ve_ui_matrix,
                            ve_ue_matrix,
                            local_dof_indices,
                            local_neighbor_dof_indices);


                        }
                    
                }
            }


          constraints.distribute_local_to_global(local_matrix,
                                                 local_dof_indices,
                                                 system_matrix);

          constraints.distribute_local_to_global(local_vector,
                                                 local_dof_indices,
                                                 system_rhs);

        }
    }

  }



    // omega
  QGauss<dim_omega>         quadrature_formula_omega(fe.degree+2);
  QGauss<dim_omega-1>       face_quadrature_formula_omega(fe.degree+2);

  FEValues<dim_omega>      fe_values_omega(fe_omega, quadrature_formula_omega, update_flags);

  FEFaceValues<dim_omega>   fe_face_values_omega(fe_omega,face_quadrature_formula_omega,
                                         face_update_flags);

  FEFaceValues<dim_omega>   fe_neighbor_face_values_omega(fe_omega,
                                                  face_quadrature_formula_omega,
                                                  face_update_flags);

  const unsigned int dofs_per_cell_omega = fe_omega.dofs_per_cell;
  //std::cout<<"dofOmega "<<dofs_per_cell_omega<<std::endl;
  std::vector<types::global_dof_index> local_dof_indices_omega(dofs_per_cell_omega);
  std::vector<types::global_dof_index>
  local_neighbor_dof_indices_omega(dofs_per_cell_omega);


  FullMatrix<double>      local_matrix_omega(dofs_per_cell_omega,dofs_per_cell_omega);
  Vector<double>          local_vector_omega(dofs_per_cell_omega);


  FullMatrix<double>      vi_ui_matrix_omega(dofs_per_cell_omega, dofs_per_cell_omega);
  FullMatrix<double>      vi_ue_matrix_omega(dofs_per_cell_omega, dofs_per_cell_omega);
  FullMatrix<double>      ve_ui_matrix_omega(dofs_per_cell_omega, dofs_per_cell_omega);
  FullMatrix<double>      ve_ue_matrix_omega(dofs_per_cell_omega, dofs_per_cell_omega);



/*
    const Mapping<dim> &mapping = fe_values.get_mapping();  
    support_points.resize(dof_handler.n_dofs());

    std::vector<Point<dim>> unit_support_points_FE_Q(fe.n_dofs_per_cell());

    std::vector<std::pair< std::pair< unsigned int, unsigned int >, unsigned int >> dof_table(dof_handler.n_dofs());
    
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
       std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
        std::vector<Point<dim>> cell_support_points(fe.n_dofs_per_cell());
        cell->get_dof_indices( local_dof_indices);
        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        {
            dof_table[local_dof_indices[i]] = dof_handler.get_fe().system_to_base_index(i);
                const unsigned int base =
            dof_table[local_dof_indices[i]].first.first;

            const unsigned int within_base_  =
            dof_table[local_dof_indices[i]].second; // same as above
                

            unit_support_points_FE_Q =  dof_handler.get_fe().base_element(base).get_unit_support_points();
            for(unsigned int j = 0; j < unit_support_points_FE_Q.size(); j++)
               cell_support_points[j] = mapping.transform_unit_to_real_cell(cell, unit_support_points_FE_Q[j]);

            support_points[local_dof_indices[i]] = cell_support_points[within_base_];
        }
    
    }
*/

typename DoFHandler<dim_omega>::active_cell_iterator
 cell_omega = dof_handler_omega.begin_active(),
 endc_omega = dof_handler_omega.end();

{
  TimerOutput::Scope t(computing_timer, "assembly - omega");

 
  for (; cell_omega!=endc_omega; ++cell_omega)
    {
      


          local_matrix_omega = 0;
          local_vector_omega = 0;

          fe_values_omega.reinit(cell_omega);
          assemble_cell_terms(fe_values_omega,
                              local_matrix_omega,
                              local_vector_omega,
                              k_inverse_function,
                              rhs_function_omega,
                              VectorField_omega,
                              Potential_omega);


          cell_omega->get_dof_indices(local_dof_indices_omega);
          /*for(unsigned l = 0; l < local_dof_indices_omega.size(); l++)
            std::cout<<local_dof_indices_omega[l]<<" ";
          std::cout<<std::endl;*/
          dof_omega_to_Omega(dof_handler_omega, local_dof_indices_omega);
          /*for(unsigned l = 0; l < local_dof_indices_omega.size(); l++)
          {
              std::cout<<local_dof_indices_omega[l]<<" ";
             // local_dof_indices_omega[l] = 0;
          }
          */
            
       //   std::cout<<std::endl<<" ---ende-----"<<std::endl;
          

          for (unsigned int face_no_omega=0;
               face_no_omega< GeometryInfo<dim_omega>::faces_per_cell;
               face_no_omega++)
            {
           //   std::cout<<"face_no_omega "<<face_no_omega<<std::endl;
              typename DoFHandler<dim_omega>::face_iterator  face_omega =
                cell_omega->face(face_no_omega);

              if (face_omega->at_boundary() )
                {
                  fe_face_values_omega.reinit(cell_omega, face_no_omega);

                  if (face_omega->boundary_id() == Dirichlet)
                    {
                      double h = cell_omega->diameter();
                      assemble_Dirichlet_boundary_terms(fe_face_values_omega,
                                                        local_matrix_omega,
                                                        local_vector_omega,
                                                        h,
                                                        Dirichlet_bc_function_omega,
                                                       VectorField_omega,
                                                       Potential_omega);
                    }
                  /*else if (face_omega->boundary_id() == Neumann)
                    {
                      assemble_Neumann_boundary_terms(fe_face_values_omega,
                                                      local_matrix_omega,
                                                      local_vector_omega);
                    }*/
                  else
                    Assert(false, ExcNotImplemented() );
                }
              else
                {

                  Assert(cell_omega->neighbor(face_no_omega).state() ==
                         IteratorState::valid,
                         ExcInternalError());

                  typename DoFHandler<dim_omega>::cell_iterator neighbor_omega =
                    cell_omega->neighbor(face_no_omega);


              
                      if (cell_omega->id() < neighbor_omega->id())
                        {
                 
                          const unsigned int neighbor_face_no_omega =
                            cell_omega->neighbor_of_neighbor(face_no_omega);

                          vi_ui_matrix_omega = 0;
                          vi_ue_matrix_omega = 0;
                          ve_ui_matrix_omega = 0;
                          ve_ue_matrix_omega = 0;

                          fe_face_values_omega.reinit(cell_omega, face_no_omega);
                          fe_neighbor_face_values_omega.reinit(neighbor_omega,
                                                         neighbor_face_no_omega);

                          double h = std::min(cell_omega->diameter(),
                                              neighbor_omega->diameter());

                        
                          assemble_flux_terms(fe_face_values_omega,
                                              fe_neighbor_face_values_omega,
                                              vi_ui_matrix_omega,
                                              vi_ue_matrix_omega,
                                              ve_ui_matrix_omega,
                                              ve_ue_matrix_omega,
                                              h,
                                              VectorField_omega, 
                                              Potential_omega);
                         
                          neighbor_omega->get_dof_indices(local_neighbor_dof_indices_omega);
                          dof_omega_to_Omega(dof_handler_omega, local_neighbor_dof_indices_omega);
                      
                       

                        distribute_local_flux_to_global(
                            vi_ui_matrix_omega,
                            vi_ue_matrix_omega,
                            ve_ui_matrix_omega,
                            ve_ue_matrix_omega,
                            local_dof_indices_omega,
                            local_neighbor_dof_indices_omega);
                        }
                    
                }
            }


          constraints.distribute_local_to_global(local_matrix_omega,
                                                 local_dof_indices_omega,
                                                 system_matrix);

          constraints.distribute_local_to_global(local_vector_omega,
                                                 local_dof_indices_omega,
                                                 system_rhs);

        
    }
}


{
  TimerOutput::Scope t(computing_timer, "assembly - coupling");
    //coupling
  //std::cout<<"start Coupling"<<std::endl;
  FullMatrix<double>      V_U_matrix_coupling(dofs_per_cell, dofs_per_cell);
  FullMatrix<double>      v_U_matrix_coupling(dofs_per_cell_omega, dofs_per_cell);
  FullMatrix<double>      V_u_matrix_coupling(dofs_per_cell, dofs_per_cell_omega);
  FullMatrix<double>      v_u_matrix_coupling(dofs_per_cell_omega, dofs_per_cell_omega);

  cell_omega = dof_handler_omega.begin_active();
  endc_omega = dof_handler_omega.end();

  for (; cell_omega!=endc_omega; ++cell_omega)
  {
      fe_values_omega.reinit(cell_omega);
      cell_omega->get_dof_indices(local_dof_indices_omega);
      dof_omega_to_Omega(dof_handler_omega, local_dof_indices_omega);

      std::vector<Point<dim_omega>> quadrature_points_omega = fe_values_omega.get_quadrature_points();

      for(unsigned int p = 0; p < quadrature_points_omega.size(); p++)
      {
        Point<dim_omega> quadrature_point_omega = quadrature_points_omega[p];
       // std::cout<<quadrature_point_omega<<std::endl;
        Point<dim> quadrature_point;
       if(dim == 2)
          quadrature_point = Point<dim>(quadrature_point_omega[0], y_l);
        if(dim ==3)
         quadrature_point = Point<dim>(quadrature_point_omega[0], y_l, y_l);

        // GridTools::Cache<dim, dim> cache(triangulation, mapping);
        //auto cell_and_ref_point = GridTools::find_active_cell_around_point(cache, quadrature_point);//
       // auto cell = cell_and_ref_point.first;
       auto cell = GridTools::find_active_cell_around_point(dof_handler, quadrature_point);
       // std::cout<<"cell "<<cell<<std::endl;
        
        fe_values.reinit(cell);
        cell->get_dof_indices(local_dof_indices);

        Point<dim> quadrature_point_mapped_cell =  mapping.transform_real_to_unit_cell(cell,quadrature_point);
       // std::cout<<quadrature_point<<" | "<<quadrature_point_mapped_cell<< std::endl;
        
        std::vector<Point<dim>> my_quadrature_points = {quadrature_point_mapped_cell};
         std::vector<double> my_quadrature_weights = {1};
         const Quadrature<dim> my_quadrature_formula(my_quadrature_points, my_quadrature_weights);

        FEValues<dim> fe_values_coupling(fe, my_quadrature_formula, update_flags_coupling);
        fe_values_coupling.reinit(cell);


      /*  for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
            std::cout << "N_" << i << " = " << fe_values_coupling[Potential].value(i,0) << std::endl;*/
  	  V_U_matrix_coupling = 0;
      v_U_matrix_coupling = 0;
      V_u_matrix_coupling = 0;
      v_u_matrix_coupling = 0;
      for(unsigned int i = 0; i < dofs_per_cell; i++)
      {
         for(unsigned int j = 0; j < dofs_per_cell; j++)
         {  
          V_U_matrix_coupling(i,j) += g * fe_values_coupling[Potential].value(i,0) * fe_values_coupling[Potential].value(j,0) * fe_values_omega.JxW(p);
         }
      }
      constraints.distribute_local_to_global(V_U_matrix_coupling,
                                         local_dof_indices,
                                         local_dof_indices,
                                         system_matrix);

      for(unsigned int i = 0; i < dofs_per_cell_omega; i++)
      {
         for(unsigned int j = 0; j < dofs_per_cell; j++)
         {  
          v_U_matrix_coupling(i,j) += - g * fe_values_omega[Potential_omega].value(i,p) * fe_values_coupling[Potential].value(j,0) * fe_values_omega.JxW(p);
         }
      }
      constraints.distribute_local_to_global(v_U_matrix_coupling,
                                         local_dof_indices_omega,
                                         local_dof_indices,
                                         system_matrix);    

    
      for(unsigned int i = 0; i < dofs_per_cell; i++)
      {
         for(unsigned int j = 0; j < dofs_per_cell_omega; j++)
         {  
          V_u_matrix_coupling(i,j) += - g * fe_values_omega[Potential_omega].value(j,p) * fe_values_coupling[Potential].value(i,0) * fe_values_omega.JxW(p);
         }
      }
      constraints.distribute_local_to_global(V_u_matrix_coupling,                                      
                                         local_dof_indices,
                                         local_dof_indices_omega,
                                         system_matrix);    

      for(unsigned int i = 0; i < dofs_per_cell_omega; i++)
            {
              for(unsigned int j = 0; j < dofs_per_cell_omega; j++)
              {  
                v_u_matrix_coupling(i,j) += g * fe_values_omega[Potential_omega].value(j,p) * fe_values_omega[Potential_omega].value(i,p) * fe_values_omega.JxW(p);
              }
            }
            constraints.distribute_local_to_global(v_u_matrix_coupling,                                      
                                              local_dof_indices_omega,
                                              local_dof_indices_omega,
                                              system_matrix);    

        
        
        /*std::cout<<"cell "<<cell<<std::endl;
        std::cout << "The point (" << query_point[0] << ", " << query_point[1] << ") is in cell with vertices: " << std::endl;
        for (unsigned int v = 0; v < GeometryInfo<2>::vertices_per_cell; ++v)
        {
            std::cout << "  (" << cell->vertex(v)[0] << ", " << cell->vertex(v)[1] << ")" << std::endl;
        }*/
          /*typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();
         for (; cell!=endc; ++cell)
          {
            fe_values.reinit(cell);
            Point<dim> quadrature_point_mapped_cell =  mapping.transform_real_to_unit_cell(cell,quadrature_point);
            std::cout<<quadrature_point_omega<<" | "<<quadrature_point_mapped_cell<< std::endl;
            
            
            
            // std::vector<double> shape_values(fe.dofs_per_cell);
            // fe_values.get_function_values(shape_values);

            // std::cout << "Shape function values at point (" 
            //           << point << "):" << std::endl;
            // for (unsigned int i = 0; i < shape_values.size(); ++i)
            //     std::cout << "N_" << i << " = " << shape_values[i] << std::endl;
          }*/





       /* std::vector<Point<dim>> my_quadrature_points = {quadrature_point };
        std::vector<double> my_quadrature_weights = {1};
        const Quadrature<dim> my_quadrature_formula(my_quadrature_points, my_quadrature_weights);
        FEValues<dim> fe_values_coupling(fe, my_quadrature_formula, update_flags);

       
          typename DoFHandler<dim>::active_cell_iterator
          cell = dof_handler.begin_active(),
          endc = dof_handler.end();
          */

      }
    //std::cout<<std::endl;
  }
}










  //  std::cout<<"set ii "<<std::endl;

    for(unsigned int i = 0; i < dof_handler.n_dofs() ; i++)//dof_table.size()
    {
    //if(dof_table[i].first.first == 1 || dof_table[i].first.first == 3)
      {
        if(system_matrix.el(i,i) == 0 )
          {
          
            system_matrix.set(i,i,1);
          }
      }

    }


}



template<int dim, int dim_omega>
template<int _dim>
void
LDGPoissonProblem<dim, dim_omega>::dof_omega_to_Omega(const DoFHandler<_dim>  &dof_handler,
                                                      std::vector<types::global_dof_index> &local_dof_indices_omega)
{
      const std::vector<types::global_dof_index> dofs_per_component =
      DoFTools::count_dofs_per_fe_component(dof_handler);
  for (unsigned int i = 0; i < local_dof_indices_omega.size(); ++i)
  {
    const unsigned int base_i =
            dof_handler.get_fe().system_to_base_index(i).first.first;

    local_dof_indices_omega[i] = base_i == 0 ? local_dof_indices_omega[i] + start_VectorField_omega : local_dof_indices_omega[i] - dofs_per_component[0] + start_Potential_omega;
  }         
}





template<int dim, int dim_omega>
template<int _dim>
void
LDGPoissonProblem<dim, dim_omega>::
assemble_cell_terms(
  const FEValues<_dim>     &cell_fe,
  FullMatrix<double>      &cell_matrix,
  Vector<double>          &cell_vector,
  const TensorFunction<2, _dim>    &_K_inverse_function,
  const Function<_dim>    &_rhs_function,
  const FEValuesExtractors::Vector VectorField, 
  const FEValuesExtractors::Scalar Potential)
{
  const unsigned int dofs_per_cell = cell_fe.dofs_per_cell;
  const unsigned int n_q_points    = cell_fe.n_quadrature_points;


  std::vector<double>              rhs_values(n_q_points);
  std::vector<Tensor<2, _dim>>     K_inverse_values(n_q_points);


  _rhs_function.value_list(cell_fe.get_quadrature_points(),
                          rhs_values);
  _K_inverse_function.value_list(cell_fe.get_quadrature_points(),
                             K_inverse_values);

  for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          const Tensor<1, _dim> psi_i_field          = cell_fe[VectorField].value(i,q);
          const double         div_psi_i_field      = cell_fe[VectorField].divergence(i,q);
          const double         psi_i_potential      = cell_fe[Potential].value(i,q);
          const Tensor<1, _dim> grad_psi_i_potential = cell_fe[Potential].gradient(i,q);

          for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
              const Tensor<1, _dim> psi_j_field        = cell_fe[VectorField].value(j,q);
              const double         psi_j_potential    = cell_fe[Potential].value(j,q);

              cell_matrix(i,j)  += ( (psi_i_field * K_inverse_values[q] *  psi_j_field)
                                     -
                                     (div_psi_i_field * psi_j_potential)
                                     -
                                     (grad_psi_i_potential * psi_j_field)
                                   ) * cell_fe.JxW(q);
            }


          cell_vector(i) += psi_i_potential *
                            rhs_values[q] *
                            cell_fe.JxW(q);
        }
    }

}


template<int dim, int dim_omega>
template<int _dim>
void
LDGPoissonProblem<dim, dim_omega>::
assemble_Dirichlet_boundary_terms(
  const FEFaceValues<_dim>     &face_fe,
  FullMatrix<double>          &local_matrix,
  Vector<double>              &local_vector,
  const double                 &h,
  const Function<_dim>        &Dirichlet_bc_function,
  const FEValuesExtractors::Vector VectorField, 
  const FEValuesExtractors::Scalar Potential)
{
  const unsigned int dofs_per_cell     = face_fe.dofs_per_cell;
  const unsigned int n_q_points        = face_fe.n_quadrature_points;



  std::vector<double>     Dirichlet_bc_values(n_q_points);

  Dirichlet_bc_function.value_list(face_fe.get_quadrature_points(),
                                   Dirichlet_bc_values);

  for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          const Tensor<1, _dim> psi_i_field     = face_fe[VectorField].value(i,q);
          const double         psi_i_potential = face_fe[Potential].value(i,q);

          for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
              const Tensor<1, _dim> psi_j_field    = face_fe[VectorField].value(j,q);
              const double         psi_j_potential = face_fe[Potential].value(j,q);


              local_matrix(i,j) += psi_i_potential * (
                                     face_fe.normal_vector(q) *
                                     psi_j_field
                                     +
                                     (penalty/h) *
                                     psi_j_potential) *
                                   face_fe.JxW(q);

            }


          local_vector(i) += (-1.0 * psi_i_field *
                              face_fe.normal_vector(q)
                              +
                              (penalty/h) *
                              psi_i_potential) *
                             Dirichlet_bc_values[q] *
                             face_fe.JxW(q);
        }
    }
}


template<int dim, int dim_omega>
void
LDGPoissonProblem<dim, dim_omega>::
assemble_Neumann_boundary_terms(
  const FEFaceValues<dim>     &face_fe,
  FullMatrix<double>          &local_matrix,
  Vector<double>              &local_vector)
{
  const unsigned int dofs_per_cell = face_fe.dofs_per_cell;
  const unsigned int n_q_points    = face_fe.n_quadrature_points;


  std::vector<double >    Neumann_bc_values(n_q_points);

  for (unsigned int q=0; q<n_q_points; ++q)
    {
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        {
          const Tensor<1, dim> psi_i_field     = face_fe[VectorField].value(i,q);
          const double         psi_i_potential = face_fe[Potential].value(i,q);

          for (unsigned int j=0; j<dofs_per_cell; ++j)
            {

              const double    psi_j_potential = face_fe[Potential].value(j,q);

             
              local_matrix(i,j) += psi_i_field *
                                   face_fe.normal_vector(q) *
                                   psi_j_potential *
                                   face_fe.JxW(q);

            }

          local_vector(i) +=  -psi_i_potential *
                              Neumann_bc_values[q] *
                              face_fe.JxW(q);
        }
    }
}


template<int dim, int dim_omega>
template<int _dim>
void
LDGPoissonProblem<dim, dim_omega>::
assemble_flux_terms(
  const FEFaceValuesBase<_dim>     &fe_face_values,
  const FEFaceValuesBase<_dim>     &fe_neighbor_face_values,
  FullMatrix<double>              &vi_ui_matrix,
  FullMatrix<double>              &vi_ue_matrix,
  FullMatrix<double>              &ve_ui_matrix,
  FullMatrix<double>              &ve_ue_matrix,
  const double                     &h,
  const FEValuesExtractors::Vector VectorField, 
  const FEValuesExtractors::Scalar Potential
  )
{
  const unsigned int n_face_points      = fe_face_values.n_quadrature_points;
  const unsigned int dofs_this_cell     = fe_face_values.dofs_per_cell;
  const unsigned int dofs_neighbor_cell = fe_neighbor_face_values.dofs_per_cell;


  Point<_dim> beta;
  for (int i=0; i<_dim; ++i)
    beta(i) = 1.0;
  beta /= sqrt(beta.square() );

 // std::cout<<"penalty "<<penalty<<" h "<<h<<std::endl;


  for (unsigned int q=0; q<n_face_points; ++q)
    {
      for (unsigned int i=0; i<dofs_this_cell; ++i)
        {
          const Tensor<1,_dim>  psi_i_field_minus  =
            fe_face_values[VectorField].value(i,q);
          const double psi_i_potential_minus  =
            fe_face_values[Potential].value(i,q);

          for (unsigned int j=0; j<dofs_this_cell; ++j)
            {
              const Tensor<1,_dim> psi_j_field_minus   =
                fe_face_values[VectorField].value(j,q);
              const double psi_j_potential_minus  =
                fe_face_values[Potential].value(j,q);

           
              vi_ui_matrix(i,j)   += ( 0.5 * (
                                        psi_i_field_minus *
                                        fe_face_values.normal_vector(q) *
                                        psi_j_potential_minus
                                        +
                                        psi_i_potential_minus *
                                        fe_face_values.normal_vector(q) *
                                        psi_j_field_minus )

                                     +
                                      beta *
                                      psi_i_field_minus *
                                      psi_j_potential_minus
                                      -
                                      beta *
                                      psi_i_potential_minus *
                                      psi_j_field_minus
                                
                                        + (penalty/h) *
                                        psi_j_potential_minus *
                                        psi_i_potential_minus
                                    
                                     ) *
                                     fe_face_values.JxW(q);
            }

          for (unsigned int j=0; j<dofs_neighbor_cell; ++j)
            {
              const Tensor<1,_dim> psi_j_field_plus    =
                fe_neighbor_face_values[VectorField].value(j,q);
              const double            psi_j_potential_plus        =
                fe_neighbor_face_values[Potential].value(j,q);

              // We compute the flux matrix where the test function is
              // from the interior of this elements face and solution
              // function is taken from the exterior. 
              vi_ue_matrix(i,j) +=  ( 0.5 * (
                                       psi_i_field_minus *
                                       fe_face_values.normal_vector(q) *
                                       psi_j_potential_plus
                                       +
                                       psi_i_potential_minus *
                                       fe_face_values.normal_vector(q) *
                                       psi_j_field_plus )                          
                                                                        
                                    -
                                     beta *
                                     psi_i_field_minus *
                                     psi_j_potential_plus
                                     +
                                     beta *
                                     psi_i_potential_minus *
                                     psi_j_field_plus                                   
                                     
                                     - 
                                     (penalty/h) *
                                     psi_i_potential_minus *
                                     psi_j_potential_plus
                                   ) *
                                   fe_face_values.JxW(q);
            }
        }

      for (unsigned int i=0; i<dofs_neighbor_cell; ++i)
        {
          const Tensor<1,_dim>  psi_i_field_plus =
            fe_neighbor_face_values[VectorField].value(i,q);
          const double         psi_i_potential_plus =
            fe_neighbor_face_values[Potential].value(i,q);

          for (unsigned int j=0; j<dofs_this_cell; ++j)
            {
              const Tensor<1,_dim> psi_j_field_minus               =
                fe_face_values[VectorField].value(j,q);
              const double        psi_j_potential_minus       =
                fe_face_values[Potential].value(j,q);


              // We compute the flux matrix where the test function is
              // from the exterior of this elements face and solution
              // function is taken from the interior. 
              ve_ui_matrix(i,j) +=  0;
            ve_ui_matrix(i,j) += (-0.5 * (
                                       psi_i_field_plus *
                                       fe_face_values.normal_vector(q) *
                                       psi_j_potential_minus
                                       +
                                       psi_i_potential_plus *
                                       fe_face_values.normal_vector(q) *
                                       psi_j_field_minus)
                                     -
                                     beta *
                                     psi_i_field_plus *
                                     psi_j_potential_minus
                                     +
                                     beta *
                                     psi_i_potential_plus *
                                     psi_j_field_minus
                                     -
                                     (penalty/h) *
                                     psi_i_potential_plus *
                                     psi_j_potential_minus
                                    ) *
                                    fe_face_values.JxW(q);
            }

          for (unsigned int j=0; j<dofs_neighbor_cell; ++j)
            {
              const Tensor<1,_dim> psi_j_field_plus =
                fe_neighbor_face_values[VectorField].value(j,q);
              const double        psi_j_potential_plus =
                fe_neighbor_face_values[Potential].value(j,q);

              // And lastly we compute the flux matrix where the test
              // function and solution function are taken from the exterior
              // cell to this face.  
              ve_ue_matrix(i,j) += 0;
              ve_ue_matrix(i,j) +=    (-0.5 * (
                                         psi_i_field_plus *
                                         fe_face_values.normal_vector(q) *
                                         psi_j_potential_plus
                                         +
                                         psi_i_potential_plus *
                                         fe_face_values.normal_vector(q) *
                                         psi_j_field_plus )
                                       +
                                       beta *
                                       psi_i_field_plus *
                                       psi_j_potential_plus
                                       -
                                       beta *
                                       psi_i_potential_plus *
                                       psi_j_field_plus
                                       +
                                       (penalty/h) *
                                       psi_i_potential_plus *
                                       psi_j_potential_plus
                                      ) *
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
template<int dim, int dim_omega>
void
LDGPoissonProblem<dim, dim_omega>::
distribute_local_flux_to_global(
   FullMatrix<double> &vi_ui_matrix,
   FullMatrix<double> &vi_ue_matrix,
   FullMatrix<double> &ve_ui_matrix,
   FullMatrix<double> &ve_ue_matrix,
  const std::vector<types::global_dof_index> &local_dof_indices,
  const std::vector<types::global_dof_index> &local_neighbor_dof_indices)
{
 /* std::cout<<"distribute_local_flux_to_global"<<std::endl;
    for(unsigned int i = 0; i < local_neighbor_dof_indices.size(); i++)
  {
    std::cout<<local_dof_indices[i]<<" "<<local_neighbor_dof_indices[i]<< " | ";
  }                     
 //std::cout<<std::endl<<"values"<<std::endl;   
*/

  for(unsigned int i = 0; i < local_dof_indices.size(); i++)
  {
    for(unsigned int j = 0; j < local_dof_indices.size(); j++)
    {
    //std::cout<<vi_ui_matrix(i,j)<<" "<<isfinite(vi_ui_matrix(i,j))<<" | ";
   //if(vi_ui_matrix(i,j)> 10)
  //  vi_ui_matrix.set(i,j,1);
    }
  }
  //std::cout<<std::endl;
  constraints.distribute_local_to_global(vi_ui_matrix,
                                         local_dof_indices,
                                         system_matrix);

  for(unsigned int i = 0; i < local_dof_indices.size(); i++)
  {
    for(unsigned int j = 0; j < local_neighbor_dof_indices.size(); j++)
    {
     // if(vi_ue_matrix(i,j)> 10)
   //  std::cout<<vi_ue_matrix(i,j)<<" "<<isfinite(vi_ui_matrix(i,j))<<" | ";
    //vi_ue_matrix.set(i,j, 1);
    }
    
  }
  //std::cout<<std::endl;
  constraints.distribute_local_to_global(vi_ue_matrix,
                                         local_dof_indices,
                                         local_neighbor_dof_indices,
                                         system_matrix);

  for(unsigned int i = 0; i < local_neighbor_dof_indices.size(); i++)
  {
    for(unsigned int j = 0; j < local_dof_indices.size(); j++)
    {
  //  std::cout<<ve_ui_matrix(i,j)<<" ";
 //   ve_ui_matrix.set(i,j, 1);
    }
  }
  //std::cout<<std::endl;
  constraints.distribute_local_to_global(ve_ui_matrix,
                                         local_neighbor_dof_indices,
                                         local_dof_indices,
                                         system_matrix);
  for(unsigned int i = 0; i < local_dof_indices.size(); i++)
  {
    for(unsigned int j = 0; j < local_dof_indices.size(); j++)
    {
   // std::cout<<ve_ue_matrix(i,j)<<" ";
   // ve_ue_matrix.set(i,j, 1);
     }
  }
 //std::cout<<std::endl;
  constraints.distribute_local_to_global(ve_ue_matrix,
                                         local_neighbor_dof_indices,
                                         system_matrix);
}

template<int dim, int dim_omega>
std::array<double, 4> LDGPoissonProblem<dim, dim_omega>::compute_errors() const
  {
    const ComponentSelectFunction<dim> potential_mask(dim + 1, dim + dim_omega +2);
    const ComponentSelectFunction<dim> vectorfield_mask(std::make_pair(0, dim),
                                                     dim + dim_omega + 2); 

    Vector<double> cellwise_errors(triangulation.n_active_cells());

    const QTrapezoid<1>  q_trapez;
    const QIterated<dim> quadrature(q_trapez, degree + 2);


    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      true_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &potential_mask);
    const double potential_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      true_solution,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::L2_norm,
                                      &vectorfield_mask);
    const double vectorfield_l2_error =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::L2_norm);

    std::cout << "Errors: ||e_potential||_L2 = " << potential_l2_error
              << ",   ||e_vectorfield||_L2 = " << vectorfield_l2_error << std::endl;



    const ComponentSelectFunction<dim_omega> potential_mask_omega(dim_omega, dim_omega + 1);
    const ComponentSelectFunction<dim_omega> vectorfield_mask_omega(std::make_pair(0, dim_omega),
                                                      dim_omega + 1); 
    Vector<double> cellwise_errors_omega(triangulation_omega.n_active_cells());

    const QTrapezoid<1>  q_trapez_omega;
    const QIterated<dim_omega> quadrature_omega(q_trapez_omega, degree + 2);

    VectorTools::integrate_difference(dof_handler_omega,
                                      solution_omega,
                                      true_solution_omega,
                                      cellwise_errors_omega,
                                      quadrature_omega,
                                      VectorTools::L2_norm,
                                      &potential_mask_omega);
    const double potential_l2_error_omega =
      VectorTools::compute_global_error(triangulation_omega,
                                        cellwise_errors_omega,
                                        VectorTools::L2_norm);

    VectorTools::integrate_difference(dof_handler_omega,
                                      solution_omega,
                                      true_solution_omega,
                                      cellwise_errors_omega,
                                      quadrature_omega,
                                      VectorTools::L2_norm,
                                      &vectorfield_mask_omega);
    const double vectorfield_l2_error_omega =
      VectorTools::compute_global_error(triangulation_omega,
                                        cellwise_errors_omega,
                                        VectorTools::L2_norm);

    std::cout << "Errors: ||e_potential_omega||_L2 = " << potential_l2_error_omega
              << ",   ||e_vectorfield_omega||_L2 = " << vectorfield_l2_error_omega << std::endl;

    
    return std::array<double, 4>{{potential_l2_error, vectorfield_l2_error, potential_l2_error_omega, vectorfield_l2_error_omega}};

    
  }


// @sect4{solve}
// As mentioned earlier I used a direct solver to solve
// the linear system of equations resulting from the LDG
// method applied to the Poisson equation. One could also
// use a iterative sovler, however, we then need to use
// a preconditoner and that was something I did not wanted
// to get into. For information on preconditioners
// for the LDG Method see this
// <a href="http://epubs.siam.org/doi/abs/10.1137/S1064827502410657  [Titel anhand dieser DOI in Citavi-Projekt übernehmen] ">
// paper</a>. The uses of a direct sovler here is
// somewhat of a limitation.  The built-in distributed
// direct solver in Trilinos reduces everything to one
// processor, solves the system and then distributes
// everything back out to the other processors.  However,
// by linking to more advanced direct sovlers through
// Trilinos one can accomplish fully distributed computations
// and not much about the following function calls will
// change.
template<int dim, int dim_omega>
void
LDGPoissonProblem<dim, dim_omega>::
solve()
{
  TimerOutput::Scope t(computing_timer, "solve");
        std::cout << "Solving linear system... ";
      Timer timer;
      
      SparseDirectUMFPACK A_direct;
  
      solution = system_rhs;
      A_direct.solve(system_matrix, solution);
  

    /*  const unsigned int max_iterations = solution.size();
      SolverControl      solver_control(max_iterations);
      SolverCG<>         solver(solver_control);
      solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());*/


      timer.stop();
      std::cout << "done (" << timer.cpu_time() << "s)" << std::endl;
/*
  // As in step-40 in order to perform a linear solve
  // we need solution vector where there is no overlap across
  // the processors and we create this by instantiating
  // <code>completely_distributed_solution</code> solution
  // vector using
  // the copy constructor on the global system right hand
  // side vector which itself is completely distributed vector.
  TrilinosWrappers::MPI::Vector
  completely_distributed_solution(system_rhs);

  // Now we can preform the solve on the completeley distributed
  // right hand side vector, system matrix and the completely
  // distributed solution.
  solver.solve(system_matrix,
               completely_distributed_solution,
               system_rhs);

  // We now distribute the constraints of our system onto the
  // completely solution vector, but in our case with the LDG
  // method there are none.
  constraints.distribute(completely_distributed_solution);

  // Lastly we copy the completely distributed solution vector,
  // <code>completely_distributed_solution</code>,
  // to solution vector which has some overlap between
  // processors, <code>solution</code>.
  // We need the overlapped portions of our solution
  // in order to be able to do computations using the solution
  // later in the code or in post processing.
  solution = completely_distributed_solution;

*/
    const std::vector<types::global_dof_index> dofs_per_component_omega =
      DoFTools::count_dofs_per_fe_component(dof_handler_omega);
   // std::cout<<"nof compoent "<<dofs_per_component_omega.size()<<std::endl;

    solution_omega.reinit(dofs_per_component_omega[0] + dofs_per_component_omega[1]);
    for(unsigned int i = 0; i < dofs_per_component_omega[0]; i++)
      solution_omega[i] = solution[start_VectorField_omega + i];

    for(unsigned int i = 0; i < dofs_per_component_omega[1]; i++)
      solution_omega[dofs_per_component_omega[0]+ i] = solution[start_Potential_omega + i];

}

// @sect4{output_results}
// This function deals with the writing of the reuslts in parallel
// to disk.  It is almost exactly the same as
// in step-40 and we wont go into it.  It is noteworthy
// that in step-40 the output is only the scalar solution,
// while in our situation, we are outputing both the scalar
// solution as well as the vector field solution. The only
// difference between this function and the one in step-40
// is in the <code>solution_names</code> vector where we have to add
// the gradient dimensions.  Everything else is taken care
// of by the deal.ii library!
template<int dim, int dim_omega>
void
LDGPoissonProblem<dim, dim_omega>::
output_results()    const
{
  std::vector<std::string> solution_names;
  switch (dim)
    {
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
      Assert(false, ExcNotImplemented() );
    }

  DataOut<dim>    data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution,
                           solution_names);

  /*Vector<float>   subdomain(triangulation.n_active_cells());

  for (unsigned int i=0; i<subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();

  data_out.add_data_vector(subdomain,"subdomain");*/

  data_out.build_patches();



  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);

  

    

//-----omega-----------
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
    
}


template<int dim, int dim_omega>
std::array<double,4>
LDGPoissonProblem<dim, dim_omega>::
run()
{
  penalty = 1;
  make_grid();
  make_dofs();
  assemble_system();
  solve();
  std::array<double, 4> results_array = compute_errors();
  output_results();
  return results_array;
}


int main(int argc, char *argv[])
{

  LDGPoissonProblem<dimension_Omega, 1> *LDGPoissonCoupled;

  const unsigned int p_degree[1] = {1};
  constexpr unsigned int p_degree_size = sizeof(p_degree) / sizeof(p_degree[0]);
  const unsigned int refinement[5] = {2,3,4,5 ,6 };
  constexpr unsigned int refinement_size =
      sizeof(refinement) / sizeof(refinement[0]);

  std::array<double, 4> results[p_degree_size][refinement_size];

  std::vector<std::string> solution_names = {"Q_Omega", "U_Omega", "q_omega", "u_omega"};
  for (unsigned int r = 0; r < refinement_size; r++) {
    for (unsigned int p = 0; p < p_degree_size; p++) {
      LDGPoissonCoupled = new  LDGPoissonProblem<dimension_Omega, 1>(p_degree[p], refinement[r]);
      std::array<double, 4> arr = LDGPoissonCoupled->run();
      results[p][r] = arr;

      delete LDGPoissonCoupled;
    }
  }
  std::cout << "--------" << std::endl;
  std::ofstream myfile;
  myfile.open("convergence_results.txt");
  for (unsigned int f = 0; f < 4; f++) {
    myfile <<solution_names[f]<<"\n";
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
