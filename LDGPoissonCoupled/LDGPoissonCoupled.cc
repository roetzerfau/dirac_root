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


#include "Functions.cc"

using namespace dealii;

const FEValuesExtractors::Vector VectorField(0);
const FEValuesExtractors::Scalar Potential(3);
const unsigned int dimension_gap = 1;


template <int dim, int dim_omega>
class LDGPoissonProblem
{

public:
  LDGPoissonProblem(const unsigned int degree,
                    const unsigned int n_refine);

  ~LDGPoissonProblem();

  void run();


private:
  void make_grid();

  void make_dofs();

  void assemble_system();

  template<int _dim>
  void assemble_cell_terms(const FEValues<_dim>   &cell_fe,
                           FullMatrix<double>     &cell_matrix,
                           Vector<double>         &cell_vector, 
                           const Function<_dim>   &_rhs_function,
                           const FEValuesExtractors::Vector VectorField, 
                           const FEValuesExtractors::Scalar Potential);

  void assemble_Neumann_boundary_terms(const FEFaceValues<dim>    &face_fe,
                                       FullMatrix<double>         &local_matrix,
                                       Vector<double>             &local_vector);

  void assemble_Dirichlet_boundary_terms(const FEFaceValues<dim>  &face_fe,
                                         FullMatrix<double>       &local_matrix,
                                         Vector<double>           &local_vector,
                                         const double              &h);

  void assemble_flux_terms(const FEFaceValuesBase<dim>  &fe_face_values,
                           const FEFaceValuesBase<dim>  &fe_neighbor_face_values,
                           FullMatrix<double>           &vi_ui_matrix,
                           FullMatrix<double>           &vi_ue_matrix,
                           FullMatrix<double>           &ve_ui_matrix,
                           FullMatrix<double>           &ve_ue_matrix,
                           const double                  &h);

  void distribute_local_flux_to_global(
    const FullMatrix<double> &vi_ui_matrix,
    const FullMatrix<double> &vi_ue_matrix,
    const FullMatrix<double> &ve_ui_matrix,
    const FullMatrix<double> &ve_ue_matrix,
    const std::vector<types::global_dof_index> &local_dof_indices,
    const std::vector<types::global_dof_index> &local_neighbor_dof_indices);

  void solve();

  void compute_errors() const;
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

  SparsityPattern                                 sparsity_pattern;

  TrilinosWrappers::SparseMatrix                  system_matrix;
  TrilinosWrappers::MPI::Vector                   solution;
  TrilinosWrappers::MPI::Vector                   system_rhs;

  ConditionalOStream                              pcout;
  TimerOutput                                     computing_timer;

  SolverControl                                   solver_control;
  TrilinosWrappers::SolverDirect                  solver;

  const RightHandSide<dim>              rhs_function;
  const DirichletBoundaryValues<dim>    Dirichlet_bc_function;
  const TrueSolution<dim>               true_solution;

  std::vector<Point<dim>> support_points;
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
  computing_timer(MPI_COMM_WORLD,
                 pcout,
                  TimerOutput::summary,
                  TimerOutput::wall_times),
  solver_control(1),
  solver(solver_control),
  rhs_function(),
  Dirichlet_bc_function()
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
  GridGenerator::hyper_cube(triangulation, 0, 1);
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
      GridGenerator::hyper_cube(triangulation_omega, 0, 1);
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


  IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();


  IndexSet locally_relevant_dofs;
  DoFTools::extract_locally_relevant_dofs(dof_handler,
                                          locally_relevant_dofs);

  const std::vector<types::global_dof_index> dofs_per_component =
    DoFTools::count_dofs_per_fe_component(dof_handler);


  constraints.clear();
  constraints.close();


  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_flux_sparsity_pattern(dof_handler,
                                       dsp);

  SparsityTools::distribute_sparsity_pattern(dsp,
                                             dof_handler.locally_owned_dofs(),
                                             MPI_COMM_WORLD,
                                             locally_relevant_dofs);


  system_matrix.reinit(locally_owned_dofs,
                       locally_owned_dofs,
                       dsp,
                       MPI_COMM_WORLD);


  solution.reinit(locally_relevant_dofs,
                                   MPI_COMM_WORLD);

  system_rhs.reinit(locally_owned_dofs,
                    locally_relevant_dofs,
                    MPI_COMM_WORLD,
                    true);

  const unsigned int n_vector_field = dim * dofs_per_component[0] + dim_omega * dofs_per_component[dim];
  const unsigned int n_potential = dofs_per_component[dim + dim_omega] + dofs_per_component[dim + dim_omega + 1];

  cout << "Number of active cells : "
        << triangulation.n_global_active_cells()
        << std::endl
        << "Number of degrees of freedom: "
        << dof_handler.n_dofs()
        << " (" << n_vector_field << " + " << n_potential << ")"
        << std::endl;
}

template <int dim, int dim_omega>
void
LDGPoissonProblem<dim, dim_omega>::
assemble_system()
{
  TimerOutput::Scope t(computing_timer, "assembly");

  QGauss<dim>         quadrature_formula(fe.degree+1);
  QGauss<dim-1>       face_quadrature_formula(fe.degree+1);


  const UpdateFlags update_flags  = update_values
                                    | update_gradients
                                    | update_quadrature_points
                                    | update_JxW_values;

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
  
  const Mapping<dim> &mapping = fe_values.get_mapping();      

  FullMatrix<double>      local_matrix(dofs_per_cell,dofs_per_cell);
  Vector<double>          local_vector(dofs_per_cell);


  FullMatrix<double>      vi_ui_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double>      vi_ue_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double>      ve_ui_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double>      ve_ue_matrix(dofs_per_cell, dofs_per_cell);

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
           /* const unsigned int multiplicity =
            dof_table[local_dof_indices[i]].first.second;*/
            const unsigned int within_base_  =
            dof_table[local_dof_indices[i]].second; // same as above
                

            unit_support_points_FE_Q =  dof_handler.get_fe().base_element(base).get_unit_support_points();
            for(unsigned int j = 0; j < unit_support_points_FE_Q.size(); j++)
               cell_support_points[j] = mapping.transform_unit_to_real_cell(cell, unit_support_points_FE_Q[j]);

            support_points[local_dof_indices[i]] = cell_support_points[within_base_];
        }
    
    }

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

                      double h = cell->diameter();
                      assemble_Dirichlet_boundary_terms(fe_face_values,
                                                        local_matrix,
                                                        local_vector,
                                                        h);
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
                                              h);

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





    // omega
  QGauss<dim_omega>         quadrature_formula_omega(fe.degree+1);
  QGauss<dim_omega-1>       face_quadrature_formula_omega(fe.degree+1);

  FEValues<dim_omega>      fe_values_omega(fe_omega, quadrature_formula_omega, update_flags);

  FEFaceValues<dim_omega>   fe_face_values_omega(fe_omega,face_quadrature_formula_omega,
                                         face_update_flags);

  FEFaceValues<dim_omega>   fe_neighbor_face_values_omega(fe_omega,
                                                  face_quadrature_formula_omega,
                                                  face_update_flags);

  const unsigned int dofs_per_cell_omega = fe_omega.dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices_omega(dofs_per_cell_omega);
  std::vector<types::global_dof_index>
  local_neighbor_dof_indices_omega(dofs_per_cell_omega);


  FullMatrix<double>      local_matrix_omega(dofs_per_cell_omega,dofs_per_cell_omega);
  Vector<double>          local_vector_omega(dofs_per_cell_omega);


  FullMatrix<double>      vi_ui_matrix_omega(dofs_per_cell_omega, dofs_per_cell_omega);
  FullMatrix<double>      vi_ue_matrix_omega(dofs_per_cell_omega, dofs_per_cell_omega);
  FullMatrix<double>      ve_ui_matrix_omega(dofs_per_cell_omega, dofs_per_cell_omega);
  FullMatrix<double>      ve_ue_matrix_omega(dofs_per_cell_omega, dofs_per_cell_omega);

 // typename DoFHandler<dim>::active_cell_iterator
 // cell_omega = dof_handler_omega.begin_active(),
 // endc_omega = dof_handler_omega.end();

//   for (; cell_omega!=endc_omega; ++cell_omega)
//     {
      
//         {

//           local_matrix_omega = 0;
//           local_vector_omega = 0;

//           fe_values_omega.reinit(cell_omega);
//           assemble_cell_terms_omega(fe_values_omega,
//                               local_matrix_omega,
//                               local_vector_omega);


//           cell_omega->get_dof_indices(local_dof_indices_omega);

//           for (unsigned int face_no_omega=0;
//                face_no_omega< GeometryInfo<dim_omega>::faces_per_cell;
//                face_no_omega++)
//             {
//               typename DoFHandler<dim_omega>::face_iterator  face_omega =
//                 cell_omega->face(face_no);

//               if (face_omega->at_boundary() )
//                 {
//                   fe_face_values_omega.reinit(cell_omega, face_no_omega);

//                   if (face_omega->boundary_id() == Dirichlet)
//                     {

//                       double h = cell_omega->diameter();
//                       assemble_Dirichlet_boundary_terms_omega(fe_face_values_omega,
//                                                         local_matrix_omega,
//                                                         local_vector_omega,
//                                                         h);
//                     }
//                   /*else if (face_omega->boundary_id() == Neumann)
//                     {
//                       assemble_Neumann_boundary_terms(fe_face_values_omega,
//                                                       local_matrix_omega,
//                                                       local_vector_omega);
//                     }*/
//                   else
//                     Assert(false, ExcNotImplemented() );
//                 }
//               else
//                 {

//                   Assert(cell_omega->neighbor(face_no_omega).state() ==
//                          IteratorState::valid,
//                          ExcInternalError());

//                   typename DoFHandler<dim>::cell_iterator neighbor_omega =
//                     cell_omega->neighbor(face_no_omega);


              
//                       if (cell_omega->id() < neighbor_omega->id())
//                         {
                 
//                           const unsigned int neighbor_face_no_omega =
//                             cell_omega->neighbor_of_neighbor(face_no_omega);

//                           vi_ui_matrix_omega = 0;
//                           vi_ue_matrix_omega = 0;
//                           ve_ui_matrix_omega = 0;
//                           ve_ue_matrix_omega = 0;

//                           fe_face_values_omega.reinit(cell_omega, face_no_omega);
//                           fe_neighbor_face_values_omega.reinit(neighbor_omega,
//                                                          neighbor_face_no_omega);

//                           double h = std::min(cell_omega->diameter(),
//                                               neighbor_omega->diameter());

                        
//                           assemble_flux_terms_omega(fe_face_values_omega,
//                                               fe_neighbor_face_values_omega,
//                                               vi_ui_matrix_omega,
//                                               vi_ue_matrix_omega,
//                                               ve_ui_matrix_omega,
//                                               ve_ue_matrix_omega,
//                                               h);

//                           neighbor_omega->get_dof_indices(local_neighbor_dof_indices_omega);

//                          /* distribute_local_flux_to_global(
//                             vi_ui_matrix_omega,
//                             vi_ue_matrix_omega,
//                             ve_ui_matrix_omega,
//                             ve_ue_matrix_omega,
//                             local_dof_indices_omega,
//                             local_neighbor_dof_indices_omega);*/


//                         }
                    
//                 }
//             }

// /*
//           constraints.distribute_local_to_global(local_matrix_omega,
//                                                  local_dof_indices_omega,
//                                                  system_matrix_omega);

//           constraints.distribute_local_to_global(local_vector_omega,
//                                                  local_dof_indices_omega,
//                                                  system_rhs_omega);*/

//         }
//     }














    std::cout<<"set ii "<<std::endl;

    for(unsigned int i = 0; i < dof_table.size(); i++)
    {
    if(dof_table[i].first.first == 1 || dof_table[i].first.first == 3)
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
LDGPoissonProblem<dim, dim_omega>::
assemble_cell_terms(
  const FEValues<_dim>     &cell_fe,
  FullMatrix<double>      &cell_matrix,
  Vector<double>          &cell_vector,
  const Function<_dim>    &_rhs_function,
  const FEValuesExtractors::Vector VectorField, 
  const FEValuesExtractors::Scalar Potential)
{
  const unsigned int dofs_per_cell = cell_fe.dofs_per_cell;
  const unsigned int n_q_points    = cell_fe.n_quadrature_points;


  std::vector<double>              rhs_values(n_q_points);


  _rhs_function.value_list(cell_fe.get_quadrature_points(),
                          rhs_values);


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

              cell_matrix(i,j)  += ( (psi_i_field * psi_j_field)
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
void
LDGPoissonProblem<dim, dim_omega>::
assemble_Dirichlet_boundary_terms(
  const FEFaceValues<dim>     &face_fe,
  FullMatrix<double>          &local_matrix,
  Vector<double>              &local_vector,
  const double                 &h)
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
          const Tensor<1, dim> psi_i_field     = face_fe[VectorField].value(i,q);
          const double         psi_i_potential = face_fe[Potential].value(i,q);

          for (unsigned int j=0; j<dofs_per_cell; ++j)
            {
              const Tensor<1, dim> psi_j_field    = face_fe[VectorField].value(j,q);
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
void
LDGPoissonProblem<dim, dim_omega>::
assemble_flux_terms(
  const FEFaceValuesBase<dim>     &fe_face_values,
  const FEFaceValuesBase<dim>     &fe_neighbor_face_values,
  FullMatrix<double>              &vi_ui_matrix,
  FullMatrix<double>              &vi_ue_matrix,
  FullMatrix<double>              &ve_ui_matrix,
  FullMatrix<double>              &ve_ue_matrix,
  const double                     &h)
{
  const unsigned int n_face_points      = fe_face_values.n_quadrature_points;
  const unsigned int dofs_this_cell     = fe_face_values.dofs_per_cell;
  const unsigned int dofs_neighbor_cell = fe_neighbor_face_values.dofs_per_cell;


  Point<dim> beta;
  for (int i=0; i<dim; ++i)
    beta(i) = 1.0;
  beta /= sqrt(beta.square() );

 // std::cout<<"penalty "<<penalty<<" h "<<h<<std::endl;


  for (unsigned int q=0; q<n_face_points; ++q)
    {
      for (unsigned int i=0; i<dofs_this_cell; ++i)
        {
          const Tensor<1,dim>  psi_i_field_minus  =
            fe_face_values[VectorField].value(i,q);
          const double psi_i_potential_minus  =
            fe_face_values[Potential].value(i,q);

          for (unsigned int j=0; j<dofs_this_cell; ++j)
            {
              const Tensor<1,dim> psi_j_field_minus   =
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
              const Tensor<1,dim> psi_j_field_plus    =
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
          const Tensor<1,dim>  psi_i_field_plus =
            fe_neighbor_face_values[VectorField].value(i,q);
          const double         psi_i_potential_plus =
            fe_neighbor_face_values[Potential].value(i,q);

          for (unsigned int j=0; j<dofs_this_cell; ++j)
            {
              const Tensor<1,dim> psi_j_field_minus               =
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
              const Tensor<1,dim> psi_j_field_plus =
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
  const FullMatrix<double> &vi_ui_matrix,
  const FullMatrix<double> &vi_ue_matrix,
  const FullMatrix<double> &ve_ui_matrix,
  const FullMatrix<double> &ve_ue_matrix,
  const std::vector<types::global_dof_index> &local_dof_indices,
  const std::vector<types::global_dof_index> &local_neighbor_dof_indices)
{
  constraints.distribute_local_to_global(vi_ui_matrix,
                                         local_dof_indices,
                                         system_matrix);

  constraints.distribute_local_to_global(vi_ue_matrix,
                                         local_dof_indices,
                                         local_neighbor_dof_indices,
                                         system_matrix);

  constraints.distribute_local_to_global(ve_ui_matrix,
                                         local_neighbor_dof_indices,
                                         local_dof_indices,
                                         system_matrix);

  constraints.distribute_local_to_global(ve_ue_matrix,
                                         local_neighbor_dof_indices,
                                         system_matrix);
}

template<int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::compute_errors() const
  {
    const ComponentSelectFunction<dim> pressure_mask(dim + 1, dim + dim_omega +2);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + dim_omega + 2); 
  /*  const ComponentSelectFunction<dim> pressure_mask(dim + 1, dim + nof_scalar_fields);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + nof_scalar_fields);

   const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(0, dim),
                                                     dim + 1);*/

    Vector<double> cellwise_errors(triangulation.n_active_cells());

    const QTrapezoid<1>  q_trapez;
    const QIterated<dim> quadrature(q_trapez, degree + 2);


    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      true_solution,
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
                                      true_solution,
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

  Vector<float>   subdomain(triangulation.n_active_cells());

  for (unsigned int i=0; i<subdomain.size(); ++i)
    subdomain(i) = triangulation.locally_owned_subdomain();

  data_out.add_data_vector(subdomain,"subdomain");

  data_out.build_patches();

  const std::string filename = ("solution."   +
                                Utilities::int_to_string(
                                  triangulation.locally_owned_subdomain(),4));

  std::ofstream output((filename + ".vtu").c_str());
  data_out.write_vtu(output);

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
    {
      std::vector<std::string>    filenames;
      for (unsigned int i=0;
           i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
           i++)
        {
          filenames.push_back("solution." +
                              Utilities::int_to_string(i,4) +
                              ".vtu");
        }
      std::ofstream master_output("solution.pvtu");
      data_out.write_pvtu_record(master_output, filenames);
    }
}


template<int dim, int dim_omega>
void
LDGPoissonProblem<dim, dim_omega>::
run()
{
  penalty = 100;
  make_grid();
  make_dofs();
  assemble_system();
  solve();
  compute_errors();
  output_results();
}


int main(int argc, char *argv[])
{

  try
    {
      using namespace dealii;

      deallog.depth_console(0);

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv,
                                                          numbers::invalid_unsigned_int);

      unsigned int degree = 1;
      unsigned int n_refine = 4;
      LDGPoissonProblem<2,1>    Poisson(degree, n_refine);
      Poisson.run();

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
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
      std::cerr << std::endl << std::endl
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
