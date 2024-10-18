// https://www.dealii.org/developer/doxygen/deal.II/step_60.html#step_60-Runningwithspacedimequaltothree
// kozlow point
//  @sect3{LDGPoisson.cc}
//  The code begins as per usual with a long list of the the included
//  files from the deal.ii library.


// step_55 step_32 step_40 
// lagrangian step_60 step_70

//custom trianulation jeder Wurzel in einem Prozessor

//TODO
// 1) solve über aufteilun lösen Scurkomplement etc
// 2 mace pralell 
#include <deal.II/lac/lapack_full_matrix.h>
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
#include <deal.II/lac/trilinos_block_sparse_matrix.h>
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
//#include <deal.II/fe_Omega/fe_coupling_values.h>

#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/meshworker/mesh_loop.h>
 #include <deal.II/distributed/shared_tria.h>
 #include <deal.II/base/partitioner.h>
 #include <deal.II/base/index_set.h>
 #include <deal.II/lac/solver_gmres.h>
 #include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/linear_operator_tools.h>
#include <deal.II/lac/precondition_block.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <fstream>
#include <iostream>
#include <stdexcept>

/*ic abe zwei mölickeiten:
1. dofs auf stric zu den korrekten 3D cells zuorden in matrix und dann mit partition_custom_signal in trianulation, die kopplunsterme einem prozessor zuordnen (dann ist das Problem mit zu roßen radius verindert)
2. add extra dof die nicts mit trianulation zu tun aben (klein omea dofs). die können dann von jedem prozessor zueriffen werden (falsch nur von dem ersten) */

// 2. variante ist warsceinlic besser
//einfac die benötiten dofs an matrix iten dran. Dofandler witout zustzice FE , sparsity und so dementsprecen macjen. solution(dof.nof_dof + extraDof)

//TODO man kann Omea trinualtion parallel macen also deren dof andler, und der rest wird auf jedem Porzessor emact
#include "Functions.cc"

using namespace dealii;
#define USE_MPI_ASSEMBLE 1
#define USE_LDG 0
#define BLOCKS 1

constexpr unsigned int dimension_Omega{3};
const FEValuesExtractors::Vector VectorField_omega(0);
const FEValuesExtractors::Scalar Potential_omega(1);

const FEValuesExtractors::Vector VectorField(0);
const FEValuesExtractors::Scalar Potential(dimension_Omega);

const unsigned int dimension_gap = 0;
const double extent = 1;
const double half_length = std::sqrt(0.5);//0.5
const double distance_tolerance = 10;
const unsigned int N_quad_points = 7;

struct Parameters {
  double radius;
  bool lumpedAverage;
};


  class InverseMatrix : public Subscriptor
  {
  public:
    InverseMatrix(const TrilinosWrappers::SparseMatrix &m)
                      : matrix(&m)
    {
      std::cout<<"m.local_size() "<<m.local_size()<<" m.m() "<<m.m()<<std::endl;
    }
 
    void vmult(TrilinosWrappers::MPI::Vector       &dst,
                   const TrilinosWrappers::MPI::Vector &src) const
    {
    dst = 0;
    //std::cout<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" src "<<std::endl;
    //src.print(std::cout);
  /*  SolverControl solver_control(src.size(), 1e-6 * src.l2_norm());
    TrilinosWrappers::SolverDirect solver(solver_control);
    solver.initialize(*matrix);
    solver.solve(dst,src);*/
TrilinosWrappers::PreconditionILU preconditioner;
  TrilinosWrappers::PreconditionILU::AdditionalData data;
  preconditioner.initialize(*matrix, data);

    SolverControl solver_control(1000);//, 1e-7 * src.l2_norm());
    TrilinosWrappers::SolverGMRES solver(solver_control);
    solver.solve(*matrix, dst,  src, preconditioner );
    
    //std::cout<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" dst "<<std::endl;
    //dst.print(std::cout);
    }
 
  private:
    const SmartPointer<const  TrilinosWrappers::SparseMatrix> matrix;
  };


    class SchurComplement : public Subscriptor
    {
    public:
      SchurComplement(
        const TrilinosWrappers::BlockSparseMatrix &system_matrix,
        const InverseMatrix &A_inverse, 
        const TrilinosWrappers::MPI::BlockVector &block_vector)
         : system_matrix(&system_matrix)
        , A_inverse(&A_inverse)
     /*, tmp1(complete_index_set(system_matrix.block(0, 0).m()))//TOD set size and rane corrcecly
      , tmp2(complete_index_set(system_matrix.block(0, 0).m()))
      , tmp3(complete_index_set(system_matrix.block(1, 1).m()))
      , tmp4(complete_index_set(system_matrix.block(1, 1).m()))*/
      , tmp1(block_vector.block(0))
      , tmp2(block_vector.block(0))
      , tmp3(block_vector.block(1))
      , tmp4(block_vector.block(1))

      {/*std::cout<<"system_matrix.block(0, 0).local_size() "<<system_matrix.block(0, 0).local_size()<<" system_matrix.block(1, 1).local_size() "<<system_matrix.block(1, 1).local_size()<<std::endl;
      std::cout<<"system_matrix.block(0, 0).m()) "<<system_matrix.block(0, 0).m()<<" system_matrix.block(1, 1).m() "<<system_matrix.block(1, 1).m()<<std::endl;

      std::cout<<"system_matrix.block(0, 1).local_size() "<<system_matrix.block(0, 1).local_size()<<" system_matrix.block(1, 0).local_size() "<<system_matrix.block(1, 0).local_size()<<std::endl;
      std::cout<<"system_matrix.block(0, 1.m()) "<<system_matrix.block(0, 1).m()<<" system_matrix.block(1, 0).m() "<<system_matrix.block(1, 0).m()<<std::endl;

      set1 = IndexSet(system_matrix.block(0, 0).m());
      set1.add_range(system_matrix.block(0, 0).local_range().first, system_matrix.block(0, 0).local_range().second);

      set2 = IndexSet(system_matrix.block(1, 1).m());
      set2.add_range(system_matrix.block(1, 1).local_range().first, system_matrix.block(1,1).local_range().second);*/
      }

      void vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const
      {
     /* tmp1 = TrilinosWrappers::MPI::Vector(src);
      tmp2 = TrilinosWrappers::MPI::Vector(src);

      tmp3 = TrilinosWrappers::MPI::Vector(set2);
      tmp4 = TrilinosWrappers::MPI::Vector(set2);*/
      //std::cout<<"---------1----------"<<src.size() <<" "<<src.local_size()<<" tmp1  "<<tmp1.size() <<" "<<tmp1.local_size() <<std::endl;


      system_matrix->block(0, 1).vmult(tmp1, src);
     // std::cout<<"---------2---------"<<std::endl;  
      A_inverse->vmult(tmp2, tmp1);
      system_matrix->block(1, 0).vmult(tmp3, tmp2);
      //dst = tmp3;
       system_matrix->block(1, 1).vmult(tmp4, src);
       dst = tmp3- tmp4;
   /*   std::cout<<"---------src----------"<<std::endl;
      src.print(std::cout);      
      std::cout<<"---------tmp1----------"<<std::endl;
      tmp1.print(std::cout);
      std::cout<<"---------tmp2----------"<<std::endl;
      tmp2.print(std::cout);
      std::cout<<"---------dst----------"<<std::endl;
      dst.print(std::cout);
      std::cout<<std::endl<<"***************************"<<std::endl<<std::endl;*/
      }
  
  
    private:
      const SmartPointer<const TrilinosWrappers::BlockSparseMatrix> system_matrix;
      const SmartPointer< const InverseMatrix> A_inverse;
  
      mutable TrilinosWrappers::MPI::Vector tmp1, tmp2, tmp3, tmp4;
       IndexSet set1,set2;
    };






template <int dim, int dim_omega> class LDGPoissonProblem {

public:
  LDGPoissonProblem(const unsigned int degree, const unsigned int n_refine,
                    Parameters parameters);

  ~LDGPoissonProblem();

  std::array<double, 4> run();
  double max_diameter;
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
      const DoFHandler<_dim> &dof_handler_Omega,
      std::vector<types::global_dof_index> &local_dof_indices_omega);

  void solve();

  std::array<double, 4> compute_errors() const;
  void output_results() const;

  const unsigned int degree;
  const unsigned int n_refine;
  double penalty;
  double h_max;
  double h_min;
  unsigned int nof_degrees;

  enum { Dirichlet, Neumann };

  // parameters
  double radius;
  double g;
  bool lumpedAverage;

 
  //parallel::distributed::Triangulation<dim> triangulation_mpi;

  //parallel::shared::Triangulation<dim> triangulation_mpi;

  parallel::shared::Triangulation<dim> triangulation;

  TrilinosWrappers::BlockSparseMatrix system_matrix;
  TrilinosWrappers::MPI::BlockVector solution;
  TrilinosWrappers::MPI::BlockVector system_rhs;

 TrilinosWrappers::MPI::Vector locally_relevant_solution_Omega;

 //TrilinosWrappers::BlockSparsityPattern  dsp_block;
  BlockSparsityPattern block_sparsity_pattern;
 /* BlockSparseMatrix<double> system_matrix;
  BlockVector<double> solution;
  BlockVector<double> system_rhs;*/

  // SparsityPattern sparsity_pattern;

  FESystem<dim> fe_Omega;
  DoFHandler<dim> dof_handler_Omega;

  //Triangulation<dim_omega> triangulation_omega;
  parallel::shared::Triangulation<dim_omega> triangulation_omega;
  FESystem<dim_omega> fe_omega;
  DoFHandler<dim_omega> dof_handler_omega;
  Vector<double> solution_omega;
  Vector<double> solution_Omega;



  IndexSet locally_owned_dofs_Omega;
  IndexSet locally_relevant_dofs_Omega;

  IndexSet locally_owned_dofs_omega_local;
  IndexSet locally_relevant_dofs_omega_local;

  IndexSet locally_owned_dofs_omega_global;
  IndexSet locally_relevant_dofs_omega_global;

  IndexSet locally_owned_dofs_total;
  IndexSet locally_relevant_dofs_total;

  std::vector<IndexSet> locally_owned_dofs_block;
  std::vector<IndexSet> locally_relevant_dofs_block;

  AffineConstraints<double> constraints;

  std::vector<bool> marked_vertices;

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
  unsigned int start_Potential_Omega;

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
      triangulation(MPI_COMM_WORLD),
      triangulation_omega(MPI_COMM_WORLD),
      fe_Omega(FESystem<dim>(FE_DGP<dim>(degree), dim), FE_DGP<dim>(degree)),
      dof_handler_Omega(triangulation),
      fe_omega(FESystem<dim_omega>(FE_DGP<dim_omega>(degree), dim_omega),
               FE_DGP<dim_omega>(degree)),
      dof_handler_omega(triangulation_omega),
#if USE_MPI_ASSEMBLE
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
  dof_handler_Omega.clear();
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
  Point<dim> p1, p2;
if (dim == 3) {
   p1 =
      Point<dim>(2 * half_length, -half_length + offset, -half_length + offset);
   p2 =
      Point<dim>(0, half_length + offset, half_length + offset);
  // std::cout<<"hyper_rectangle "<<p1 << " "<<p2<<std::endl;
 
}
 if (dim == 2) {
   p1 =
      Point<dim>(-half_length + offset, -half_length + offset);
   p2 =
      Point<dim>(half_length + offset, half_length + offset);

 }
 GridGenerator::hyper_rectangle(triangulation, p1, p2);
#endif


  triangulation.refine_global(n_refine);

 max_diameter = 0.0;
 typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_Omega.begin_active(),
        endc = dof_handler_Omega.end();
  for (; cell != endc; ++cell) {
    double cell_diameter = cell->diameter(); // Get the diameter of the cell^
    
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
  std::cout<<"max_diameter "<<max_diameter<<std::endl;
  if (radius > max_diameter && !lumpedAverage) {
    std::cout << "!!!!!!!!!!!!!! MAX DIAMETER > RADIUS !!!!!!!!!!!!!!!!"
              << max_diameter << radius << std::endl;
    //throw std::invalid_argument("MAX DIAMETER > RADIUS");
  }



//---------------omega-------------------------
  if (constructed_solution == 3)
    GridGenerator::hyper_cube(triangulation_omega, 0, 2 * half_length);
  else
    GridGenerator::hyper_cube(triangulation_omega, -extent / 2, extent / 2);
  triangulation_omega.refine_global(n_refine);

 typename DoFHandler<dim_omega>::active_cell_iterator
        cell_omega = dof_handler_omega.begin_active(),
        endc_omega = dof_handler_omega.end();
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

  dof_handler_Omega.distribute_dofs(fe_Omega);
  const unsigned int dofs_per_cell = fe_Omega.dofs_per_cell;
  pcout << "dofs_per_cell " << dofs_per_cell << std::endl;
  //DoFRenumbering::component_wise(dof_handler_Omega);

  dof_handler_omega.distribute_dofs(fe_omega);
  const unsigned int dofs_per_cell_omega = fe_omega.dofs_per_cell;
  pcout << "dofs_per_cell_omega " << dofs_per_cell_omega << std::endl;
 //DoFRenumbering::component_wise(dof_handler_omega);


  const std::vector<types::global_dof_index> dofs_per_component_Omega =
      DoFTools::count_dofs_per_fe_component(dof_handler_Omega);

  const unsigned int n_vector_field_Omega = dim * dofs_per_component_Omega[0];
  const unsigned int n_potential_Omega = dofs_per_component_Omega[dofs_per_component_Omega.size()-1];

  for (unsigned int i = 0; i < dofs_per_component_Omega.size(); i++)
    pcout << "dofs_per_component_Omega " << dofs_per_component_Omega[i] << std::endl;

  pcout <<  "Omega ----------------------------"<<std::endl
        <<"Number of global active cells: "
        << triangulation.n_global_active_cells() << std::endl
        << "Number of degrees of freedom: " << dof_handler_Omega.n_dofs() << " ("
        << n_vector_field_Omega << " + " << n_potential_Omega << ")" << std::endl;


const std::vector<types::global_dof_index> dofs_per_component_omega =
      DoFTools::count_dofs_per_fe_component(dof_handler_omega);

  const unsigned int n_vector_field_omega = dim_omega * dofs_per_component_omega[0];
  const unsigned int n_potential_omega = dofs_per_component_omega[dofs_per_component_omega.size()-1];

   pcout <<  "omega ----------------------------"<<std::endl
       << "Number of global active cells: "
        << triangulation_omega.n_global_active_cells() << std::endl
        << "Number of degrees of freedom: " << dof_handler_omega.n_dofs() << " ("
        << n_vector_field_omega << " + " << n_potential_omega << ")" << std::endl;



  start_VectorField_omega = dof_handler_Omega.n_dofs();
  start_Potential_omega =  dof_handler_Omega.n_dofs() + dofs_per_component_omega[0];
  start_Potential_Omega = n_vector_field_Omega;
  pcout << " start_VectorField_omega " << start_VectorField_omega
        << " start_Potential_Omega " << start_Potential_Omega << " start_Potential_omega "
        << start_Potential_omega << std::endl;


  
  constraints.clear();
  constraints.close();
  unsigned int n_dofs_total = dof_handler_Omega.n_dofs() + dof_handler_omega.n_dofs();


  #if USE_MPI_ASSEMBLE
  locally_owned_dofs_Omega = dof_handler_Omega.locally_owned_dofs();
  locally_relevant_dofs_Omega;
  DoFTools::extract_locally_relevant_dofs(dof_handler_Omega, locally_relevant_dofs_Omega);

  locally_owned_dofs_omega_local = dof_handler_omega.locally_owned_dofs();
  // if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0 )
   //locally_owned_dofs_omega_local.clear();
   
  locally_owned_dofs_omega_global.set_size(locally_owned_dofs_omega_local.size());
  locally_owned_dofs_omega_global.add_indices(locally_owned_dofs_omega_local,  dof_handler_Omega.n_dofs());
  locally_relevant_dofs_omega_local;
  DoFTools::extract_locally_relevant_dofs(dof_handler_omega, locally_relevant_dofs_omega_local);
  locally_relevant_dofs_omega_global.set_size(locally_relevant_dofs_omega_local.size());
  locally_relevant_dofs_omega_global.add_indices(locally_relevant_dofs_omega_local,  dof_handler_Omega.n_dofs());



  locally_owned_dofs_block.push_back(locally_owned_dofs_Omega);
  locally_owned_dofs_block.push_back(locally_owned_dofs_omega_local);

  locally_relevant_dofs_block.push_back(locally_relevant_dofs_Omega);
  locally_relevant_dofs_block.push_back(locally_relevant_dofs_omega_local);



  locally_owned_dofs_total.set_size(n_dofs_total);
  locally_relevant_dofs_total.set_size(n_dofs_total);

  locally_owned_dofs_total.add_indices(locally_owned_dofs_Omega);
  locally_relevant_dofs_total.add_indices(locally_relevant_dofs_Omega);
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
  {

   locally_owned_dofs_total.add_indices(locally_owned_dofs_omega_global);
     locally_relevant_dofs_total.add_indices(locally_relevant_dofs_omega_global);
  }

  locally_owned_dofs_total.compress();
  locally_relevant_dofs_total.compress();

 /* 
  std::cout <<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" locally_owned_dofs_Omega "<<  locally_owned_dofs_Omega.size()<<" elem "<<locally_owned_dofs_Omega.n_elements()<<std::endl;
  for (auto index : locally_owned_dofs_Omega)
        std::cout << index << " ";
    std::cout << std::endl;
  std::cout <<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" locally_owned_dofs_omega_local "<<  locally_owned_dofs_omega_local.size()<<" elem "<<locally_owned_dofs_omega_local.n_elements()<< std::endl;
  for (auto index : locally_owned_dofs_omega_local)
        std::cout << index << " ";
    std::cout << std::endl;
    std::cout <<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" locally_owned_dofs_omega_global "<<  locally_owned_dofs_omega_global.size()<<" elem "<<locally_owned_dofs_omega_global.n_elements()<< std::endl;
  for (auto index : locally_owned_dofs_omega_global)
        std::cout << index << " ";
    std::cout << std::endl;

    std::cout <<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" locally_owned_dofs_total "<<  locally_owned_dofs_total.size()<< " elem "<<locally_owned_dofs_total.n_elements()<<std::endl;
  for (auto index : locally_owned_dofs_total)
        std::cout << index << " ";
    std::cout << std::endl;*/

std::cout <<"----------------------------------- "<<  std::endl;
 /*  std::cout <<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" locally_relevant_dofs_Omega "<<  locally_relevant_dofs_Omega.size()<<" elem "<<locally_relevant_dofs_Omega.n_elements()<< std::endl;
  for (auto index : locally_relevant_dofs_Omega)
        std::cout << index << " ";
    std::cout << std::endl;
 std::cout <<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" locally_relevant_dofs_omega_local "<<  locally_relevant_dofs_omega_local.size()<<" elem "<<locally_relevant_dofs_omega_local.n_elements()<< std::endl;
  for (auto index : locally_relevant_dofs_omega_local)
        std::cout << index << " ";
    std::cout << std::endl;*/
/*
    std::cout <<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" locally_relevant_dofs_total "<<  locally_relevant_dofs_total.size()<< " elem "<<locally_relevant_dofs_total.n_elements()<< std::endl;
  for (auto index : locally_relevant_dofs_total)
        std::cout << index << " ";
    std::cout << std::endl;*/
#endif
std::vector<unsigned int> dofs_per_block;
#if BLOCKS
dofs_per_block.push_back(dof_handler_Omega.n_dofs());
dofs_per_block.push_back(dof_handler_omega.n_dofs());
#else
dofs_per_block.push_back(n_vector_field_Omega);
dofs_per_block.push_back(n_potential_Omega);
dofs_per_block.push_back(n_vector_field_omega);
dofs_per_block.push_back(n_potential_omega);
#endif
#if BLOCKS
BlockDynamicSparsityPattern dsp_block(2,2);
//dsp_block =  TrilinosWrappers::BlockSparsityPattern(2,2);
  dsp_block.block(0, 0).reinit(dof_handler_Omega.n_dofs(), dof_handler_Omega.n_dofs());
    // Block 1,1: Sparsity for the extra DoFs (global DoFs)
  dsp_block.block(1, 1).reinit(dof_handler_omega.n_dofs(), dof_handler_omega.n_dofs());

  // Block 0,1 and 1,0: Sparsity for connections between mesh-based and extra DoFs
  dsp_block.block(0, 1).reinit(dof_handler_Omega.n_dofs(), dof_handler_omega.n_dofs());  // Block for coupling DoFs
  dsp_block.block(1, 0).reinit(dof_handler_omega.n_dofs(), dof_handler_Omega.n_dofs());


  DoFTools::make_flux_sparsity_pattern(dof_handler_Omega, dsp_block.block(0,0));//,  constraints,false,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
  DoFTools::make_flux_sparsity_pattern(dof_handler_omega, dsp_block.block(1,1) );

 /* SparsityTools::distribute_sparsity_pattern(dsp_block.block(0,0),
                                                 locally_owned_dofs_Omega,
                                                 MPI_COMM_WORLD,
                                                 locally_relevant_dofs_Omega);*/
 /*SparsityTools::distribute_sparsity_pattern(dsp_block.block(1,1),
                                                 locally_owned_dofs_omega_local,
                                                 MPI_COMM_WORLD,
                                                 locally_relevant_dofs_omega_local);*/
/*SparsityTools::distribute_sparsity_pattern(dsp_block,
                                                 locally_owned_dofs_total,
                                                 MPI_COMM_WORLD,
                                                 locally_relevant_dofs_total);*/

 /*TrilinosWrappers::BlockSparsityPattern sp_block(locally_owned_dofs_block,
                                                locally_owned_dofs_block,
                                                locally_relevant_dofs_block,
                                                MPI_COMM_WORLD);*/
//DoFTools::make_flux_sparsity_pattern(dof_handler_Omega, sp_block.block(0,0), constraints,false,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
 // DoFTools::make_flux_sparsity_pattern(dof_handler_omega, sp_block.block(1,1),constraints,false,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) );
TrilinosWrappers::BlockSparsityPattern sp_block=  TrilinosWrappers::BlockSparsityPattern(2,2);
  sp_block.block(0, 0).reinit(locally_owned_dofs_Omega,locally_owned_dofs_Omega,locally_relevant_dofs_Omega,MPI_COMM_WORLD);

    // Block 1,1: Sparsity for the extra DoFs (global DoFs)
  sp_block.block(1, 1).reinit(locally_owned_dofs_omega_local,locally_owned_dofs_omega_local,locally_relevant_dofs_omega_local,MPI_COMM_WORLD);

  // Block 0,1 and 1,0: Sparsity for connections between mesh-based and extra DoFs
  sp_block.block(0, 1).reinit(locally_owned_dofs_Omega,locally_owned_dofs_omega_local,locally_relevant_dofs_Omega,MPI_COMM_WORLD);  // Block for coupling DoFs
 
  sp_block.block(1, 0).reinit(locally_owned_dofs_omega_local,locally_owned_dofs_Omega,locally_relevant_dofs_omega_local,MPI_COMM_WORLD);
 

  DoFTools::make_flux_sparsity_pattern(dof_handler_Omega, sp_block.block(0,0),constraints,false,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));//,  constraints,false,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
  DoFTools::make_flux_sparsity_pattern(dof_handler_omega, sp_block.block(1,1),constraints,false,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) );
  sp_block.collect_sizes();

#else
    const std::vector<types::global_dof_index> block_sizes_Omega = {n_vector_field_Omega, n_potential_Omega};
    BlockDynamicSparsityPattern                dsp_Omega(block_sizes_Omega, block_sizes_Omega);
    DoFTools::make_flux_sparsity_pattern(dof_handler_Omega, dsp_Omega);

    const std::vector<types::global_dof_index> block_sizes_omega = {n_vector_field_omega, n_potential_omega};
    BlockDynamicSparsityPattern                dsp_omega(block_sizes_omega, block_sizes_omega);
    DoFTools::make_flux_sparsity_pattern(dof_handler_omega, dsp_omega);

//https://mathoverflow.net/questions/158437/how-to-write-this-result-successive-schur-complements-compose-nicely?rq=1
//TODO macen das entries reinescrieben werden 
  BlockDynamicSparsityPattern dsp_block(4,4);
  dsp_block.block(0, 0).reinit(n_vector_field_Omega, n_vector_field_Omega, dsp_Omega.block(0,0).row_index_set());
  dsp_block.block(1, 0).reinit(n_potential_Omega, n_vector_field_Omega, dsp_Omega.block(1,0).row_index_set());
  dsp_block.block(0, 1).reinit(n_vector_field_Omega, n_potential_Omega, dsp_Omega.block(0,1).row_index_set());
  dsp_block.block(1, 1).reinit(n_potential_Omega, n_potential_Omega, dsp_Omega.block(1,1).row_index_set());

  dsp_block.block(2, 2).reinit(n_vector_field_omega, n_vector_field_omega, dsp_omega.block(0,0).row_index_set());
  dsp_block.block(3, 2).reinit(n_potential_omega, n_vector_field_omega, dsp_omega.block(1,0).row_index_set());
  dsp_block.block(2, 3).reinit(n_vector_field_omega, n_potential_omega, dsp_omega.block(0,1).row_index_set());
  dsp_block.block(3, 3).reinit(n_potential_omega, n_potential_omega, dsp_omega.block(1,1).row_index_set());


  dsp_block.block(2, 0).reinit(n_vector_field_omega, n_vector_field_Omega);
  dsp_block.block(3, 0).reinit(n_potential_omega, n_vector_field_Omega);
  dsp_block.block(2, 1).reinit(n_vector_field_omega, n_potential_Omega);
  dsp_block.block(3, 1).reinit(n_potential_omega, n_potential_Omega);


  dsp_block.block(0, 2).reinit(n_vector_field_Omega, n_vector_field_omega);
  dsp_block.block(1, 2).reinit(n_potential_Omega, n_vector_field_omega);
  dsp_block.block(0, 3).reinit(n_vector_field_Omega, n_potential_omega);
  dsp_block.block(1, 3).reinit(n_potential_Omega, n_potential_omega);
#endif

  
  // for(unsigned int i = 0; i < dof_handler_Omega.n_dofs(); i++)
  // for(unsigned int j = 0; j < dof_handler_omega.n_dofs(); j++)
  // {
  //   dsp_block.block(0, 1).add(i,j);
  //   dsp_block.block(1, 0).add(j,i);
  // }


dsp_block.collect_sizes();
 


#if COUPLED
  {
marked_vertices.resize(triangulation.n_vertices());
Point<dim> corner1, corner2;
if(dim == 3)
{
corner1 =  Point<dim>(0, - radius , - radius);
corner2 =  Point<dim>(2 * half_length,  radius,  radius);
}
if(dim == 2)
{
corner1 =  Point<dim>( - 2*radius , - 2*radius);
corner2 =  Point<dim>( 2*radius,  2*radius);
}
std::pair<Point<dim>, Point<dim>> corner_pair(corner1, corner2);     
    
BoundingBox<dim> bbox(corner_pair);

const std::vector<Point<dim>> &vertices = triangulation.get_vertices();

for (unsigned int i = 0; i < triangulation.n_vertices(); i++)
{
// marked_vertices.push_back(true);
    if (bbox.point_inside(vertices[i]))
    {
       //std::cout<<vertex<<" ";
     // std::cout<<true<<" "<<std::endl;
      marked_vertices[i] = true;
    }
    else
    {
       /*std::cout<<vertex<<" ";
       std::cout<<false<<" "<<std::endl;*/
      marked_vertices[i] = false;
    }
   
}
    // coupling

    QGauss<dim> quadrature_formula(fe_Omega.degree + 1);
    FEValues<dim> fe_values(fe_Omega, quadrature_formula, update_flags);
    const Mapping<dim> &mapping = fe_values.get_mapping();

    QGauss<dim_omega> quadrature_formula_omega(fe_Omega.degree + 1);
    FEValues<dim_omega> fe_values_omega(fe_omega, quadrature_formula_omega,
                                        update_flags);

    pcout << "setup dofs Coupling" << std::endl;
    std::vector<types::global_dof_index> local_dof_indices_test(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_trial(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_omega(
        dofs_per_cell_omega);
    unsigned int nof_quad_points;
    bool AVERAGE = radius != 0 && !lumpedAverage;
    pcout << "AVERAGE (use circel) " << AVERAGE << std::endl;
    // weight
    if (AVERAGE) {
      nof_quad_points = N_quad_points;
    } else {
      nof_quad_points = 1;
    }
    std::cout<<"nof_quad_points "<<nof_quad_points<<std::endl;
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
//std::cout<<"concept "<<concepts::internal::is_triangulation_or_dof_handler<dof_handler_Omega> <<std::endl;
#if TEST
      auto start = std::chrono::high_resolution_clock::now();  //Start time
    auto cell_test_array = GridTools::find_all_active_cells_around_point(
        mapping, dof_handler_Omega, quadrature_point_test, 1e-10);
    auto end = std::chrono::high_resolution_clock::now();    // End time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
/*std::cout << "Time taken to execute find_all_active_cells_around_point: " << duration << " ms" << std::endl;
       std::cout << "cell_test_array " << cell_test_array.size() << std::endl;
   auto map = GridTools::vertex_to_cell_map(triangulation);
    auto start1 = std::chrono::high_resolution_clock::now();
    auto cell_test = GridTools::find_active_cell_around_point(
            mapping, dof_handler_Omega, quadrature_point_test, marked_vertices);
    auto all_cells  = GridTools::find_all_active_cells_around_point(
                       mapping, dof_handler_Omega, quadrature_point_test,1e-10 ,cell_test);
      auto end1 = std::chrono::high_resolution_clock::now();    // End time
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count();
    std::cout << "Time taken to execute find_active_cell_around_secondvariant: " 
              << duration1 << " ms" << std::endl;*/

        for (auto cellpair : cell_test_array)
#else
        auto cell_test = GridTools::find_active_cell_around_point(
            dof_handler_Omega, quadrature_point_test);
#endif

        {
#if TEST
          auto cell_test = cellpair.first;
#endif

#if USE_MPI_ASSEMBLE
         if (cell_test != dof_handler_Omega.end())
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
                  fe_Omega, my_quadrature_formula_test, update_flags_coupling);
              fe_values_coupling_test.reinit(cell_test);

              // std::cout << "coupled " << std::endl;
              for (unsigned int q_avag = 0; q_avag < nof_quad_points;
                   q_avag++) {
                // Quadrature weights and points
                quadrature_point_trial = quadrature_points_circle[q_avag];

#if TEST
                    auto start = std::chrono::high_resolution_clock::now();  //Start time
    auto cell_trial_array = GridTools::find_all_active_cells_around_point(
        mapping, dof_handler_Omega, quadrature_point_trial, 1e-10, marked_vertices);
    auto end = std::chrono::high_resolution_clock::now();    // End time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
 /*   std::cout << "Time taken to execute find_all_active_cells_around_point_trial: " << duration << " ms" << std::endl;
                  std::cout << "cell_trial_array " << cell_trial_array.size()  << std::endl;*/

                for (auto cellpair_trial : cell_trial_array)
#else
              auto cell_trial = GridTools::find_active_cell_around_point(
                  dof_handler_Omega, quadrature_point_trial);
#endif

                {
#if TEST
                  auto cell_trial = cellpair_trial.first;
#endif
                  if (cell_trial != dof_handler_Omega.end()) {
                    if (cell_trial->is_locally_owned() &&
                        cell_test->is_locally_owned()) {

                      cell_trial->get_dof_indices(local_dof_indices_trial);

                      for (unsigned int i = 0;
                           i < local_dof_indices_test.size(); i++) {
                        for (unsigned int j = 0;
                             j < local_dof_indices_trial.size(); j++) {
                          dsp_block.add(local_dof_indices_test[i],
                                  local_dof_indices_trial[j]);
                          sp_block.add(local_dof_indices_test[i],
                                  local_dof_indices_trial[j]);
                        }
                      }

                      for (unsigned int i = 0;
                           i < local_dof_indices_omega.size(); i++) {
                        for (unsigned int j = 0;
                             j < local_dof_indices_trial.size(); j++) {
                          dsp_block.add(local_dof_indices_omega[i],
                                  local_dof_indices_trial[j]);
                        sp_block.add(local_dof_indices_omega[i],
                                  local_dof_indices_trial[j]);
                        }
                      }

                      for (unsigned int i = 0;
                           i < local_dof_indices_test.size(); i++) {
                        for (unsigned int j = 0;
                             j < local_dof_indices_omega.size(); j++) {
                          dsp_block.add(local_dof_indices_test[i],
                                  local_dof_indices_omega[j]);
                            sp_block.add(local_dof_indices_test[i],
                                  local_dof_indices_omega[j]);
                        }
                      }

                      for (unsigned int i = 0;
                           i < local_dof_indices_omega.size(); i++) {
                        for (unsigned int j = 0;
                             j < local_dof_indices_omega.size(); j++) {
                          dsp_block.add(local_dof_indices_omega[i],
                                  local_dof_indices_omega[j]);                               
                            sp_block.add(local_dof_indices_omega[i],
                                  local_dof_indices_omega[j]);                        }
                      }



                   } 
                 /*  else
                  std::cout<<"düdüm1"<<std::endl;*/
                  }
                /*else
                  std::cout<<"düdüm2"<<std::endl;*/
                }
              }
            }
            // else
          // std::cout<<"düdüm3"<<std::endl;
        }
        // std::cout<<std::endl;
      }
    }
  }
#endif

   /* SparsityTools::distribute_sparsity_pattern(
      dsp_block, local_dofs, MPI_COMM_WORLD,
       relevant_dofs);*/

  /*
        Table<2, DoFTools::Coupling> coupling(dim + 1, dim + 1);
        for (unsigned int c = 0; c < dim + 1; ++c)
          for (unsigned int d = 0; d < dim + 1; ++d)
            if (!((c == dim) && (d == dim)))
              coupling[c][d] = DoFTools::always;
            else
              coupling[c][d] = DoFTools::none;
  
        DoFTools::make_sparsity_pattern(
          dof_handler, coupling, dsp, constraints, false);*/
  
  block_sparsity_pattern.copy_from(dsp_block);   

    /*  sp_block.block(0, 0).compress();
    sp_block.block(1, 1).compress();
     sp_block.block(0, 1).compress();
      sp_block.block(1, 0).compress();*/
  sp_block.compress();

   pcout<<"Sparsity "  <<sp_block.n_rows()<<" "<<sp_block.n_cols()<<std::endl;
nof_degrees = dsp_block.n_rows();
  //std::cout<<"n_nonzero_elements "<<block_sparsity_pattern.n_nonzero_elements() <<std::endl;                                    

  //std::ofstream out("sparsity-pattern-2.svg");
  //sp_block.print_svg(out);

  //system_matrix.reinit(locally_owned_dofs_block, block_sparsity_pattern, MPI_COMM_WORLD);
  system_matrix.reinit(sp_block);
  solution.reinit(locally_relevant_dofs_block,  MPI_COMM_WORLD);
  system_rhs.reinit(locally_owned_dofs_block, locally_relevant_dofs_block,  MPI_COMM_WORLD, true);

  locally_relevant_solution_Omega.reinit(locally_owned_dofs_Omega, locally_relevant_dofs_Omega,  MPI_COMM_WORLD);
  std::cout<<"Ende setup dof"<<std::endl;
}

template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::assemble_system() {
  TimerOutput::Scope t(computing_timer, "assembly");
  pcout << "assemble_system" << std::endl;

  QGauss<dim> quadrature_formula(fe_Omega.degree + 1);
  QGauss<dim - 1> face_quadrature_formula(fe_Omega.degree + 1);

  const unsigned int dofs_per_cell = fe_Omega.dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> local_neighbor_dof_indices(
      dofs_per_cell);

  const IndexSet &locally_owned_dofs = dof_handler_Omega.locally_owned_dofs();

  FEValues<dim> fe_values(fe_Omega, quadrature_formula, update_flags);

  FEFaceValues<dim> fe_face_values(fe_Omega, face_quadrature_formula,
                                   face_update_flags);

  FEFaceValues<dim> fe_neighbor_face_values(fe_Omega, face_quadrature_formula,
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

    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler_Omega
                                                              .begin_active(),
                                                   endc = dof_handler_Omega.end();
    // unsigned int cell_number = 0;
    for (; cell != endc; ++cell) {
      // std::cout<<"cell_number "<<cell_number<<std::endl;
      // cell_number++;
      // unsigned int cell_id = cell->index();
      // std::cout<<cell_id<<std::endl;
#if USE_MPI_ASSEMBLE
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
        // for(unsigned int dof : local_dof_indices)
        // std::cout<<dof<<" ";
        // std::cout<<std::endl;

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
#if 1// USE_MPI_ASSEMBLE
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
#endif

  // omega
  QGauss<dim_omega> quadrature_formula_omega(fe_Omega.degree + 1);
  QGauss<dim_omega - 1> face_quadrature_formula_omega(fe_Omega.degree + 1);

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

#if 0// USE_MPI_ASSEMBLE
// if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
#endif
  #if 1
  {
    TimerOutput::Scope t(computing_timer, "assembly - omega");
    pcout << "assemly - omega" << std::endl;

    for (; cell_omega != endc_omega; ++cell_omega) {
       unsigned int cell_id_omega = cell_omega->index();
       //std::cout<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" cell_id "<<cell_id_omega<<std::endl;

    if (cell_omega->is_locally_owned())

    {
    
      local_matrix_omega = 0;
      local_vector_omega = 0;
    
      fe_values_omega.reinit(cell_omega);

      assemble_cell_terms(fe_values_omega, local_matrix_omega,
                          local_vector_omega, k_inverse_function,
                          rhs_function_omega, VectorField_omega,
                          Potential_omega);

      cell_omega->get_dof_indices(local_dof_indices_omega);

      dof_omega_to_Omega(dof_handler_omega, local_dof_indices_omega);
/*
      indices_i.clear();
      local_dof_indices_omega_locally_owned.clear();
      for (unsigned int i = 0; i < local_dof_indices_omega.size(); i++) {
        types::global_dof_index dof_index = local_dof_indices_omega[i];
        if (locally_owned_dofs.is_element(dof_index)) {
          indices_i.push_back(dof_index);
          local_dof_indices_omega_locally_owned.push_back(dof_index);
        }
      }
      */

      for (unsigned int face_no_omega = 0;
           face_no_omega < GeometryInfo<dim_omega>::faces_per_cell;
           face_no_omega++) {
            
          //std::cout<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" cell_id "<<cell_id_omega<<" face_no_omega "<<face_no_omega<<std::endl;
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
           // std::cout<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" cell_id "<<cell_id_omega<<" face_no_omega "<<face_no_omega<< " Dirichlet"<<std::endl;
            
          }
          // else if (face_omega->boundary_id() == Neumann)
          //   {
          //     assemble_Neumann_boundary_terms(fe_face_values_omega,
          //                                     local_matrix_omega,
          //                                 local_vector_omega);
          //   }
          else
            Assert(false, ExcNotImplemented());
        } else {

         /* Assert(cell_omega->neighbor(face_no_omega).state() ==
                     IteratorState::valid,
                 ExcInternalError());*/

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
   /*                            
            indices_j.clear();
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
            }
*/
            distribute_local_flux_to_global(
                vi_ui_matrix_omega, vi_ue_matrix_omega, ve_ui_matrix_omega,
                ve_ue_matrix_omega, local_dof_indices_omega,
                local_neighbor_dof_indices_omega);
          }
        }
        
      }//face iterate

      constraints.distribute_local_to_global(
          local_matrix_omega, local_dof_indices_omega, system_matrix);

      constraints.distribute_local_to_global(
          local_vector_omega, local_dof_indices_omega, system_rhs);
      //#endif
    
   }//end locally ownedS
    
    }
    
  }
  #endif
#if 0// USE_MPI_ASSEMBLE
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
#endif
  std::cout << "ende omega loop" << std::endl;
#if 1
#if 1// USE_MPI_ASSEMBLE
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
        mapping, dof_handler_Omega, quadrature_point_test, 1e-10, marked_vertices);
    n_te = cell_test_array.size();
    //   n_te = 1;
    pcout << "cell_test_array " << cell_test_array.size() << std::endl;

    for (auto cellpair : cell_test_array)
#else
    auto cell_test = GridTools::find_active_cell_around_point(
        dof_handler_Omega, quadrature_point_test);
    n_te = 1;
#endif

    {
#if TEST
      auto cell_test = cellpair.first;
#endif

#if 0// USE_MPI_ASSEMBLE
      if (cell_test != dof_handler_Omega.end())
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
                  fe_Omega, my_quadrature_formula_test, update_flags_coupling);
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
                fe_Omega, my_quadrature_formula_test,
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
            mapping, dof_handler_Omega, quadrature_point_test, 1e-10, marked_vertices);
        n_te = cell_test_array.size();
        //   n_te = 1;
        // pcout << "cell_test_array " << cell_test_array.size() << std::endl;
        for (auto cellpair : cell_test_array)
#else
        auto cell_test = GridTools::find_active_cell_around_point(
            dof_handler_Omega, quadrature_point_test);
        n_te = 1;
#endif

        {
#if TEST
          auto cell_test = cellpair.first;
#endif

#if 1// USE_MPI_ASSEMBLE
          if (cell_test != dof_handler_Omega.end())
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
                  fe_Omega, my_quadrature_formula_test, update_flags_coupling);
              fe_values_coupling_test.reinit(cell_test);

#if !COUPLED
            //  std::cout << "not coupled" << std::endl;
              //-------------face -----------------
              // n_ftest = 0;
              if (!insideCell_test) {
               // pcout << "Omega rhs face " << std::endl;
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
                        fe_Omega, my_quadrature_formula_test, update_flags_coupling);
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
                        mapping, dof_handler_Omega, quadrature_point_trial, 1e-10, marked_vertices);
                // pcout<< "cell_trial_array " << cell_trial_array.size() <<
                // std::endl;
                n_tr = cell_trial_array.size();
                // n_tr  =1;
                for (auto cellpair_trial : cell_trial_array)
#else
                auto cell_trial = GridTools::find_active_cell_around_point(
                    dof_handler_Omega, quadrature_point_trial);
                n_tr = 1;
#endif

                {
#if TEST
                  auto cell_trial = cellpair_trial.first;
#endif
                 if (cell_trial != dof_handler_Omega.end())
                    if (cell_trial->is_locally_owned() &&
                       cell_test->is_locally_owned()) 
                     {
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
                          fe_Omega, my_quadrature_formula_trial,
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
                            fe_Omega, my_quadrature_formula_test,
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
                              fe_Omega, my_quadrature_formula_trial,
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

#if 0// USE_MPI_ASSEMBLE
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
#endif
#endif
  // std::cout << "ende coupling loop" << std::endl;

  // std::cout << "set ii " << std::endl;
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
/*
  for (unsigned int i = 0; i < dof_handler_Omega.n_dofs() + dof_handler_omega.n_dofs(); i++) // dof_table.size()
  {
    // if(dof_table[i].first.first == 1 || dof_table[i].first.first == 3)
    {
      if (system_matrix.el(i, i) == 0) {

        system_matrix.set(i, i, 1);
      }
    }
  }
*/
}

template <int dim, int dim_omega>
template <int _dim>
void LDGPoissonProblem<dim, dim_omega>::dof_omega_to_Omega(
    const DoFHandler<_dim> &dof_handler_omega,
    std::vector<types::global_dof_index> &local_dof_indices_omega) {
//std::cout<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" start dof_omega_to_Omega"<<std::endl;

  const std::vector<types::global_dof_index> dofs_per_component_omega ={16,16};
   //   DoFTools::count_dofs_per_fe_component(dof_handler_omega);
//std::cout<< Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" "<<dofs_per_component_omega[0]<<" "<< dofs_per_component_omega[1]<<std::endl;
  for (unsigned int i = 0; i < local_dof_indices_omega.size(); ++i) {
  //  std::cout<< local_dof_indices_omega[i]<<std::endl;
   /* const unsigned int base_i =
        dof_handler_omega.get_fe().system_to_base_index(i).first.first;

    local_dof_indices_omega[i] =
        base_i == 0 ? local_dof_indices_omega[i] + start_VectorField_omega
                    : local_dof_indices_omega[i] - dofs_per_component_omega[0] + start_Potential_omega;*/
    
    local_dof_indices_omega[i] =   start_VectorField_omega +  local_dof_indices_omega[i];           
                    //local_dof_indices_omega[i] - dofs_per_component_omega[0] + start_Potential_omega;
          
  }
 // std::cout<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" end dof_omega_to_Omega"<<std::endl;
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
 // std::cout<<"dofs_this_cell "<<dofs_this_cell <<" dofs_neighbor_cell "<<dofs_neighbor_cell<<std::endl;
  const unsigned int dofs_cell = dofs_this_cell;
  
  for (unsigned int q = 0; q < n_face_points; ++q) {
    // ------------- test -----------------------------
    for (unsigned int i = 0; i < dofs_cell; ++i) {
      //this
      const Tensor<1, _dim> psi_i_field_minus =
          fe_face_values[VectorField].value(i, q);
      const double psi_i_potential_minus =
          fe_face_values[Potential].value(i, q);

    //neighbor
    const Tensor<1, _dim> psi_i_field_plus =
          fe_neighbor_face_values[VectorField].value(i, q);
      const double psi_i_potential_plus =
          fe_neighbor_face_values[Potential].value(i, q);
      // --------------------- trial -------------------------------
      for (unsigned int j = 0; j < dofs_cell; ++j) {
        const Tensor<1, _dim> psi_j_field_minus =
            fe_face_values[VectorField].value(j, q);
        const double psi_j_potential_minus =
            fe_face_values[Potential].value(j, q);
        const Tensor<1, _dim> psi_j_field_plus =
            fe_neighbor_face_values[VectorField].value(j, q);
        const double psi_j_potential_plus =
            fe_neighbor_face_values[Potential].value(j, q);
        //this this
        vi_ui_matrix(i, j) +=
            (0.5 * (psi_i_field_minus * fe_face_values.normal_vector(q) *
                        psi_j_potential_minus +
                    psi_i_potential_minus * fe_face_values.normal_vector(q) *
                        psi_j_field_minus) +
             (penalty / h) * psi_j_potential_minus * psi_i_potential_minus

             ) *
            fe_face_values.JxW(q);

        //this neighbor      
        vi_ue_matrix(i, j) +=
            (0.5 * (psi_i_field_minus * fe_face_values.normal_vector(q) *
                        psi_j_potential_plus +
                    psi_i_potential_minus * fe_face_values.normal_vector(q) *
                        psi_j_field_plus) -
             (penalty / h) * psi_i_potential_minus * psi_j_potential_plus) *
            fe_face_values.JxW(q);

        //neighbor this
        ve_ui_matrix(i, j) +=
            (-0.5 * (psi_i_field_plus * fe_face_values.normal_vector(q) *
                         psi_j_potential_minus +
                     psi_i_potential_plus * fe_face_values.normal_vector(q) *
                         psi_j_field_minus) -
             (penalty / h) * psi_i_potential_plus * psi_j_potential_minus) *
            fe_face_values.JxW(q);


         //neighbor neighbor
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
    const ComponentSelectFunction<dim> potential_mask(dim,
                                                      dim + 1);
    const ComponentSelectFunction<dim> vectorfield_mask(std::make_pair(0, dim),
                                                        dim + 1);
    double alpha = 0.5;
    const DistanceWeight<dim> distance_weight(alpha, radius, max_diameter); //, radius

    const ProductFunction<dim> connected_function_potential(potential_mask,
                                                            distance_weight);
    const ProductFunction<dim> connected_function_vectorfield(vectorfield_mask,
                                                              distance_weight);

    Vector<double> cellwise_errors_Q(triangulation.n_active_cells());
    Vector<double> cellwise_errors_U(triangulation.n_active_cells());
    pcout << "triangulation.n_active_cells() " << triangulation.n_active_cells()
          << " dof_handler_Omega.n_dofs() " << dof_handler_Omega.n_dofs()
          << " dof_handler_Omega.n_locally_owned_dofs() "
          << dof_handler_Omega.n_locally_owned_dofs() << " solution size "
          << solution.size() << " mpi "
          << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << std::endl;

    const QTrapezoid<1> q_trapez;
    const QIterated<dim> quadrature(q_trapez, degree + 2);

    VectorTools::integrate_difference(
        dof_handler_Omega, solution.block(0), true_solution, cellwise_errors_U, quadrature,
        VectorTools::L2_norm, &connected_function_potential); //
    /*  std::cout<<"cellwise_error.size() "<<cellwise_errors.size()<<std::endl;
     for (unsigned int i = 0; i < cellwise_errors.size(); i++)
      std::cout << cellwise_errors[i] << " "<<std::endl;

      std::cout<<"--------------------"<<std::endl; */
    /*#if USE_MPI_ASSEMBLE
        cellwise_errors.compress(VectorOperation::add); // TODO scauen was es
    noc  // fpr #endif
    */
    potential_l2_error = VectorTools::compute_global_error(
        triangulation, cellwise_errors_U, VectorTools::L2_norm);
    // std::cout<<"mpi  "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<< "
    // potential_l2_error "<<potential_l2_error<<std::endl;
    //  vectorfield Omega
    VectorTools::integrate_difference(
        dof_handler_Omega, solution.block(0), true_solution, cellwise_errors_Q, quadrature,
        VectorTools::L2_norm, &connected_function_vectorfield);

    /*
    #if USE_MPI_ASSEMBLE
        cellwise_errors.compress(VectorOperation::add); // TODO scauen was es
    noc #endif
    */
    vectorfield_l2_error = VectorTools::compute_global_error(
        triangulation, cellwise_errors_Q, VectorTools::L2_norm);

    // std::cout<<"mpi  "<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<< "
    // vectorfield_l2_error "<<vectorfield_l2_error<<std::endl;


    DataOut<dim> data_out;
    data_out.attach_triangulation(triangulation);
    data_out.add_data_vector(cellwise_errors_Q, "Q");
    data_out.add_data_vector(cellwise_errors_U, "U");
    data_out.build_patches();
    std::ofstream output("error.vtk");
    data_out.write_vtk(output);


    std::cout<<"omega"<<std::endl;
    //-------------omega----------------------------------
    //if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      // if (locally_owned_dofs.is_element(dof_index)) TODO allgemeine MPI
      // sachen, auch wenn alle andere 0 sind
      const ComponentSelectFunction<dim_omega> potential_mask_omega(
          dim_omega, dim_omega + 1);
      const ComponentSelectFunction<dim_omega> vectorfield_mask_omega(
          std::make_pair(0, dim_omega), dim_omega + 1);
      Vector<double> cellwise_errors_u(
          triangulation_omega.n_active_cells());
      Vector<double> cellwise_errors_q(
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
          dof_handler_omega, solution.block(1), true_solution_omega,
          cellwise_errors_u, quadrature_omega, VectorTools::L2_norm,
          &potential_mask_omega);

      potential_l2_error_omega = VectorTools::compute_global_error(
          triangulation_omega, cellwise_errors_u, VectorTools::L2_norm);

      /*
       std::cout<<"cellwise_errors_omega_potential ";
         for (unsigned int i = 0; i < cellwise_errors_omega.size(); i++)
          std::cout << cellwise_errors_omega[i] << " ";
           std::cout << std::endl;
      */

      VectorTools::integrate_difference(
          dof_handler_omega, solution.block(1), true_solution_omega,
          cellwise_errors_q, quadrature_omega, VectorTools::L2_norm,
          &vectorfield_mask_omega);

      vectorfield_l2_error_omega = VectorTools::compute_global_error(
          triangulation_omega, cellwise_errors_q, VectorTools::L2_norm);

  /*  } else {
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
        */
  }

 /* return std::array<double, 4>{{potential_l2_error, vectorfield_l2_error,
                                std::sqrt(global_potential_l2_error_omega),
                                std::sqrt(global_vectorfield_l2_error_omega)}};*/

 return std::array<double, 4>{{potential_l2_error, vectorfield_l2_error,
                                potential_l2_error_omega,
                                vectorfield_l2_error_omega}};
}

template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::solve() {
  TimerOutput::Scope t(computing_timer, "solve");
  pcout << "Solving linear system... "<<std::endl;
solution = system_rhs;
/*Utilities::MPI::Partitioner partitioner(complete_index_set(nof_degrees),MPI_COMM_WORLD);
const unsigned int local_size = partitioner.locally_owned_range().size();
const unsigned int n_elements = partitioner.locally_owned_range().n_elements();
std::cout<<"nof_degrees "<<nof_degrees<< " local_size "<<local_size<<" n_elements "<<n_elements<<std::endl;
for(auto idx = partitioner.locally_owned_range().begin(); idx != partitioner.locally_owned_range().end(); idx++)
  std::cout<<*idx<<std::endl;

//system_matrix_mpi.reinit(partitioner, partitioner, MPI_COMM_WORLD);
const auto locally_owned = partitioner.locally_owned_range();
//std::cout<<locally_owned[1]<<std::endl;*/


#if 0//USE_MPI_ASSEMBLE
  std::cout<< dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" systemmatrix: m "<<system_matrix.m() << " n "<<system_matrix.n()<< " local size "<<system_matrix.local_size()
  <<" local range "<<system_matrix.local_range().first<<" "<<system_matrix.local_range().second<<std::endl;


  TrilinosWrappers::MPI::Vector completely_distributed_solution(
      locally_owned_dofs, MPI_COMM_WORLD);

  SolverControl solver_control(dof_handler_Omega.n_dofs(), 1e-12);
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
   const unsigned int max_iterations = solution.size();
    SolverControl      solver_control(max_iterations);

  Timer timer;

 SparseDirectUMFPACK A_direct;
  //A_direct.solve(system_matrix, solution);


  /*  SolverGMRES<BlockVector<double>> solver_direct(solver_control);//
   PreconditionJacobi<BlockSparseMatrix<double>> preconditioner;
   preconditioner.initialize(system_matrix, 1.0);
  solver_direct.solve(system_matrix, solution, system_rhs, preconditioner);//PreconditionIdentity()*/
 /* SparseMatrix sparse_matrix;
  sparse_matrix.reinit()*/

  // SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver_direct(solver_control);//
  //TrilinosWrappers::PreconditionBlockJacobi preconditioner;
  // preconditioner.initialize(system_matrix);
  // solver_direct.solve(system_matrix, solution, system_rhs);//PreconditionIdentity() , preconditioner
        TrilinosWrappers::MPI::BlockVector completely_distributed_solution(
        system_rhs);
  completely_distributed_solution = solution;


  const InverseMatrix A_inverse(system_matrix.block(0,0));

  
  TrilinosWrappers::MPI::Vector tmp(completely_distributed_solution.block(0));

 
  TrilinosWrappers::MPI::Vector schur_rhs(system_rhs.block(1));
  A_inverse.vmult(tmp, system_rhs.block(0));
  system_matrix.block(1, 0).vmult(schur_rhs, tmp);
  schur_rhs -= system_rhs.block(1);
 // schur_rhs.print(std::cout);

 SchurComplement schur_complement(system_matrix, A_inverse, system_rhs);


  SolverControl solver_control1(1000);//completely_distributed_solution.block(1).local_size()
  SolverGMRES<TrilinosWrappers::MPI::Vector > solver(solver_control1);
 
  // SparseILU<double> preconditioner;
  // preconditioner.initialize(preconditioner_matrix.block(1, 1),
  //                           SparseILU<double>::AdditionalData());

  // InverseMatrix<SparseMatrix<double>, SparseILU<double>> m_inverse(
  //   preconditioner_matrix.block(1, 1), preconditioner);

TrilinosWrappers::PreconditionILU preconditioner;
  TrilinosWrappers::PreconditionILU::AdditionalData data;
  preconditioner.initialize(system_matrix.block(1, 1), data);

  solver.solve(schur_complement, completely_distributed_solution.block(1),schur_rhs, preconditioner);
  pcout<<"Schur complete "<<std::endl;

  system_matrix.block(0, 1).vmult(tmp, completely_distributed_solution.block(1));
  tmp *= -1;
  tmp += system_rhs.block(0);
 
  A_inverse.vmult(completely_distributed_solution.block(0), tmp);

 // A_inverse.vmult(completely_distributed_solution.block(0), system_rhs.block(0));//unkoppled

    constraints.distribute(completely_distributed_solution);

  solution = completely_distributed_solution;

#if 0
BlockVector<double> rhs =  BlockVector<double>(system_rhs);

SparseDirectUMFPACK K_inv_umfpack;
SparseMatrix<double> sparse_matrix;
sparse_matrix.copy_from(system_matrix.block(0,0));
K_inv_umfpack.initialize(system_matrix.block(0,0));

auto K  = linear_operator(system_matrix.block(0,0));
auto k = linear_operator(system_matrix.block(1,1));

auto Ct = linear_operator(system_matrix.block(0,1));
//auto C  = transpose_operator(Ct);
auto C = linear_operator(system_matrix.block(1,0));

SolverCG<Vector<double>> solver_cg(solver_control);
SolverGMRES<Vector<double>> solver_gmres(solver_control);

//PreconditionJacobi<SparseMatrix<double>> preconditioner_K;
//preconditioner_K.initialize(sparse_matrix, 1.0);
/*
PreconditionBlockJacobi<SparseMatrix<double>> preconditioner_K;
PreconditionBlockJacobi<SparseMatrix<double>>::AdditionalData data(fe_Omega.dofs_per_cell);
preconditioner_K.initialize(system_matrix.block(0,0), data);
 */
/*PreconditionLU<double> LU;
 LAPACKFullMatrix<double> m;
 m.copy_from(system_matrix.block(0,0));
LU.initialize(m);*/

ReductionControl reduction_control_K(2000, 1.0e-18, 1.0e-10);
SolverGMRES<Vector<double>>  solver_K(solver_control);


//auto K_inv= inverse_operator(K, solver_K, preconditioner_K );
auto K_inv = linear_operator(K, K_inv_umfpack);
 //TODO das paralkek macen. akso alle cellen mit wurzel auf einen processsor andere verteilen
auto S = k - C * K_inv * Ct;

auto S_inv = inverse_operator(S, solver_gmres, PreconditionIdentity());
 


auto temp = rhs.block(1)- C * K_inv * rhs.block(0);
solution.block(1) = S_inv * ( temp);
Vector<double> sol = Vector(solution.block(1));
solution.block(0) = K_inv * (rhs.block(0) - Ct * sol);


/*auto temp = system_rhs.block(1)- C * K_inv * system_rhs.block(0);
solution.block(1) = S_inv * ( temp);
solution.block(0) = K_inv * (system_rhs.block(0) - Ct * solution.block(1)); */
#endif
#endif
#if 0
      TrilinosWrappers::MPI::Vector tmp_Omega = solution.block(0);
      TrilinosWrappers::MPI::Vector tmp_omega = solution.block(1);
      TrilinosWrappers::SparseMatrix& K = system_matrix.block(0, 0);
      TrilinosWrappers::SparseMatrix& k = system_matrix.block(1, 1);

      TrilinosWrappers::SparseMatrix& Ct = system_matrix.block(0,1);
      TrilinosWrappers::SparseMatrix& C = system_matrix.block(1,0);

     TrilinosWrappers::MPI::Vector& f_Omega =  system_rhs.block(0);
     TrilinosWrappers::MPI::Vector& f_omega =  system_rhs.block(0);

#endif
  timer.stop();
  std::cout << "done (" << timer.cpu_time() << "s)" << std::endl;
 
  int rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  std::cout<<"rank "<<rank<<std::endl;
  /*if (rank == 0)
    {
        std::cout << "Full BlockVector (rank 0):" << std::endl;

        for (unsigned int b = 0; b < 2; ++b)
        {
           const unsigned int global_size =  solution.block(b).size();
            std::cout << "Block " << b << ":" << std::endl;

            // Gather the full vector for this block
            //std::vector<double> full_block(global_size);
          // solution.block(b).gather(full_block);

            // Print the full block
            for (unsigned int i = 0; i < global_size; ++i)
            {
              //  std::cout << full_block[i] << " ";
            }
            std::cout << std::endl;
        }
    }*/

/*
  solution_Omega.reinit(dof_handler_Omega.n_dofs());
  for (unsigned int i = 0; i < dof_handler_Omega.n_dofs(); i++) {
    types::global_dof_index dof_index =  i;
#if 1// USE_MPI_ASSEMBLE
    if (locally_owned_dofs_Omega.is_element(dof_index))
#endif
{
      solution_Omega[i] = solution[dof_index];
      //std::cout<<solution[dof_index]<<" ";
      }
      else
      {
       // std::cout<< "**" <<" ";
       }
  }*/

//  std::cout<<"------------------------" <<std::endl;
/*
  solution_omega.reinit(dof_handler_omega.n_dofs());
  for (unsigned int i = 0; i < dof_handler_omega.n_dofs(); i++) {
    types::global_dof_index dof_index = start_VectorField_omega + i;
#if 0// USE_MPI_ASSEMBLE
    if (locally_owned_dofs.is_element(dof_index))
#endif
      solution_omega[i] = solution[dof_index];
  }
*/
#if 0// USE_MPI_ASSEMBLE
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
    solution_names.push_back("U");

    break;

  case 2:
    solution_names.push_back("Q_x");
    solution_names.push_back("Q_y");
    solution_names.push_back("U");
    break;

  case 3:
    solution_names.push_back("Q_x");
    solution_names.push_back("Q_y");
    solution_names.push_back("Q_z");
    solution_names.push_back("U");
    break;

  default:
    Assert(false, ExcNotImplemented());
  }

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_Omega);
   /*    Postprocessor postprocessor(Utilities::MPI::this_mpi_process(
                                    MPI_COMM_WORLD),
                                  solution.min());*/
  
  data_out.add_data_vector(solution.block(0),
                         solution_names); //, DataOut<dim>::type_cell_data  
 
  Vector<float>   subdomain(triangulation.n_active_cells());
  for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
  
  data_out.add_data_vector(subdomain,"subdomain");
  data_out.build_patches(degree);
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



 /*
std::ofstream vtk_file;
  const std::string filename1 = ("system_matrix."   +
                                  Utilities::int_to_string(
                                    triangulation.locally_owned_subdomain(),4));
    vtk_file.open((filename1 + ".vtk").c_str());
    
    // VTK Header
    vtk_file << "# vtk DataFile Version 3.0\n";
    vtk_file << "Sparse Matrix visualization\n";
    vtk_file << "ASCII\n";
    vtk_file << "DATASET STRUCTURED_POINTS\n";
    vtk_file << "DIMENSIONS " << system_matrix.block(0,0).m() << " " << system_matrix.block(0,0).n() << " 1\n";
    vtk_file << "SPACING 1 1 1\n";
    vtk_file << "ORIGIN 0 0 0\n";
    vtk_file << "POINT_DATA " << system_matrix.block(0,0).m() * system_matrix.block(0,0).n() << "\n";
    vtk_file << "SCALARS matrix_values float\n";
    vtk_file << "LOOKUP_TABLE default\n";

    // Write matrix entries
    for (unsigned int i = 0; i < system_matrix.block(0,0).m(); ++i) {
        for (unsigned int j = 0; j < system_matrix.block(0,0).n(); ++j) {
            vtk_file << system_matrix.block(0,0).el(i, j) << "\n";
        }
    }

    vtk_file.close();*/
    
    /*
    std::cout<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<< " solution "<<std::endl;
    //solution.block(0).print(std::cout);
for (unsigned int i = 0; i < dof_handler_Omega.n_dofs(); i++) {
    types::global_dof_index dof_index =  i;
    if (locally_owned_dofs_Omega.is_element(dof_index))
      {
      std::cout<<solution[dof_index]<<" ";
      }
      else
      {
        std::cout<< "**" <<" ";
       }
  }
    std::cout<<std::endl;
    std::cout<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<< " system_rhs "<<std::endl;
for (unsigned int i = 0; i < dof_handler_Omega.n_dofs(); i++) {
    types::global_dof_index dof_index =  i;
    if (locally_owned_dofs_Omega.is_element(dof_index))
      {
      std::cout<<system_rhs[dof_index]<<" ";
      }
      else
      {
        std::cout<< "**" <<" ";
       }
  }
  std::cout<<std::endl;
*/
 // ------analytical solution--------
 /* std::cout << "analytical solution" << std::endl;
  DoFHandler<dim> dof_handler_Lag(triangulation);
  FESystem<dim> fe_Lag(FESystem<dim>(FE_DGQ<dim>(degree), dim),
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
  data_out_const.write_vtu(output_const);*/

   //-----omega-----------
  std::cout << "omega solution" << std::endl;
  std::vector<std::string> solution_names_omega;
  solution_names_omega.emplace_back("q");
  solution_names_omega.emplace_back("u");

 DataOut<dim_omega> data_out_omega;
  data_out_omega.attach_dof_handler(dof_handler_omega);

  data_out_omega.add_data_vector(solution.block(1),
                         solution_names_omega); //, DataOut<dim>::type_cell_data  
 
  Vector<float>   subdomain_omega(triangulation_omega.n_active_cells());
  for (unsigned int i=0; i<subdomain_omega.size(); ++i)
      subdomain_omega(i) = triangulation_omega.locally_owned_subdomain();
  
  data_out_omega.add_data_vector(subdomain_omega,"subdomain");
  data_out_omega.build_patches(degree);
    const std::string filename_omega = ("solution_omega."   +
                                  Utilities::int_to_string(
                                    triangulation_omega.locally_owned_subdomain(),4));
  std::ofstream output_omega((filename_omega + ".vtu").c_str());
  data_out_omega.write_vtu(output_omega);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
      {
        std::vector<std::string>    filenames;
        for (unsigned int i=0;
             i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
             i++)
          {
            filenames.push_back("solution_omega." +
                                Utilities::int_to_string(i,4) +
                                ".vtu");
          }
        std::ofstream master_output("solution_omega.pvtu");
        data_out_omega.write_pvtu_record(master_output, filenames);
      }





}

template <int dim, int dim_omega>
std::array<double, 4> LDGPoissonProblem<dim, dim_omega>::run() {
  pcout << "n_refine " << n_refine << "  degree " << degree << std::endl;

  penalty = 5;
  make_grid();
  make_dofs();
  assemble_system();
  solve();
  //output_results();
  std::array<double, 4> results_array= compute_errors();
  return results_array;
}

int main(int argc, char *argv[]) {
  std::cout << "USE_MPI_ASSEMBLE " << USE_MPI_ASSEMBLE << std::endl;
#if 1
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

  /*
  Parameters parameters;
      parameters.radius = 0.01;
      parameters.lumpedAverage = true;
  
  LDGPoissonProblem<dimension_Omega, 1> LDGPoissonCoupled_s(0,3, parameters);
    std::array<double, 4> arr = LDGPoissonCoupled_s.run();
    std::cout << rank << " Result_ende: U " << arr[0] << " Q " << arr[1] << " u "
              << arr[2] << " q " << arr[3] << std::endl;
    return 0;
  */
  std::cout << "dimension_Omega " << dimension_Omega << std::endl;
  const unsigned int n_r = 2;
  const unsigned int n_LA = 2;
  double radii[n_r] = { 0.01,0.1};
  bool lumpedAverages[n_LA] = { false, true};
  std::vector<std::array<double, 4>> result_scenario;
  std::vector<std::string> scenario_names;
  for (unsigned int rad = 0; rad < n_r; rad++) {
    for (unsigned int LA = 0; LA < n_LA; LA++) {

      std::string LA_string = lumpedAverages[LA] ? "true" : "false";
      std::string radius_string = std::to_string(radii[rad]);
      std::string name = "_LA_" + LA_string + "_rad_" + radius_string;
      scenario_names.push_back(name);

      Parameters parameters;
      parameters.radius = radii[rad];
      parameters.lumpedAverage = lumpedAverages[LA];
      const unsigned int p_degree[2] = {0,1};
      constexpr unsigned int p_degree_size =
          sizeof(p_degree) / sizeof(p_degree[0]);
    //  const unsigned int refinement[6] = {2,3,4,5,6,7};
      const unsigned int refinement[6] = {3,4, 5, 6,7,8};

      constexpr unsigned int refinement_size =
          sizeof(refinement) / sizeof(refinement[0]);

      std::array<double, 4> results[p_degree_size][refinement_size];
      double max_diameter[refinement_size];

      std::vector<std::string> solution_names = {"U_Omega", "Q_Omega",
                                                 "u_omega", "q_omega"};
                                         
      for (unsigned int r = 0; r < refinement_size; r++) {
        for (unsigned int p = 0; p < p_degree_size; p++) {
          LDGPoissonProblem<dimension_Omega, 1> LDGPoissonCoupled =
              LDGPoissonProblem<dimension_Omega, 1>(p_degree[p], refinement[r],
                                                    parameters);
          std::array<double, 4> arr;
          try
          {
            arr = LDGPoissonCoupled.run();
          }
          catch(const std::exception& e)
          {
           std::cout  << e.what() << std::endl;
           arr = {42,42,42,42};
          }
          

          std::cout << rank << " Result_ende: U " << arr[0] << " Q " << arr[1]
                    << " u " << arr[2] << " q " << arr[3] << std::endl;
          results[p][r] = arr;
          max_diameter[r] = LDGPoissonCoupled.max_diameter;
        }
      }

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
        // std::cout << "--------" << std::endl;
        std::ofstream myfile;
        std::ofstream csvfile;
#if COUPLED
        std::string filename = "convergence_results_coupled" + name;
        myfile.open(filename + ".txt");
        csvfile.open(filename + ".csv");
#else
        std::string filename = "convergence_results_uncoupled" + name;
        myfile.open(filename + ".txt");
        csvfile.open(filename + ".csv");
#endif
        for (unsigned int f = 0; f < solution_names.size(); f++) {
          myfile << solution_names[f] << "\n";
          myfile << "refinement/p_degree, ";
          myfile << "diameter h;";

          csvfile << solution_names[f] << "\n";
          csvfile << "refinement/p_degree;";
          csvfile << "diameter h;";

          std::cout << solution_names[f] << "\n";
          std::cout << "refinement/p_degree;";
          std::cout << "diameter h;";
          for (unsigned int p = 0; p < p_degree_size; p++) {
            myfile << p_degree[p] << ",";
            csvfile << p_degree[p] << ";";
            std::cout << p_degree[p] << ";";
          }
          myfile << "\n";
          csvfile << "\n";
          std::cout << "\n";
          for (unsigned int r = 0; r < refinement_size; r++) {
            myfile << refinement[r] << ";" << max_diameter[r] << ";";
            csvfile << refinement[r] << ";" << max_diameter[r]  << ";";
            std::cout << refinement[r] <<";" << max_diameter[r] << ";";
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
