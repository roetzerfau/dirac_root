
//  @sect3{LDGPoisson.cc}

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/config.h>
//#include <deal.II/distributed/cell_weights.h>
#include <deal.II/distributed/tria.h>

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


#include <deal.II/fe/fe_dgq.h>

#include <deal.II/fe/fe_dgp.h>
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
#include <deal.II/lac/dynamic_sparsity_pattern.h>


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


#include <deal.II/grid/tria.h>           // For Triangulation
#include <deal.II/grid/grid_tools.h>      // For GridTools
#include <deal.II/grid/grid_tools_cache.h>
#include <deal.II/grid/grid_generator.h>   // For GridGenerator
#include <deal.II/base/timer.h>           // For Timer (if needed)
#include <deal.II/base/logstream.h>       // For logging
#include <deal.II/grid/grid_out.h>
//#include <boost/archive/text_oarchive.hpp>
//#include <boost/archive/text_iarchive.hpp>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/mapping_q1_eulerian.h>
#include <deal.II/lac/vector_memory.h>


#include <fstream>
#include <iostream>
#include <stdexcept>
#include <malloc.h>
#include <sys/resource.h>
#include <numeric> 

#include "Functions.cc"

using namespace dealii;



//Geometrie
//case 1: 2D/0Dv-> im hintergrund iwrd trotzdem noch 1D problem gelöst
//case 2: 2D/1D 
//case 3: 3D/1D




const FEValuesExtractors::Vector VectorField_omega(0);
const FEValuesExtractors::Scalar Potential_omega(1);

const FEValuesExtractors::Vector VectorField(0);
const FEValuesExtractors::Scalar Potential(dimension_Omega);


const double extent = 1.0;
const double alpha = 1.0;
const double half_length =  is_omega_on_face ? std::sqrt(0.5): std::sqrt(0.5);//0.5  -sqrt(2)* 0.001
const double distance_tolerance = 100;//100
const unsigned int N_quad_points = 10;
const double reduction = 1e-8;
const double tolerance = 1e-10;

double arrr = 0.000;
struct Parameters {
  double radius;
  bool lumpedAverage;
  std::string folder_name;
};
size_t getCurrentRSS() {
    std::ifstream statm("/proc/self/statm");
    if (!statm.is_open()) {
        std::cerr << "Could not open /proc/self/statm" << std::endl;
        return 0;
    }

    size_t rss = 0;
    statm >> rss;  // Read the number of resident pages
    statm.close();

    long pageSize = sysconf(_SC_PAGESIZE);  // Get the page size in bytes
    return rss * pageSize;  // Convert pages to bytes
}



template <int dim>
void project 	( 	const Mapping< dim> & 	mapping,
		const DoFHandler< dim> & 	dof,
		const Quadrature< dim > & 	quadrature,
		const Function<dim> & 	function,
		TrilinosWrappers::MPI::Vector & 	vec, const IndexSet& locally_owned_dofs, int degree)
    {

  TrilinosWrappers::SparsityPattern trilinos_sparsity(locally_owned_dofs, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof, trilinos_sparsity,{},true,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    trilinos_sparsity.compress();
TrilinosWrappers::SparseMatrix system_matrix;
system_matrix.reinit(trilinos_sparsity);
TrilinosWrappers::MPI::Vector rhs(vec);
MatrixCreator::create_mass_matrix(mapping, dof, QGauss<dim>(degree +1), system_matrix);
VectorTools::create_right_hand_side(dof,QGauss<dim>(degree +1), function, rhs);
/*std::cout<<"system matrix"<<std::endl;
system_matrix.print(std::cout);
std::cout<<"rhs"<<std::endl;
rhs.print(std::cout);*/
    // Step 3: Solve the system
    SolverControl solver_control(1000, 1e-12);
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

    TrilinosWrappers::PreconditionIdentity precondition;
   // precondition.initialize(system_matrix);

    solver.solve(system_matrix, vec, rhs, precondition);

    }



class BlockPreconditioner : public dealii::Subscriptor {
public:
    BlockPreconditioner(TrilinosWrappers::PreconditionILU &precond0, TrilinosWrappers::PreconditionILU &precond1)
        : preconditioner0(precond0), preconditioner1(precond1) {}

    void vmult(TrilinosWrappers::MPI::BlockVector &dst, const TrilinosWrappers::MPI::BlockVector &src) const {
        TrilinosWrappers::MPI::Vector temp0(dst.block(0));
        TrilinosWrappers::MPI::Vector temp1(dst.block(1));

        preconditioner0.vmult(temp0, src.block(0));
        preconditioner1.vmult(temp1, src.block(1));

        dst.block(0) = temp0;
        dst.block(1) = temp1;
    }

private:
    TrilinosWrappers::PreconditionILU &preconditioner0;
    TrilinosWrappers::PreconditionILU &preconditioner1;
};
  class InverseMatrix : public Subscriptor
  {
  public:
    InverseMatrix(const TrilinosWrappers::SparseMatrix &m)
                      : matrix(&m)
    {
    }
 
    void vmult(TrilinosWrappers::MPI::Vector       &dst,
                   const TrilinosWrappers::MPI::Vector &src) const
    {
    dst = 0;
    //std::cout<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" src "<<std::endl;
    //src.print(std::cout);
   if(dimension_Omega == 2)
    {
    //ReductionControl solver_control(src.size(), tolerance * src.l2_norm(), reduction);
    SolverControl solver_control(src.size(), tolerance );
    TrilinosWrappers::SolverDirect solver(solver_control);
    solver.initialize(*matrix);
    solver.solve(dst,src);
    }
    else{
TrilinosWrappers::PreconditionILUT preconditioner;
  TrilinosWrappers::PreconditionILUT::AdditionalData data;
  preconditioner.initialize(*matrix, data);

    //ReductionControl solver_control(matrix->local_size(), tolerance * src.l2_norm(), reduction);//, 1e-7 * src.l2_norm());
    SolverControl solver_control(std::max( (int) matrix->local_size(),1000), tolerance );//, 1e-7 * src.l2_norm());
    TrilinosWrappers::SolverGMRES solver(solver_control);
    solver.solve(*matrix, dst,  src, preconditioner );
    }
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
      , tmp1(block_vector.block(0))
      , tmp2(block_vector.block(0))
      , tmp3(block_vector.block(1))
      , tmp4(block_vector.block(1))

      {
      }

      void vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const
      {

      system_matrix->block(0, 1).vmult(tmp1, src);
      A_inverse->vmult(tmp2, tmp1);
      system_matrix->block(1, 0).vmult(tmp3, tmp2);
       system_matrix->block(1, 1).vmult(tmp4, src);
       dst = tmp3- tmp4;
      }
  
  
    private:
      const SmartPointer<const TrilinosWrappers::BlockSparseMatrix> system_matrix;
      const SmartPointer< const InverseMatrix> A_inverse;
  
      mutable TrilinosWrappers::MPI::Vector tmp1, tmp2, tmp3, tmp4;
 
    };



class SchurComplement_A_22 : public Subscriptor
    {
    public:
      SchurComplement_A_22(
        const TrilinosWrappers::BlockSparseMatrix &system_matrix,
        const InverseMatrix &A_inverse, 
        const TrilinosWrappers::MPI::BlockVector &block_vector)
         : system_matrix(&system_matrix)
        , A_inverse(&A_inverse)
    /*  , tmp1(block_vector.block(1))
      , tmp2(block_vector.block(1))
      , tmp3(block_vector.block(0))
      , tmp4(block_vector.block(0))*/

      {
      }

      void vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const
      {
    //PrimitiveVectorMemory<
      TrilinosWrappers::MPI::Vector tmp1(system_matrix->block(1,1).locally_owned_range_indices());
      TrilinosWrappers::MPI::Vector tmp2(system_matrix->block(1,1).locally_owned_range_indices());
      TrilinosWrappers::MPI::Vector tmp3(system_matrix->block(0,0).locally_owned_range_indices());
      TrilinosWrappers::MPI::Vector tmp4(system_matrix->block(0,0).locally_owned_range_indices());

      system_matrix->block(1,0).vmult(tmp1, src);
    
      A_inverse->vmult(tmp2, tmp1);
      system_matrix->block(0,1).vmult(tmp3, tmp2);
     
       system_matrix->block(0, 0).vmult(tmp4, src);
       dst = tmp3- tmp4;
  
      }
  
  
    private:
      const SmartPointer<const TrilinosWrappers::BlockSparseMatrix> system_matrix;
      const SmartPointer< const InverseMatrix> A_inverse;
  
     // mutable TrilinosWrappers::MPI::Vector tmp1, tmp2, tmp3, tmp4;
      
    };



template <int dim, int dim_omega> class LDGPoissonProblem {

public:
  LDGPoissonProblem(const unsigned int degree, const unsigned int n_refine,
                    Parameters parameters);

  ~LDGPoissonProblem();

  std::array<double, 4> run();
  double max_diameter;
  double max_diameter_omega;
  unsigned int nof_cells;
  unsigned int nof_cells_omega;

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
                           const FEValuesExtractors::Scalar &Potential,
                           bool no_gradient);
  template <int _dim>
  void assemble_Neumann_boundary_terms(
      const FEFaceValues<_dim> &face_fe, FullMatrix<double> &local_matrix,
      Vector<double> &local_vector, const Function<_dim> &Neumann_bc_function, 
       const FEValuesExtractors::Vector VectorField,
    const FEValuesExtractors::Scalar Potential);

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
  void dof_omega_local_2_global(
      const DoFHandler<_dim> &dof_handler_Omega,
      std::vector<types::global_dof_index> &local_dof_indices_omega);

  void solve();

  std::array<double, 4> compute_errors() ;//const
  void output_results() const;

  void memory_consumption(std::string _name);
  const unsigned int degree;
  const unsigned int n_refine;
  double penalty;
  double h_max;
  double h_min;
  
  double minimal_cell_diameter;
  double maximal_cell_diameter;
  double minimal_cell_diameter_2D;
  double maximal_cell_diameter_2D;

  

  //unsigned int nof_degrees;
  unsigned int dimension_gap;
  bool AVERAGE;
  unsigned int nof_quad_points = 1;

  int rank_mpi;
  enum { NotDefined, Dirichlet, Neumann };

  // parameters
  double radius;
  double g;
  bool lumpedAverage;
  std::string folder_name;

  //parallel::distributed::Triangulation<dim> triangulation_dist;

  //parallel::shared::Triangulation<dim> triangulation_mpi;

  parallel::shared::Triangulation<dim> triangulation;
  GridTools::Cache<dim, dim> cache;
  unsigned int cell_weight(
      const typename  parallel::distributed::Triangulation<dim>::cell_iterator 
                      &cell,
      const typename  parallel::distributed::Triangulation<dim>::CellStatus status) const;
  BoundingBox<dim> bbox;

  TrilinosWrappers::BlockSparseMatrix system_matrix;
  TrilinosWrappers::MPI::BlockVector solution;
  TrilinosWrappers::MPI::BlockVector system_rhs;

  FESystem<dim> fe_Omega;
  DoFHandler<dim> dof_handler_Omega;

  parallel::shared::Triangulation<dim_omega> triangulation_omega;
  FESystem<dim_omega> fe_omega;
  DoFHandler<dim_omega> dof_handler_omega;


  Vector<double> solution_omega;
  Vector<double> cellwise_errors_U;
  Vector<double> cellwise_errors_Q;
  Vector<double> cellwise_errors_u;
  Vector<double> cellwise_errors_q;

/*
  IndexSet locally_owned_dofs_Omega;
  IndexSet locally_relevant_dofs_Omega;

  IndexSet locally_owned_dofs_omega_local;
  IndexSet locally_relevant_dofs_omega_local;
*/
  // IndexSet locally_owned_dofs_omega_global;
 /// IndexSet locally_relevant_dofs_omega_global;

  //IndexSet locally_owned_dofs_total;
  //IndexSet locally_relevant_dofs_total;



  AffineConstraints<double> constraints;

  std::vector<bool> marked_vertices;



  ConditionalOStream pcout;
  TimerOutput computing_timer;

  const RightHandSide<dim> rhs_function;
  const KInverse<dim> K_inverse_function;
  const DirichletBoundaryValues<dim> Dirichlet_bc_function;
  const NeumannBoundaryValues<dim> Neumann_bc_function;
   const NeumannBoundaryValues_omega<dim_omega> Neumann_bc_function_omega;
  const TrueSolution<dim> true_solution;
  const TrueSolution_omega<dim_omega> true_solution_omega;

  const RightHandSide_omega<dim_omega> rhs_function_omega;
  const KInverse<dim_omega> k_inverse_function;
  const DirichletBoundaryValues_omega<dim_omega> Dirichlet_bc_function_omega;

  unsigned int start_VectorField_omega;
  unsigned int start_Potential_omega;
  unsigned int start_Potential_Omega;

  const UpdateFlags update_flags = update_values | update_gradients |
                                   update_quadrature_points | update_JxW_values;
  const UpdateFlags update_flags_coupling = update_values | update_JxW_values| update_quadrature_points | update_normal_vectors ;//|update_gradients

  const UpdateFlags face_update_flags = update_values | update_normal_vectors |
                                        update_quadrature_points |
                                        update_JxW_values;
};

template <int dim, int dim_omega>
LDGPoissonProblem<dim, dim_omega>::LDGPoissonProblem(
    const unsigned int degree, const unsigned int n_refine,
    Parameters parameters)
    : degree(degree), n_refine(n_refine),
      triangulation(MPI_COMM_WORLD),//, parallel::shared::Triangulation<dim>::none, false, parallel::shared::Triangulation<dim>::Settings::partition_zorder),
      //triangulation_dist(MPI_COMM_WORLD),
      cache(triangulation),
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
      rhs_function(),
      Dirichlet_bc_function(), rhs_function_omega(),
      Dirichlet_bc_function_omega(), radius(parameters.radius),
      lumpedAverage(parameters.lumpedAverage), folder_name(parameters.folder_name) {

  g = constructed_solution == 3 || constructed_solution == 2
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
  pcout<<"make grid"<<std::endl;

  double offset = 0.0;

 /* triangulation.signals.weight.connect(
        [&](const typename parallel::distributed::Triangulation<
              dim>::cell_iterator &cell,
          const typename parallel::distributed::Triangulation<dim>::CellStatus status) -> unsigned int {
          return this->cell_weight(cell, status);
        });*/
#if CYLINDER
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
#else
  Point<dim> p1, p2;
if (dim == 3) {
   p1 =
      Point<dim>(2*half_length, -half_length + offset, -half_length + offset);
   p2 =
      Point<dim>(0, half_length + offset, half_length + offset);
}
 if (dim == 2) {
   p1 =
      Point<dim>(-half_length + offset, -half_length + offset);	
   p2 =
      Point<dim>(half_length + offset, half_length + offset);

 }

pcout<<"grid extent, p1:  "<<p1 <<" p2: "<<p2<<std::endl;
//std::vector< unsigned int > 	repetitions({std::pow(2,n_refine),std::pow(2,n_refine),3});
std::vector< unsigned int > 	repetitions({1,1,1});
GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,  p1, p2);//_with_simplices
 //GridGenerator::hyper_rectangle(triangulation,  p1, p2);
 
#endif
triangulation.refine_global(n_refine);
pcout<<"refined++++++ "<<n_refine<<" refined global level "<<triangulation.n_global_levels()-1<<std::endl;

/*for (unsigned int i =0; i <n_refine; ++i)
{
  typename Triangulation<dim>::active_cell_iterator
  cell = triangulation.begin_active(),
  endc = triangulation.end();
  for (; cell != endc; ++cell)
  {
if(dim == 3)
{
 // pcout<<"dim == 3"<<std::endl;
 // pcout<<int(RefinementCase<dim>::cut_y)<< " " <<int(RefinementCase<dim>::cut_z)<<" "<<int(RefinementCase<dim>::cut_y | RefinementCase<dim>::cut_z)<<std::endl;
    if(i <= n_refine - refinement[0])
     cell->set_refine_flag();
    else
      cell->set_refine_flag(RefinementCase<dim>(6));//RefinementCase<dim>::cut_y | RefinementCase<dim>::cut_z
    
     
}
else
      cell->set_refine_flag();

  }
  triangulation.execute_coarsening_and_refinement();
}*/
 int level_max = n_refine;

 //pcout<<"3D maximal_cell_diameter "<<  GridTools::maximal_cell_diameter(triangulation)<<" std::pow(maximal_cell_diameter,2) "<<std::pow(GridTools::maximal_cell_diameter(triangulation),2)<<std::endl;
unsigned int refine_omega =  n_refine;//n_refine - refinement[0] + 1;
#if GRADEDMESH
#if ANISO
 double h_max = (2 * half_length)/std::pow(2,n_refine)* std::sqrt(2);
#else
double  h_max = GridTools::maximal_cell_diameter(triangulation);
#endif 
 //h_max = dim == 3 ? h_max * std::sqrt(3) :  h_max  * std::sqrt(2);

 //
 pcout<<"h_max "<<h_max<<" level_max "<<level_max<<std::endl;
 double mu = alpha/(degree + 1);
 double delta = 1.0;
 pcout<<"mu "<<mu<<std::endl;
 for (unsigned int i =n_refine; i <n_refine * 5; ++i)
    {
      typename Triangulation<dim>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();
   for (; cell != endc; ++cell)
        {
        double r = 0, r_max = 0, r_min = 10000;
        for(unsigned int i = 0; i < cell->n_vertices();i++)  
        {
          double r = distance_to_singularity<dim>(cell->vertex(i))* 1.0000001 ;//TODO nicht center, sondern inf or sup  
          if(r_max < r)
            r_max = r;
          if(r_min > r)
          r_min = r;
          
        }
          
        r = r_min;//r_min im paper
    
    cell->clear_refine_flag();
      
      //std::cout<<"level "<<cell->level()<<std::endl;
      //if(cell->point_inside(nearest_point_on_singularity(cell->center())))//dieser Zelle enthält singularität, aber falsch, da schräge linien nicht beachtet
   
      // //TODO eigentlich wenn innerhalb der singularität celle

#if ANISO
     // if(r < 2 * half_length/std::pow(2,triangulation.n_global_levels()-1) * 1.1 * std::sqrt(2))//innere Bereich
     if(r <= delta  * (2 * half_length)/std::pow(2,cell->level()) *  std::sqrt(2))//innere Bereich
     //if(r <  0.00001)
#else
      //if(r <  GridTools::minimal_cell_diameter(triangulation)* 1.1)
      if(r < 0.5 * cell->diameter())
#endif
      {
     //   std::cout<<"aaaa"<<std::endl;
        //if(r <= 0.0)
        //std::cout<<"aaaaa "<<cell->level()<<" " <<2 * half_length/std::pow(2,cell->level())* std::sqrt(2)<< " " << std::pow(h_max,1.0/mu)<<std::endl;
#if ANISO        
        if(2 * half_length/std::pow(2,cell->level())* std::sqrt(2) > std::pow(h_max,1.0/mu))  
#else
        if(cell->diameter() >  std::pow(h_max,1.0/mu))
#endif
        {
         //std::cout<<"refine"<<std::endl;
if(dim == 3 && ANISO)
       cell->set_refine_flag(RefinementCase<dim>(6));//RefinementCase<dim>::cut_y | RefinementCase<dim>::cut_z
else
      cell->set_refine_flag();



        }

      }
      else  //äußerer Bereich
      {
#if ANISO       
        if(2 * half_length/std::pow(2,cell->level())* std::sqrt(2) > h_max * std::pow(r,1 - mu)) 
#else
        if(cell->diameter() > h_max * std::pow(r,1 - mu) )// factor um relation * 1.5
#endif
        {
if(dim == 3 &&  ANISO)
       cell->set_refine_flag(RefinementCase<dim>(6));//RefinementCase<dim>::cut_y | RefinementCase<dim>::cut_z
else
       cell->set_refine_flag();
       
        }
      }
    //     else
//std::cout<<"r "<<r<<" h_max * std::pow(r,0.5) "<<h_max * std::pow(r,0.5)<< " cell->diameter() " <<cell->diameter()<<std::endl;
        
        }
      triangulation.execute_coarsening_and_refinement();
    }


#endif

unsigned int level_min = std::numeric_limits<unsigned int >::max();
{
 typename Triangulation<dim>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();
      for (; cell != endc; ++cell)
      {
        level_max = std::max(cell->level(), level_max);
        level_min = std::min((unsigned int)(cell->level()), level_min);
      }
       
pcout<<"level_max "<<level_max<<" level_min "<<level_min<<std::endl;
}




 minimal_cell_diameter = GridTools::minimal_cell_diameter(triangulation);
 maximal_cell_diameter = GridTools::maximal_cell_diameter(triangulation);
 
 minimal_cell_diameter_2D = 2 * half_length/std::pow(2,level_max)* std::sqrt(2);
 maximal_cell_diameter_2D = 2 * half_length/std::pow(2,n_refine)* std::sqrt(2);
arrr =  half_length/std::pow(2,n_refine)/2;
pcout<<"arrr "<<arrr<<std::endl;
 h_min = minimal_cell_diameter_2D;
 pcout<<"2D minimal_cell_diameter "<<minimal_cell_diameter_2D<< " maximal_cell_diameter "<< maximal_cell_diameter_2D;
 #if GRADEDMESH
 pcout<<" std::pow(maximal_cell_diameter,1/mu) "<<std::pow(maximal_cell_diameter_2D,1.0/mu)<<std::endl;
 #else
 pcout<<std::endl;
 #endif

 pcout<<"3D minimal_cell_diameter "<<minimal_cell_diameter<< " maximal_cell_diameter ";
 #if GRADEDMESH
 pcout<<maximal_cell_diameter<<" std::pow(maximal_cell_diameter,1.0/mu) "<<std::pow(maximal_cell_diameter,1.0/mu)<<std::endl;
 #else
 pcout<<std::endl;
 #endif
 
 
 #if MEMORY_CONSUMPTION
 pcout << "Memory consumption of triangulation: "
              << triangulation.memory_consumption() / (1024.0 * 1024.0) // Convert to MB
	             << " MB" << std::endl;
#endif		         
			     unsigned int global_active_cells = triangulation.n_global_active_cells();
			       
				     pcout << "Total number of active cells (global): " << global_active_cells << std::endl;
nof_cells = global_active_cells;

GridOut grid_out;
std::string gradedMesh_string = GRADEDMESH ==1 ? "true" : "false";
std::ofstream out("grid_Omega_gradedMesh_"+ gradedMesh_string +"_n_refine_"+std::to_string(n_refine)+"_degree_"+std::to_string(degree)+ ".vtk"); // Choose your preferred filename and format
grid_out.write_vtk(triangulation, out);


  Point<dim> corner1, corner2;
double margin = 1.0;
double h = minimal_cell_diameter * 2;
if(dim == 3)
{
corner1 =  Point<dim>(0, - (margin*radius + h), - (margin*radius + h));//2*radius
corner2 =  Point<dim>(2 * half_length,  (margin*radius + h),  (margin*radius + h));//radius
}
if(dim == 2)
{
corner1 =  Point<dim>( -half_length,- (margin*radius + h));
corner2 =  Point<dim>(half_length,(margin*radius + h));
}
std::pair<Point<dim>, Point<dim>> corner_pair(corner1, corner2);     
    
 bbox = BoundingBox<dim>(corner_pair);



const std::vector<Point<dim>> &vertices = triangulation.get_vertices();

SparsityPattern cell_connection_graph;
DynamicSparsityPattern connectivity_cells;
std::vector<unsigned int> cells_inside_box;
std::vector<bool> is_cell_inside_box;
std::vector<unsigned int> cell_weights;

int num_processes = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
bool is_shared_triangulation = num_processes > 1 ? true : false;
bool is_repartioned =  is_shared_triangulation && (geo_conf != GeometryConfiguration::TwoD_ZeroD) && COUPLED;
pcout<<"is_shared_triangulation "<<  is_shared_triangulation<<" is_repartioned "<<is_repartioned<<std::endl;
if(is_repartioned)
{
GridTools::get_face_connectivity_of_cells(triangulation,connectivity_cells);
}

 max_diameter = 0.0;
 typename DoFHandler<dim>::active_cell_iterator
        cell = dof_handler_Omega.begin_active(),
        endc = dof_handler_Omega.end();


  unsigned int cell_number = 0;
  for (; cell != endc; ++cell) {
if( is_repartioned)
   {
    bool cell_is_inside_box = false;
    for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
    if (bbox.point_inside(vertices[cell->vertex_index(v)]))
    {
     cell_is_inside_box = true;
    break;
    }
   
    }
    is_cell_inside_box.push_back(cell_is_inside_box);
    if(cell_is_inside_box)
    {
      cells_inside_box.push_back(cell_number);

      //cell_weights.push_back(10);
    } 
    //else
     // cell_weights.push_back(0);
     cell_weights.push_back(distance_to_singularity<dim>(cell->center())* 100);

}

   // if (cell->is_locally_owned()) //weil mpi danach nochmal sortiert wird das hier auskommentieren
    {
  
    double cell_diameter = cell->diameter(); 
    
    if (cell_diameter > max_diameter) {
      max_diameter = cell_diameter;
    }
   
    for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell;
         face_no++) {
      Point<dim> p = cell->face(face_no)->center();
      if (cell->face(face_no)->at_boundary()) {
       double error = 0.000001;
        if((std::abs(p[0] - 0)< error || std::abs(p[0] - 2 * half_length)<error) && geo_conf == GeometryConfiguration::ThreeD_OneD && (constructed_solution >= 2)) {//
        cell->face(face_no)->set_boundary_id(Neumann);
         //pcout<<"Neumann"<<std::endl;
        }
        else{
           cell->face(face_no)->set_boundary_id(Dirichlet);
       //pcout<<"Dirichlet"<<std::endl;
        }
       
      }
       
      

    }
    }
    cell_number++;
  }
  max_diameter = GridTools::maximal_cell_diameter(triangulation);
  pcout<<" is_cell_inside_box "<<cells_inside_box.size()<<std::endl;//TODOD einfach zu viele Einträge um in Sparsity matrix reinzumachen
if(is_repartioned)
{
  pcout<<"Cell connection graph "<<std::endl;
  //connectivity.reinit(triangulation.n_global_active_cells(),triangulation.n_global_active_cells());
  //pcout<<"trow"<<std::endl;
  unsigned int row = 0;
   typename Triangulation<dim>::active_cell_iterator
      cell = triangulation.begin_active(),
      endc = triangulation.end();
  for (; cell != endc; ++cell)
  {
    IndexSet index_set;
    index_set.set_size(triangulation.n_global_active_cells());
    index_set.add_index(row);
    DynamicSparsityPattern dsp = connectivity_cells.get_view(index_set);
    connectivity_cells.clear_row(row);
    if(is_cell_inside_box[row])
      {

      for (auto col = dsp.begin(); col != dsp.end(); ++col) 
     {
        const unsigned int column = col->column();
       if(is_cell_inside_box[column] == true)
          connectivity_cells.add(row, column);
     }

      //if(cell-center()[0] )
      //connectivity_cells.add(row, column);
      // 
      }
     // connectivity_cells.add_entries(row, cells_inside_box.begin(), cells_inside_box.end());
        else
     {
      // typename DynamicSparsityPattern::iterator it(&connectivity_cells,row,0 );
      // std::cout<<"index"<<it->index()<<std::endl;
    // typename DynamicSparsityPattern::iterator it_end(&connectivity_cells,row,connectivity_cells.row_length(row) );
     //for(; it != it_end; it++)
      //for (auto col = connectivity_cells.begin(row); col != connectivity_cells.end(row); ++col) 
   
   
   
    for (auto col = dsp.begin(); col != dsp.end(); ++col) 
     {
        const unsigned int column = col->column();
       if(is_cell_inside_box[column] == false)
          connectivity_cells.add(row, column);
     }
    
    
    
     }
    // std::cout<<"connectivity_cells.row_length(row) "<<connectivity_cells.row_length(row)<<std::endl;
  // pcout<<row<<" "<<std::endl;
  row++;
  }
 // pcout<<std::endl;
 //connectivity.symmetrize();
 // memory_consumption("copy graph");
 pcout<<rank_mpi<<" copy graph"<<std::endl;
  cell_connection_graph.copy_from(connectivity_cells);
  //connectivity.reinit(1,1);
   connectivity_cells.reinit(1,1);
  //  memory_consumption("connectivity_cells");
  pcout<<"los "<< dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)<<std::endl;
 //GridTools::partition_triangulation(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),cell_connection_graph,triangulation, SparsityTools::Partitioner::zoltan );
 GridTools::partition_triangulation(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD), cell_connection_graph,triangulation, SparsityTools::Partitioner::zoltan );
 //GridTools::partition_triangulation_zorder(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),triangulation);
 }  



				       







  pcout<<"max_diameter "<<max_diameter<<" radius "<<radius<<std::endl;
  if (radius > max_diameter && !lumpedAverage) {
    pcout << "!!!!!!!!!!!!!! MAX DIAMETER > RADIUS !!!!!!!!!!!!!!!!"
              << max_diameter << radius << std::endl;
    //throw std::invalid_argument("MAX DIAMETER > RADIUS");
  }
//---------------omega-------------------------
    if(dim == 2)
    GridGenerator::hyper_cube(triangulation_omega, -half_length ,  half_length);
    if(dim == 3)
    GridGenerator::hyper_cube(triangulation_omega,0 ,  2*half_length);

  triangulation_omega.refine_global(refine_omega);//level_max
  nof_cells_omega = triangulation_omega.n_global_active_cells();
 typename DoFHandler<dim_omega>::active_cell_iterator
        cell_omega = dof_handler_omega.begin_active(),
        endc_omega = dof_handler_omega.end();
  max_diameter_omega = 0.0;
  for (; cell_omega != endc_omega; ++cell_omega) {

    if (cell_omega->is_locally_owned())
    {

      double cell_diameter = cell_omega->diameter(); 
          
      if (cell_diameter > max_diameter_omega) {
        max_diameter_omega = cell_diameter;
      }
    
    for (unsigned int face_no = 0;
         face_no < GeometryInfo<dim_omega>::faces_per_cell; face_no++) {
      if (cell_omega->face(face_no)->at_boundary())
      {
        if(constructed_solution != 2)
        cell_omega->face(face_no)->set_boundary_id(Dirichlet);
        else
        cell_omega->face(face_no)->set_boundary_id(NotDefined);

        //cell_omega->face(face_no)->set_boundary_id(Dirichlet);
        //cell_omega->face(face_no)->set_boundary_id(Neumann);
      }

    }
    }
  }
  max_diameter_omega = GridTools::maximal_cell_diameter(triangulation_omega);
/*
GridOut grid_out_omega;
std::ofstream out_omega("grid_omega.vtk"); 
grid_out_omega.write_vtk(triangulation_omega, out_omega);
*/

//handle omega
#if 1// COUPLED
marked_vertices.resize(triangulation.n_vertices());

for (unsigned int i = 0; i < triangulation.n_vertices(); i++)
{
    if (bbox.point_inside(vertices[i]))
    {
      marked_vertices[i] = true;
     // pcout<< "marked_vertices[i] "<< marked_vertices[i]<<std::endl;
    }
    else
    {
      marked_vertices[i] = false;
    }
     //marked_vertices[i] = true;
}
//pcout<< "memory consump marked vertices "<<MemoryConsumption::memory_consumption(marked_vertices)/(1024.0 * 1024.0 * 1024.0) // Convert to MB
	 //             << " GB" << std::endl;
#endif
//malloc_trim(0);  // Force memory release

pcout<<"ende make_grid"<<std::endl;
}

template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::make_dofs() {
  TimerOutput::Scope t(computing_timer, "setup");

  IndexSet locally_owned_dofs_Omega;
  IndexSet locally_relevant_dofs_Omega;

  IndexSet locally_owned_dofs_omega_local;
  IndexSet locally_relevant_dofs_omega_local;


  dof_handler_Omega.distribute_dofs(fe_Omega);
  const unsigned int dofs_per_cell = fe_Omega.dofs_per_cell;
  pcout << "dofs_per_cell " << dofs_per_cell << std::endl;
 

  dof_handler_omega.distribute_dofs(fe_omega);
  const unsigned int dofs_per_cell_omega = fe_omega.dofs_per_cell;
  pcout << "dofs_per_cell_omega " << dofs_per_cell_omega << std::endl;

#if !COUPLED
 // DoFRenumbering::component_wise(dof_handler_Omega); //uncomment for unput result
 // DoFRenumbering::component_wise(dof_handler_omega); //TODO nochmal kontrollieren
#endif

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
        << n_vector_field_Omega << " + " << n_potential_Omega << ")"<<std::endl
        <<" triangulation.n_vertices() "<<triangulation.n_vertices()<< std::endl;
  unsigned int locally_owned_cells = triangulation.n_locally_owned_active_cells();
  std::cout << rank_mpi<<" Number of locally owned active cells: " << locally_owned_cells <<" Number of locally owned DoF: " << dof_handler_Omega.n_locally_owned_dofs()<<std::endl;
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
  
  DoFTools::extract_locally_relevant_dofs(dof_handler_Omega, locally_relevant_dofs_Omega);
 /* std::cout<<"Memory locally_owned_dofs_Omega "<< locally_owned_dofs_Omega.memory_consumption()/ (1024.0 * 1024.0 ) // Convert to MB
	          << " MB" << std::endl;
  std::cout<<"Memory locally_relevant_dofs_Omega "<< locally_relevant_dofs_Omega.memory_consumption()/ (1024.0 * 1024.0 ) // Convert to MB
	            << " MB" << std::endl;*/
  locally_owned_dofs_omega_local = dof_handler_omega.locally_owned_dofs();
  // if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0 )
   //locally_owned_dofs_omega_local.clear();
   
  DoFTools::extract_locally_relevant_dofs(dof_handler_omega, locally_relevant_dofs_omega_local);
  
    /*
  locally_owned_dofs_omega_global.set_size(locally_owned_dofs_omega_local.size());
  locally_owned_dofs_omega_global.add_indices(locally_owned_dofs_omega_local,  dof_handler_Omega.n_dofs());
*/
    /*
  locally_relevant_dofs_omega_global.set_size(locally_relevant_dofs_omega_local.size());
  locally_relevant_dofs_omega_global.add_indices(locally_relevant_dofs_omega_local,  dof_handler_Omega.n_dofs());
*/

  std::vector<IndexSet> locally_owned_dofs_block;
  std::vector<IndexSet> locally_relevant_dofs_block;

  locally_owned_dofs_block.push_back(locally_owned_dofs_Omega);
  locally_owned_dofs_block.push_back(locally_owned_dofs_omega_local);

  locally_relevant_dofs_block.push_back(locally_relevant_dofs_Omega);
  locally_relevant_dofs_block.push_back(locally_relevant_dofs_omega_local);


/*
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
*/

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
 /* 
std::cout <<"----------------------------------- "<<  std::endl;
 std::cout <<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" locally_relevant_dofs_Omega "<<  locally_relevant_dofs_Omega.size()<<" elem "<<locally_relevant_dofs_Omega.n_elements()<< std::endl;
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

{
pcout<<"BlockSparsityPattern"<<std::endl;
//TrilinosWrappers::BlockSparsityPattern sp_block=  TrilinosWrappers::BlockSparsityPattern(2,2);
TrilinosWrappers::BlockSparsityPattern sp_block=  TrilinosWrappers::BlockSparsityPattern(locally_owned_dofs_block,locally_owned_dofs_block,locally_relevant_dofs_block, MPI_COMM_WORLD );
//BlockSparsityPattern sp_block=  BlockSparsityPattern(locally_owned_dofs_block,locally_owned_dofs_block,locally_relevant_dofs_block, MPI_COMM_WORLD );
  sp_block.block(0, 0).reinit(locally_owned_dofs_Omega,locally_owned_dofs_Omega,locally_relevant_dofs_Omega,MPI_COMM_WORLD);
 // std::cout<<"sparsity memory block(0, 0) "<<sp_block.memory_consumption()/ (1024.0 * 1024.0 * 1024.0) // Convert to MB
//	              << " GB" << std::endl;
    // Block 1,1: Sparsity for the extra DoFs (global DoFs)
  sp_block.block(1, 1).reinit(locally_owned_dofs_omega_local,locally_owned_dofs_omega_local,locally_relevant_dofs_omega_local,MPI_COMM_WORLD);
 // std::cout<<"sparsity memory block(1, 1) "<<sp_block.memory_consumption()/ (1024.0 * 1024.0 * 1024.0) // Convert to MB
//	              << " GB" << std::endl;
  // Block 0,1 and 1,0: Sparsity for connections between mesh-based and extra DoFs
  sp_block.block(0, 1).reinit(locally_owned_dofs_Omega,locally_owned_dofs_omega_local,locally_relevant_dofs_Omega,MPI_COMM_WORLD);  // Block for coupling DoFs
 // std::cout<<"sparsity memory block(0, 1) "<<sp_block.memory_consumption()/ (1024.0 * 1024.0 * 1024.0) // Convert to MB
	 //             << " GB" << std::endl;
  sp_block.block(1, 0).reinit(locally_owned_dofs_omega_local,locally_owned_dofs_Omega,locally_relevant_dofs_omega_local,MPI_COMM_WORLD);
 // std::cout<<"sparsity memory block(1, 0)"<<sp_block.memory_consumption()/ (1024.0 * 1024.0 * 1024.0) // Convert to MB
//	              << " GB" << std::endl;
 
  Table< 2, DoFTools::Coupling >	cell_integrals_mask_Omega(dim +1, dim +1);
  for (unsigned int c = 0; c < dim + 1; ++c)
  {
    for (unsigned int d = 0; d < dim + 1; ++d)
    {
        if (c == dim || d == dim || c == d) //coupling between scalar values with its test functions (for dimension coupling)
           cell_integrals_mask_Omega[c][d] = DoFTools::always; //coupling between each entry of vector values and with pressure
        else
         cell_integrals_mask_Omega[c][d] = DoFTools::none;
        //std::cout<<cell_integrals_mask_Omega[c][d]<< " ";
    }
    //std::cout<<std::endl;
  }
  Table< 2, DoFTools::Coupling > 	face_integrals_mask_Omega(dim +1, dim +1);
    for (unsigned int c = 0; c < dim + 1; ++c)
  {
    for (unsigned int d = 0; d < dim + 1; ++d)
    {
        if (c == dim || d == dim ) //coupling between scalar values with its test functions (for dimension coupling) and with vector values
           face_integrals_mask_Omega[c][d] = DoFTools::always; 
        else
         face_integrals_mask_Omega[c][d] = DoFTools::none;
        //std::cout<<face_integrals_mask_Omega[c][d]<< " ";
    }
   // std::cout<<std::endl;
  }
  DoFTools::make_flux_sparsity_pattern(dof_handler_Omega, sp_block.block(0,0),cell_integrals_mask_Omega, face_integrals_mask_Omega,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
 // std::cout<<"sparsity memory flx block(0, 0)"<<sp_block.memory_consumption()/ (1024.0 * 1024.0 * 1024.0) // Convert to MB
	//              << " GB" << std::endl;
  DoFTools::make_flux_sparsity_pattern(dof_handler_omega, sp_block.block(1,1),constraints,false,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) );
 // std::cout <<"sparsity memory flx block(1, 1)"<<sp_block.memory_consumption()/ (1024.0 * 1024.0 * 1024.0) // Convert to MB
	//              << " GB" << std::endl;
  sp_block.collect_sizes();
  //malloc_trim(0);  // Force memory release

  int error_flag = 0, global_error_flag = 0;
  AVERAGE = radius != 0 && !lumpedAverage && (COUPLED || VESSEL);// &&(constructed_solution == 3 || constructed_solution == 2) && geo_conf == GeometryConfiguration::ThreeD_OneD;//||constructed_solution == 2
pcout << "AVERAGE (use circel) " << AVERAGE << " radius "<<radius << " lumpedAverage "<<lumpedAverage<<std::endl;
// weight
if (AVERAGE) {
  unsigned int n = std::ceil(radius/(pow(2,minimal_cell_diameter_2D/std::sqrt(2)))) + 1;
 // std::cout<<"n "<<n <<" "<< minimal_cell_diameter_2D<<" "<<minimal_cell_diameter_2D/std::sqrt(2)<<std::endl;
  nof_quad_points = 25 * n_refine;// std::pow(2,n);
} else {
  nof_quad_points = 1;
}
pcout<<"nof_quad_points "<<nof_quad_points<<std::endl;
std::cout <<"std::sqrt(2)/2.0 "<<std::sqrt(2)/2.0<<std::endl;
#if COUPLED
if(geo_conf != GeometryConfiguration::TwoD_ZeroD)  {
    // coupling
  pcout<<"Sparsity Coupling "<<std::endl;
 typename DoFHandler<dim>::active_cell_iterator
        cell_start = dof_handler_Omega.begin_active();

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
   

    typename DoFHandler<dim_omega>::active_cell_iterator
        cell_omega = dof_handler_omega.begin_active(),
        endc_omega = dof_handler_omega.end();

    for (; cell_omega != endc_omega; ++cell_omega) 
    {
      //if (cell_omega->is_locally_owned())
      {
      fe_values_omega.reinit(cell_omega);
      cell_omega->get_dof_indices(local_dof_indices_omega);
      dof_omega_local_2_global(dof_handler_omega, local_dof_indices_omega);

      //std::vector<Point<dim_omega>> quadrature_points_omega = {Point<dim_omega>(std::sqrt(2)/2.0 + arrr)};
      std::vector<Point<dim_omega>> quadrature_points_omega = fe_values_omega.get_quadrature_points();

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
     //   pcout<<"quadrature_point_test "<<quadrature_point_test<<std::endl;

//pcout <<"stat "<<std::endl;
   auto start = std::chrono::high_resolution_clock::now();  //Start time
    auto cell_test_first = GridTools::find_active_cell_around_point(
          cache, quadrature_point_test, cell_start, marked_vertices);
       //    pcout<<"###+++# " <<cell_test_first.first<<" "<<cell_test_first.second<<std::endl;
    auto end = std::chrono::high_resolution_clock::now();    // End time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

//pcout << "Time taken to execute find_all_active_cells_around_point: " << duration << " ms" << std::endl;      
          
	#if FASTER
   auto cell_test_array = find_all_active_cells_around_point<dim, dim>(
                       mapping, triangulation, quadrature_point_test,1e-10 ,cell_test_first, &cache.get_vertex_to_cell_map());//, cache.get_vertex_to_cell_map()
   #else
      auto cell_test_array = GridTools::find_all_active_cells_around_point(
                       mapping, triangulation, quadrature_point_test,1e-10 ,cell_test_first);//, cache.get_vertex_to_cell_map()
   #endif
 //pcout<<"cell_test_array.size() "<<cell_test_array.size()<<std::endl;
        for (auto cellpair : cell_test_array)
        {

          auto cell_test_tri = cellpair.first;
         typename DoFHandler<dim>::active_cell_iterator
        cell_test = dof_handler_Omega.begin_active(cell_test_tri->level()),
         endc_test = dof_handler_Omega.end();

    for (; cell_test != endc_test; ++cell_test) {
      if(cell_test->level() == cell_test_tri->level() && cell_test->index() == cell_test_tri->index())
      {
        //std::cout<<"break"<<std::endl;
        break;
      }
    }
       // std::advance(cell_test, cell_test_tri->index());
        cell_start =cell_test;  
        if(cell_test_tri->index() != cell_test->index())
        pcout<<"cellcomp " <<cell_test_tri->index()<<" " <<cell_test_tri<<" : "<<cell_test<<std::endl;

#if USE_MPI_ASSEMBLE
         if (cell_test != dof_handler_Omega.end())
            if (cell_test->is_locally_owned())
#endif
            {

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

              for (unsigned int q_avag = 0; q_avag < nof_quad_points;
                   q_avag++) {
                // Quadrature weights and points
                quadrature_point_trial = quadrature_points_circle[q_avag];

#if TEST
//pcout<<"asdf " <<quadrature_point_trial<<std::endl;
    auto cell_trial_first = GridTools::find_active_cell_around_point(
          cache, quadrature_point_trial, cell_start, marked_vertices);
  //         pcout<<"###### " <<cell_trial_first.first<<" "<<cell_trial_first.second<<std::endl;
   #if FASTER
   auto cell_trial_array = find_all_active_cells_around_point<dim, dim>(
                       mapping, triangulation, quadrature_point_trial,1e-10 ,cell_trial_first, &cache.get_vertex_to_cell_map());//, cache.get_vertex_to_cell_map()*/ //correct
   #else
   auto cell_trial_array = GridTools::find_all_active_cells_around_point(
                       mapping, triangulation, quadrature_point_trial,1e-10 ,cell_trial_first);//, cache.get_vertex_to_cell_map()
   #endif
 //pcout<<"----" <<std::endl;
    for (auto cellpair_trial : cell_trial_array)
#else
              auto cell_trial = GridTools::find_active_cell_around_point(
                  dof_handler_Omega, quadrature_point_trial);
#endif

                {
#if TEST
                  auto cell_trial_tri = cellpair_trial.first;

    
                  typename DoFHandler<dim>::active_cell_iterator
                  cell_trial = dof_handler_Omega.begin_active(cell_trial_tri->level()),
                  endc_trial = dof_handler_Omega.end();

              for (; cell_trial != endc_trial; ++cell_trial) {
                if(cell_trial->level() == cell_trial_tri->level() && cell_trial->index() == cell_trial_tri->index())
                {
                  //std::cout<<"break"<<std::endl;
                  break;
                }
              }
                if(cell_trial_tri->index() != cell_trial->index())
              pcout<<"cellcomp trial " <<cell_trial_tri<<" : "<<cell_trial<<std::endl;
#endif

                  if (cell_trial != dof_handler_Omega.end()) {
                    if (cell_trial->is_locally_owned() &&
                        cell_test->is_locally_owned()) {

                  
                      cell_trial->get_dof_indices(local_dof_indices_trial);

                      for (unsigned int i = 0;
                           i < local_dof_indices_test.size(); i++) {
                        for (unsigned int j = 0;
                             j < local_dof_indices_trial.size(); j++) {
                          sp_block.add(local_dof_indices_test[i],
                                  local_dof_indices_trial[j]);
                        }
                      }

                      for (unsigned int i = 0;
                           i < local_dof_indices_omega.size(); i++) {
                        for (unsigned int j = 0;
                             j < local_dof_indices_trial.size(); j++) {                        
                        sp_block.add(local_dof_indices_omega[i],
                                  local_dof_indices_trial[j]);
                        }
                      }

                      for (unsigned int i = 0;
                           i < local_dof_indices_test.size(); i++) {
                        for (unsigned int j = 0;
                             j < local_dof_indices_omega.size(); j++) {                         
                            sp_block.add(local_dof_indices_test[i],
                                  local_dof_indices_omega[j]);
                        }
                      }

                      for (unsigned int i = 0;
                           i < local_dof_indices_omega.size(); i++) {
                        for (unsigned int j = 0;
                             j < local_dof_indices_omega.size(); j++) {                       
                            sp_block.add(local_dof_indices_omega[i],
                                  local_dof_indices_omega[j]);                        
                        }
                      }
                   } 
                   else
                   {
                  std::cout<<"düdüm1"<<std::endl;
                  error_flag = 1;
                 //throw std::runtime_error("cell coupling error");
                // MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                  }
                  }
               // else
                 // std::cout<<"düdüm2"<<std::endl;
                }
              }
            }
         //  else
          // std::cout<<"düdüm3"<<std::endl;
        }
        // std::cout<<std::endl;
      }
    }
    }
  }

if(geo_conf == GeometryConfiguration::TwoD_ZeroD)  {
  
  pcout << "2D/0D" << std::endl;
   
  bool insideCell_test = true;
  bool insideCell_trial = true;
  
  Point<dim> quadrature_point_trial;
  Point<dim> quadrature_point_coupling(y_l, z_l);
  Point<dim> normal_vector(0);

  QGauss<dim> quadrature_formula(fe_Omega.degree + 1);
  FEValues<dim> fe_values(fe_Omega, quadrature_formula, update_flags);
  const Mapping<dim> &mapping = fe_values.get_mapping();


  std::vector<Point<dim>> quadrature_points_circle;
  quadrature_points_circle = equidistant_points_on_circle<dim>(
      quadrature_point_coupling, radius, normal_vector,
      nof_quad_points);
  Point<dim> quadrature_point_test = quadrature_point_coupling;


  std::vector<types::global_dof_index> local_dof_indices_test(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices_trial(dofs_per_cell);
  // test function
  std::vector<double> my_quadrature_weights = {1};
  unsigned int n_te;
#if TEST
  auto cell_test_array = GridTools::find_all_active_cells_around_point(
      mapping, dof_handler_Omega, quadrature_point_test, 1e-10, marked_vertices);
  n_te = cell_test_array.size();
  //std::cout << "cell_test_array " << cell_test_array.size() << std::endl;

  for (auto cellpair : cell_test_array)
#else
  auto cell_test = GridTools::find_active_cell_around_point(
      dof_handler_Omega, quadrature_point_test);
  n_te = 1;
#endif

  {
#if TEST
    auto cell_test = cellpair.first;
    //pcout<<cell_test<<std::endl;
#endif


    if (cell_test != dof_handler_Omega.end())
      if (cell_test->is_locally_owned())
      {
        cell_test->get_dof_indices(local_dof_indices_test);

    for (unsigned int q_avag = 0; q_avag < nof_quad_points;
                 q_avag++) {
     // Quadrature weights and points
      quadrature_point_trial = quadrature_points_circle[q_avag];

#if TEST
  auto cell_trial_array = GridTools::find_all_active_cells_around_point(
      mapping, dof_handler_Omega, quadrature_point_trial, 1e-10, marked_vertices);
  n_te = cell_test_array.size();
  //std::cout << "cell_trial_array " << cell_trial_array.size() << std::endl;

  for (auto cellpair : cell_trial_array)
#else
  auto cell_trial = GridTools::find_active_cell_around_point(
      dof_handler_Omega, quadrature_point_trial);
  n_te = 1;
#endif
 {
#if TEST
    auto cell_trial = cellpair.first;
  //  pcout<<"cell_trial " <<cell_trial<<std::endl;
#endif


    if (cell_trial != dof_handler_Omega.end())
         if (cell_trial->is_locally_owned() &&
               cell_test->is_locally_owned()) 
      {
        
        cell_trial->get_dof_indices(local_dof_indices_trial);

          // V_U_matrix_coupling
          for (unsigned int i = 0; i < dofs_per_cell; i++) {
            for (unsigned int j = 0; j < dofs_per_cell; j++) {
              sp_block.add(local_dof_indices_test[i],
                local_dof_indices_trial[j]);
             
            }
          }

        }//if cell_trial
        else 
        {
          std::cout<<"düdüm1 - assem"<<std::endl;
          error_flag = 1;
        }
      
    }//cell_trial



      

      }//nof_quad_points
 } //if cell_test
//#endif
      }// for cell_test_array
  }


  #endif
 MPI_Allreduce(&error_flag, &global_error_flag, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
if (global_error_flag) {
       // printf("Process exiting function due to global error.\n");
        throw std::runtime_error("cell coupling error");
  }
  pcout<<"start to compress"<<std::endl;


  sp_block.compress();

   pcout<<"Sparsity "  <<sp_block.n_rows()<<" "<<sp_block.n_cols()<<" n_nonzero_elements " <<sp_block.n_nonzero_elements()<<std::endl;
#if MEMORY_CONSUMPTION
   std::cout<<"mpi_rank "<<rank_mpi<<" sparsity memory "<<sp_block.memory_consumption()/(1024*1024)<<" MB"<<std::endl;
   std::cout<<"mpi_rank "<<rank_mpi<<" dof_handler_Omega "<<dof_handler_Omega.memory_consumption()/(1024*1024)<<" MB"<<std::endl;
#endif
   //pcout<<"start reinit"<<std::endl;
  // memory_consumption("before system_matrix reinit");
  system_matrix.reinit(sp_block);
#if MEMORY_CONSUMPTION
  std::cout<<"mpi_rank "<<rank_mpi<<" memory system_matrix "<<system_matrix.memory_consumption()/(1024*1024)<<" MB"<<std::endl;
#endif
  //memory_consumption("after system_matrix reinit");
}
  //pcout<<"system_matrix.reinit"<<std::endl;
  solution.reinit(locally_relevant_dofs_block,  MPI_COMM_WORLD);
 // memory_consumption("after solution.reinit");
   //pcout<<"solution.reinit"<<std::endl;
  system_rhs.reinit(locally_owned_dofs_block, locally_relevant_dofs_block,  MPI_COMM_WORLD, true);
 // memory_consumption("after system_rhs.reinit");
 //  pcout<<"system_rhs.reinit"<<std::endl;

 //  std::cout<<rank_mpi<<" memory system_matrix "<<system_matrix.memory_consumption()/ (1024.0 * 1024.0 * 1024.0)<<" memory system_rhs "<<system_rhs.memory_consumption()/ (1024.0 * 1024.0 * 1024.0)<<std::endl;
  pcout<<"Ende setup dof"<<std::endl;


//std::cout<<"malloc_trim "<<malloc_trim(0)<<std::endl;

}

template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::assemble_system() {
  TimerOutput::Scope t(computing_timer, "assembly");
  //malloc_trim(0);  // Force memory release




  pcout << "assemble_system" << std::endl;
 typename DoFHandler<dim>::active_cell_iterator
        cell_start = dof_handler_Omega.begin_active();
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

  FESubfaceValues<dim> fe_subface_values(fe_Omega, face_quadrature_formula,
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
    for (; cell != endc; ++cell) {

#if USE_MPI_ASSEMBLE
      if (cell->is_locally_owned())
#endif
      {

        local_matrix = 0;
        local_vector = 0;

        fe_values.reinit(cell);
        assemble_cell_terms(fe_values, local_matrix, local_vector,
                            K_inverse_function, rhs_function, VectorField,
                            Potential, false);

        cell->get_dof_indices(local_dof_indices);

        for (unsigned int face_no = 0;
             face_no < GeometryInfo<dim>::faces_per_cell; face_no++) {
          typename DoFHandler<dim>::face_iterator face = cell->face(face_no);

          if (face->at_boundary()) {
            fe_face_values.reinit(cell, face_no);

            if (face->boundary_id() == Dirichlet) {
             
              double h = cell->diameter();
              assemble_Dirichlet_boundary_terms(
                  fe_face_values, local_matrix, local_vector, h,
                  Dirichlet_bc_function, VectorField, Potential);
            } else if (face->boundary_id() == Neumann) {
            
              assemble_Neumann_boundary_terms(fe_face_values, local_matrix,
                                            local_vector,
                                             Neumann_bc_function, VectorField, Potential);
            } else
            {
                 //std::cout<<rank_mpi<< " c " <<cell->index()<<" f "<<face->index()<<" Omega, boundary condition not implemented "<<std::endl;
                //Assert(false, ExcNotImplemented());
            }
              
          } else 
          {
               
                  //
                  Assert(cell->neighbor(face_no).state() ==
                         IteratorState::valid,
                         ExcInternalError());

                  typename DoFHandler<dim>::cell_iterator neighbor =
                    cell->neighbor(face_no);

               
                  if (face->has_children())
                    {
                     // std::cout<<"has childeren"<<std::endl;
    
                      const unsigned int neighbor_face_no =
                        cell->neighbor_of_neighbor(face_no);

                    
                      for (unsigned int subface_no=0;
                           subface_no < face->n_children();
                           ++subface_no)
                        {
                         
                          typename DoFHandler<dim>::cell_iterator neighbor_child =
                            cell->neighbor_child_on_subface(face_no,
                                                            subface_no);

                          Assert(!neighbor_child->has_children(),
                                 ExcInternalError());

                       
                          vi_ui_matrix = 0;
                          vi_ue_matrix = 0;
                          ve_ui_matrix = 0;
                          ve_ue_matrix = 0;

                          fe_subface_values.reinit(cell, face_no, subface_no);
                          fe_neighbor_face_values.reinit(neighbor_child,
                                                         neighbor_face_no);

                        
                          double h = std::min(cell->diameter(),
                                              neighbor_child->diameter());

                        
                        assemble_flux_terms(fe_subface_values,
                                      fe_neighbor_face_values,
                          vi_ui_matrix, vi_ue_matrix, ve_ui_matrix,
                          ve_ue_matrix, h, VectorField, Potential);

                          
                          neighbor_child->get_dof_indices(local_neighbor_dof_indices);

                        
                          distribute_local_flux_to_global(
                            vi_ui_matrix,
                            vi_ue_matrix,
                            ve_ui_matrix,
                            ve_ue_matrix,
                            local_dof_indices,
                            local_neighbor_dof_indices);
                        }
                    }
                  else
                    {
                     
                      if (neighbor->level() == cell->level() &&
                          cell->id() < neighbor->id())
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

                              assemble_flux_terms(fe_face_values, fe_neighbor_face_values,
                        vi_ui_matrix, vi_ue_matrix, ve_ui_matrix,
                        ve_ue_matrix, h, VectorField, Potential);

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
  //system_matrix.compress(VectorOperation::add);
  //system_rhs.compress(VectorOperation::add);

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


// if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )

  #if 1//COUPLED
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

if(constructed_solution == 2)
{
 assemble_cell_terms(fe_values_omega, local_matrix_omega,
                          local_vector_omega, k_inverse_function,
                          rhs_function_omega, VectorField_omega,
                          Potential_omega,true );
}
else
{
   assemble_cell_terms(fe_values_omega, local_matrix_omega,
                          local_vector_omega, k_inverse_function,
                          rhs_function_omega, VectorField_omega,
                          Potential_omega,false );
}


      cell_omega->get_dof_indices(local_dof_indices_omega);

      dof_omega_local_2_global(dof_handler_omega, local_dof_indices_omega);


      for (unsigned int face_no_omega = 0;
           face_no_omega < GeometryInfo<dim_omega>::faces_per_cell;
           face_no_omega++) {
            
        
        typename DoFHandler<dim_omega>::face_iterator face_omega =
            cell_omega->face(face_no_omega);

        if (face_omega->at_boundary()) {
          fe_face_values_omega.reinit(cell_omega, face_no_omega);

          if (face_omega->boundary_id() == Dirichlet) {
            double h = 1;//cell_omega->diameter();
            //std::cout<<rank_mpi << " c " <<cell_omega->index()<<" f "<<face_omega->index()<<" omega Dirichlet "<<std::endl;
            assemble_Dirichlet_boundary_terms(
                fe_face_values_omega, local_matrix_omega, local_vector_omega, h,
                Dirichlet_bc_function_omega, VectorField_omega,
                Potential_omega);
          // std::cout<<Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)<<" cell_id "<<cell_id_omega<<" face_no_omega "<<face_no_omega<< " Dirichlet"<<std::endl;
            
          }
          else if (face_omega->boundary_id() == Neumann)
            {
             // std::cout<<rank_mpi << " c " <<cell_omega->index()<<" f "<<face_omega->index()<<" omega Neumann "<<std::endl;
              assemble_Neumann_boundary_terms(fe_face_values_omega,
                                              local_matrix_omega,
                                          local_vector_omega, Neumann_bc_function_omega, VectorField_omega, Potential_omega);
            }
          else
           {
               // std::cout<<rank_mpi << " c " <<cell_omega->index()<<" f "<<face_omega->index()<<" omega, boundary condition not implemented "<<std::endl;
                if(constructed_solution != 2)
                Assert(false, ExcNotImplemented());
           } 
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

            double h = 1;//std::min(cell_omega->diameter(), neighbor_omega->diameter());

           assemble_flux_terms(
                fe_face_values_omega, fe_neighbor_face_values_omega,
                vi_ui_matrix_omega, vi_ue_matrix_omega, ve_ui_matrix_omega,
                ve_ue_matrix_omega, h, VectorField_omega, Potential_omega);

            neighbor_omega->get_dof_indices(local_neighbor_dof_indices_omega);
           dof_omega_local_2_global(dof_handler_omega,
                              local_neighbor_dof_indices_omega);
            if(constructed_solution != 2)
            {
            distribute_local_flux_to_global(
                  vi_ui_matrix_omega, vi_ue_matrix_omega, ve_ui_matrix_omega,
                  ve_ue_matrix_omega, local_dof_indices_omega,
                  local_neighbor_dof_indices_omega);
            }
  
          }
        }
        
      }//face iterate

      constraints.distribute_local_to_global(
          local_matrix_omega, local_dof_indices_omega, system_matrix);

     constraints.distribute_local_to_global(
          local_vector_omega, local_dof_indices_omega, system_rhs);
    
   }//end locally ownedS
    
    }
    
  }
    pcout << "ende omega loop" << std::endl;
  #endif
#if 0// USE_MPI_ASSEMBLE
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
#endif

#if 1
{
#if 1// USE_MPI_ASSEMBLE
// if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
#endif
  TimerOutput::Scope t(computing_timer, "assembly - coupling");
  // coupling
  pcout << "assemble Coupling" << std::endl;
  
  if (geo_conf == GeometryConfiguration::TwoD_ZeroD) {
    pcout << "2D/0D" << std::endl;
#if COUPLED 
    double beta = 2 * numbers::PI * D * radius;
#else
    double beta = (2 * numbers::PI)/(2 * numbers::PI + std::log( radius));
#endif
    FullMatrix<double> V_U_matrix_coupling(dofs_per_cell, dofs_per_cell);
    bool insideCell_test = true;
    bool insideCell_trial = true;
    
    Point<dim> quadrature_point_trial;
    Point<dim> quadrature_point_coupling(y_l, z_l);
    Point<dim> normal_vector(0);
  

    std::vector<Point<dim>> quadrature_points_circle;
    quadrature_points_circle = equidistant_points_on_circle<dim>(
        quadrature_point_coupling, radius, normal_vector,
        nof_quad_points);
    Point<dim> quadrature_point_test = quadrature_point_coupling;
    //std::cout<<"quadrature_point_test "<<quadrature_point_test<<std::endl;


    std::vector<types::global_dof_index> local_dof_indices_test(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_trial(dofs_per_cell);
    // test function
    std::vector<double> my_quadrature_weights = {1};
    unsigned int n_te;
#if TEST
    auto cell_test_array = GridTools::find_all_active_cells_around_point(
        mapping, dof_handler_Omega, quadrature_point_test, 1e-10, marked_vertices);
    n_te = cell_test_array.size();
   // std::cout << "cell_test_array " << cell_test_array.size() << std::endl;

    for (auto cellpair : cell_test_array)
#else
    auto cell_test = GridTools::find_active_cell_around_point(
        dof_handler_Omega, quadrature_point_test);
    n_te = 1;
#endif

    {
#if TEST
      auto cell_test = cellpair.first;
    //  pcout<<"cell_test "<<cell_test<<std::endl;
#endif


      if (cell_test != dof_handler_Omega.end())
        if (cell_test->is_locally_owned())
        {
          std::vector<unsigned int> face_no_test;
          cell_test->get_dof_indices(local_dof_indices_test);

          unsigned int n_ftest = 0;
          for (unsigned int face_no = 0;
               face_no < GeometryInfo<dim>::faces_per_cell; face_no++) {
            typename DoFHandler<dim>::face_iterator face_test =
                cell_test->face(face_no);
            auto bounding_box = face_test->bounding_box();
           if(bounding_box.point_inside(quadrature_point_test) == true)
           {
           n_ftest += 1;
            face_no_test.push_back(face_no);
          }
          }
          if (n_ftest == 0) {
            insideCell_test = true;
            n_ftest = 1;
            face_no_test.push_back(0);
          } else {
            insideCell_test = false;
              
          }

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
//#if !COUPLED
          //-------------face -----------------
          if(!insideCell_test)
          {
            
          for (unsigned int face_no = 0;
               face_no < GeometryInfo<dim>::faces_per_cell; face_no++) {
            typename DoFHandler<dim>::face_iterator face_test =
                cell_test->face(face_no);
            //std::cout<<"c "<<cell_test<< " f "<<face_no<<std::endl;
            Point<dim - 1> quadrature_point_test_mapped_face =
                mapping.project_real_point_to_unit_point_on_face(
                    cell_test, face_no, quadrature_point_test);
            for(unsigned int tf = 0; tf < dim -1; tf++)
              quadrature_point_test_mapped_face[tf] = std::max(0.0,quadrature_point_test_mapped_face[tf]);
            auto bounding_box = face_test->bounding_box();

            if (bounding_box.point_inside(quadrature_point_test) == true) {
             // std::cout<<"c "<<cell_test<< " f "<<face_no<<std::endl;
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
               // std::cout<< "q " <<q <<std::endl; 
                for (unsigned int i = 0; i < dofs_this_cell; ++i) {
                  //std::cout<< "q " <<q <<" i "<<i<<std::endl; 
#if COUPLED
      if(constructed_solution == 1)
      {
      local_vector(i) +=beta * (quadrature_point_test[0] +1) * fe_values_coupling_test_face[Potential].value(i, q) * 1 /
      (n_te * n_ftest);
   //   std::cout<< "face quadrature_point_test[0] "<<quadrature_point_test[0]<<std::endl;
      }
      else{
                        local_vector(i) += beta *
                            fe_values_coupling_test_face[Potential].value(i, q) * 1 /
                            (n_te * n_ftest);
      }

#else
                local_vector(i) +=
                      fe_values_coupling_test_face[Potential].value(i, q) * 1 /
                      (n_te * n_ftest);
#endif
                }
              }
              constraints.distribute_local_to_global(
                  local_vector, local_dof_indices_test, system_rhs);
            }
          }
          }
          //-------------face ende----------------- 

          if (insideCell_test) {
         //   std::cout<<"cell"<<std::endl;
            Point<dim> quadrature_point_test_mapped_cell =
                mapping.transform_real_to_unit_cell(cell_test,
                                                    quadrature_point_test);
           // std::cout << "quadrature_point_test_mapped_cell "
             //         << quadrature_point_test_mapped_cell << std::endl;
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
            
              local_vector = 0;
             for (unsigned int i = 0; i < dofs_per_cell; i++) {
            
              //std::cout<<"fe_values_coupling_test[Potential].value(i, 0) "<<fe_values_coupling_test[Potential].value(i, 0)<<std::endl;
#if COUPLED
if(constructed_solution == 1){
              local_vector(i) += beta* (quadrature_point_test[0] +1) * fe_values_coupling_test[Potential].value(i, 0);;
              //std::cout<< "cell quadrature_point_test[0] "<<quadrature_point_test[0]<<std::endl;
            }else
            {
             local_vector(i) += beta *
              fe_values_coupling_test[Potential].value(i, 0);
            }

#else
              local_vector(i) +=
              fe_values_coupling_test[Potential].value(i, 0) ;
#endif              
              
              
              
             
            }
           
            constraints.distribute_local_to_global(
                local_vector, local_dof_indices_test, system_rhs);
          }
//#endif
#if COUPLED 
// TODO right hand side
      for (unsigned int q_avag = 0; q_avag < nof_quad_points;
                   q_avag++) {
       // Quadrature weights and points
        quadrature_point_trial = quadrature_points_circle[q_avag];
        //std::cout<<"quadrature_point_trial "<<quadrature_point_trial<<std::endl;


        double weight;
        double C_avag;
        if (AVERAGE) {
          double perimeter = 2.0 * numbers::PI * radius;
          double h_avag = perimeter / (nof_quad_points);

          double weights_odd = 4.0 / 3.0 * h_avag;
          double weights_even = 2.0 / 3.0 * h_avag;
          double weights_first_last = h_avag / 3.0;

          C_avag = 1.0 / (2.0 * numbers::PI);
      
          if (q_avag == 0)
            weight = 2 * weights_first_last;
          else {

            if (q_avag % 2 == 0)
              weight = weights_even;
            else
              weight = weights_odd;
          }
          //weight = ((2.0 * numbers::PI * radius) / (nof_quad_points));
        } else {
          weight = 1.0;
          C_avag = 1.0;
        }
        //weight = 1.0;
        C_avag = 1.0;
        weight = 1.0 / nof_quad_points;
        // C_avag = 1.0;
        unsigned int n_tr;
        //std::cout<<"C_avag " << C_avag << " weight " <<weight <<std::endl;
#if TEST
    auto cell_trial_array = GridTools::find_all_active_cells_around_point(
        mapping, dof_handler_Omega, quadrature_point_trial, 1e-10, marked_vertices);
    n_tr= cell_trial_array.size();
   // std::cout << "cell_trial_array " << cell_trial_array.size() << std::endl;

    for (auto cellpair : cell_trial_array)
#else
    auto cell_trial = GridTools::find_active_cell_around_point(
        dof_handler_Omega, quadrature_point_trial);
    n_tr = 1;
#endif
   {
#if TEST
      auto cell_trial = cellpair.first;
   //  pcout<<"cell_trial " <<cell_trial<<std::endl;
#endif


      if (cell_trial != dof_handler_Omega.end())
           if (cell_trial->is_locally_owned() &&
                 cell_test->is_locally_owned()) 
        {
          cell_trial->get_dof_indices(local_dof_indices_trial);

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

        unsigned int n_ftrial = 0;//wie viel faces der celle liegen am Punkt
        std::vector<unsigned int> face_no_trial;
        for (unsigned int face_no = 0;
              face_no < GeometryInfo<dim>::faces_per_cell;
              face_no++) {
          typename DoFHandler<dim>::face_iterator face_trial =
              cell_trial->face(face_no);
          auto bounding_box = face_trial->bounding_box();
          if(bounding_box.point_inside(
                          quadrature_point_trial,
                          distance_tolerance) == true)
          {
            n_ftrial += 1;
            face_no_trial.push_back(face_no);
          }
          
        }

        if (n_ftrial == 0) {
          insideCell_trial = true;
          n_ftrial = 1;
          face_no_trial.push_back(0);

        } else {
          insideCell_trial = false;
        }
        std::cout<<"cell_trial "<< cell_test <<" insideCell_test "<<insideCell_test <<" n_ftest "<<n_ftest<<" n_te "<<n_te<< 
        " cell_trial "<< cell_trial <<" insideCell_trial "<<insideCell_trial <<" n_ftrial "<<n_ftrial<<" n_tr "<<n_tr<<std::endl;

 //std::cout<<"n_tr "<<n_tr <<" n_ftrial "<<n_ftrial<<" n_te "<< n_te <<" n_ftest "<< n_ftest<<std::endl;
        for (unsigned int ftest = 0; ftest < n_ftest; ftest++) {

          //std::cout<<"face_no_test[ftest] "<<face_no_test[ftest] <<std::endl;
          Point<dim - 1> quadrature_point_test_mapped_face =
              mapping.project_real_point_to_unit_point_on_face(
                  cell_test, face_no_test[ftest],
                  quadrature_point_test);
                  for(unsigned int tf = 0; tf < dim -1; tf++)
                  quadrature_point_test_mapped_face[tf] = std::max(0.0,quadrature_point_test_mapped_face[tf]);
                  
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
                    for(unsigned int tf = 0; tf < dim -1; tf++)
                    quadrature_point_trial_mapped_face[tf] = std::max(0.0,quadrature_point_trial_mapped_face[tf]);
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
                cell_trial, face_no_trial[ftrial]);

            V_U_matrix_coupling = 0;

            double psi_potential_test;
            double psi_potential_trial;
           
            // V_U_matrix_coupling
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
                V_U_matrix_coupling(i, j) +=   beta *
                    psi_potential_test * psi_potential_trial *
                    C_avag * weight * 1 /
                    (n_tr * n_ftrial) * 1 / (n_te * n_ftest);
              }
            }
           constraints.distribute_local_to_global(
                V_U_matrix_coupling, local_dof_indices_test,
                local_dof_indices_trial, system_matrix);

          }
        }
      }else
        std::cout<<"düdüm1 - assem"<<std::endl;



        

        }
   }
#endif
        }
    }
  }
  
  if (geo_conf == GeometryConfiguration::TwoD_OneD || geo_conf == GeometryConfiguration::ThreeD_OneD) {
    pcout<<"2D/1D  3D/1D"<<std::endl;
    std::vector<types::global_dof_index> local_dof_indices_test(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices_trial(dofs_per_cell);

    FullMatrix<double> V_U_matrix_coupling(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> v_U_matrix_coupling(dofs_per_cell_omega, dofs_per_cell);
    FullMatrix<double> V_u_matrix_coupling(dofs_per_cell, dofs_per_cell_omega);
    FullMatrix<double> v_u_matrix_coupling(dofs_per_cell_omega,
                                           dofs_per_cell_omega);

    bool insideCell_test = true;
    bool insideCell_trial = true;

MappingCartesian<dim> mymapping;
#if PAPER_SOLUTION
    double beta = 2 * numbers::PI * D * radius;
    g = beta;
#else
double beta =g;
#endif
#if !COUPLED
  beta =1;//g
#endif
    cell_omega = dof_handler_omega.begin_active();
    endc_omega = dof_handler_omega.end();

  for (; cell_omega != endc_omega; ++cell_omega) //man braucht nur das auskommentieren für einzelnen Punkt, einzelner Punkt, dann auch Vessel
    {
      //if (cell_omega->is_locally_owned())
 //     {
    fe_values_omega.reinit(cell_omega);
      cell_omega->get_dof_indices(local_dof_indices_omega);
      dof_omega_local_2_global(dof_handler_omega, local_dof_indices_omega);
  
  //std::vector<Point<dim_omega>> quadrature_points_omega = {Point<dim_omega>(std::sqrt(2)/2.0 + arrr)};//
 std::vector<Point<dim_omega>> quadrature_points_omega = fe_values_omega.get_quadrature_points();
//
          //
      for (unsigned int p = 0; p < quadrature_points_omega.size(); p++)
       {
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
       // pcout<<"quadrature_point_coupling "<<quadrature_point_coupling<<std::endl;
   
    
        unsigned int n_te;
#if TEST
    auto cell_test_first = GridTools::find_active_cell_around_point(
          cache, quadrature_point_test, cell_start, marked_vertices);
   #if FASTER
   auto cell_test_array = find_all_active_cells_around_point<dim, dim>(
                       mapping, triangulation, quadrature_point_test,1e-10 ,cell_test_first, &cache.get_vertex_to_cell_map());
   #else
      auto cell_test_array = GridTools::find_all_active_cells_around_point(
                       mapping, triangulation, quadrature_point_test,1e-10 ,cell_test_first);
   #endif




        n_te = cell_test_array.size();    
       // pcout << "quadrature_point_omega "<<quadrature_point_omega<<" cell_test_array " << cell_test_array.size() << std::endl;
        for (auto cellpair : cell_test_array)
#else
        auto cell_test = GridTools::find_active_cell_around_point(
            dof_handler_Omega, quadrature_point_test);
        n_te = 1;
#endif

        {
#if TEST
     auto cell_test_tri = cellpair.first;
         typename DoFHandler<dim>::active_cell_iterator
        cell_test = dof_handler_Omega.begin_active(cell_test_tri->level()),
         endc_test = dof_handler_Omega.end();

    for (; cell_test != endc_test; ++cell_test) {
      if(cell_test->level() == cell_test_tri->level() && cell_test->index() == cell_test_tri->index())
      {
        //std::cout<<"break"<<std::endl;
        break;
      }
    }
    if (cell_test == dof_handler_Omega.end())
    {
      pcout<<"cell_test nicht gefunden "<<cell_test_tri<<" : "<<cell_test<<std::endl;
    }
    
    cell_start = cell_test;
    //pcout<<"cellcomp " <<cell_test_tri->index()<<" " <<cell_test_tri<<" : "<<cell_test<<std::endl;
#endif

#if 1// USE_MPI_ASSEMBLE
          if (cell_test != dof_handler_Omega.end())
            if (cell_test->is_locally_owned())
#endif
            {
              //std::cout<<"cell_test "<<cell_test->center() <<std::endl;
              cell_test->get_dof_indices(local_dof_indices_test);

              std::vector<unsigned int> face_no_test;
              unsigned int n_ftest = 0;
              for (unsigned int face_no = 0;
                   face_no < GeometryInfo<dim>::faces_per_cell; face_no++) {
                typename DoFHandler<dim>::face_iterator face_test =
                    cell_test->face(face_no);
                auto bounding_box = face_test->bounding_box();
                
               /* pcout <<"quadrature_point_test "<<quadrature_point_test <<" points "
                 <<bounding_box.get_boundary_points().first<<" | " <<bounding_box.get_boundary_points().second<<
                " length "<<bounding_box.side_length(0)<<" "<<bounding_box.side_length(1)<<" "<<bounding_box.side_length(2)<<std::endl;
               */
                
                    if(bounding_box.point_inside(quadrature_point_test,
                                              distance_tolerance) == true)
                                              {
                                                n_ftest += 1;
                                                face_no_test.push_back(face_no);
                                              }
                 //  n_ftest += bounding_box.signed_distance (quadrature_point_test) <= 0.0000001;
                
                /*if(bounding_box.point_inside(quadrature_point_test,
                                              distance_tolerance) != (bounding_box.signed_distance (quadrature_point_test) <= 0.00001))
                std::cout<<"pointinside " <<(bounding_box.point_inside(quadrature_point_test,
                                              distance_tolerance) == true) << " | "<< (bounding_box.signed_distance (quadrature_point_test) <= 0.00001)<<std::endl;
                */            
                
              }
              if (n_ftest == 0) {
                insideCell_test = true;
                n_ftest = 1;
                face_no_test.push_back(0);
              } else {
                insideCell_test = false;
              
              }



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

#if !COUPLED || VESSEL
              //pcout << "not coupled" << std::endl;
              //-------------face -----------------
           
              if (!insideCell_test) {
              // pcout << "Omega rhs face " << std::endl;
             /* Point<dim - 1> quadrature_point_test_mapped_face =
                      mapping.project_real_point_to_unit_point_on_face(
                          cell_test, 0, quadrature_point_test);*/
               // pcout<<"cell "<<cell_test<<std::endl;
                for (unsigned int face_no = 0;
                     face_no < GeometryInfo<dim>::faces_per_cell; face_no++) {
                  typename DoFHandler<dim>::face_iterator face_test =
                      cell_test->face(face_no);

                  
                  Point<dim - 1> quadrature_point_test_mapped_face = 
                      mapping.project_real_point_to_unit_point_on_face(//das ist nicht reference Face, sondern face in eigenen Koordinatensystem, wird nochmal gedreht 
                          cell_test, face_no, quadrature_point_test);// für jedes Face braucht man eigenes Referenzface, welche dann gedreht wird und dann zu richtigen quadrature point wird
                  
                  if(face_no == 2 || face_no ==3)//TODO checken warum man das machen muss
                  quadrature_point_test_mapped_face = {quadrature_point_test_mapped_face[1],quadrature_point_test_mapped_face[0] };
                  
                //  std::cout<<"face_no "<<face_no<<" ----quadrature_point_test "<<quadrature_point_test<<" quadrature_point_test_mapped_face "<<quadrature_point_test_mapped_face<<std::endl;
                  for(unsigned int tf = 0; tf < dim -1; tf++)
                  quadrature_point_test_mapped_face[tf] = std::max(0.0,quadrature_point_test_mapped_face[tf]);
                  
                  auto bounding_box = face_test->bounding_box();
                  //std::cout<<"quadrature_point_test "<<quadrature_point_test<<" quadrature_point_test_mapped_face "<<quadrature_point_test_mapped_face<<std::endl;

                  if (bounding_box.point_inside(quadrature_point_test,
                                               distance_tolerance) == true) {
                 /*std::cout<<"TEST cell_trial "<< cell_test <<" insideCell_test "<<insideCell_test <<" n_ftest "<<n_ftest<<" n_te "<<n_te<<" face_no_test ";
                 for(const auto& kk : face_no_test)
                 std::cout<<kk<<" ";
                 std::cout<<std::endl;*/
                 
                  // if(bounding_box.signed_distance (quadrature_point_test) <= 0.0000001){
                 
                    //std::cout<<" n_te "<< n_te<<" n_ftest "<<n_ftest<<std::endl;
                    
                   
                 /*   if(std::abs(fe_values_coupling_test_face.normal_vector(0)[1])> 0.000001)
                        std::vector<Point<dim - 1>> quadrature_point_test_face = {
                          Point<dim-1>({quadrature_point_test_mapped_face[1],quadrature_point_test_mapped_face[0]})};
                    else*/

                  std::vector<Point<dim - 1>> quadrature_point_test_face = {
                          quadrature_point_test_mapped_face};

                    const Quadrature<dim - 1> my_quadrature_formula_test(
                        quadrature_point_test_face, my_quadrature_weights);

                    FEFaceValues<dim> fe_values_coupling_test_face(
                        mapping, fe_Omega, my_quadrature_formula_test, update_flags_coupling);

                    fe_values_coupling_test_face.reinit(cell_test, face_no);
                   
                    unsigned int n_face_points =
                        fe_values_coupling_test_face.n_quadrature_points;
                       // pcout <<"Boundingbox "<<fe_values_coupling_test_face.normal_vector(0)<<std::endl;
                       // pcout <<"Boundingbox "<<bounding_box.get_boundary_points().first<<" | " <<bounding_box.get_boundary_points().second<<
                //" length "<<bounding_box.side_length(0)<<" "<<bounding_box.side_length(1)<<" "<<bounding_box.side_length(2)<<std::endl;
                     //  std::cout <<"quadrature_point(0) " <<fe_values_coupling_test_face.get_quadrature_points().size()<<" v "<<fe_values_coupling_test_face.get_quadrature_points()[0]<<std::endl;
                       // std::cout<< "face_no "<<face_no<<" present_face_no "<<fe_values_coupling_test_face.get_face_number()<<" present_face_index "
                      //  <<fe_values_coupling_test_face.get_face_index()<<std::endl;
                    unsigned int dofs_this_cell =
                        fe_values_coupling_test_face.dofs_per_cell;
                    
                    local_vector = 0;
                  //  std::cout<<"n_face_points "<<n_face_points <<" dofs_this_cell "<<dofs_this_cell<<std::endl;
                    for (unsigned int q = 0; q < n_face_points; ++q) {
                      for (unsigned int i = 0; i < dofs_this_cell; ++i) {
                        double z;
                        if(constructed_solution == 1)
                          z = (1 + quadrature_point_omega[0]);
                        else if(constructed_solution == 2)
                          z = 1;
                        else if(constructed_solution == 3  )
                          z = (1 + quadrature_point_omega[0]);
                        else
                          z = 0;
                        local_vector(i) +=
                           beta * fe_values_coupling_test_face[Potential].value(i,
                                                                         q) *
                           1.0 / (n_te * n_ftest) *
                           z *fe_values_omega.JxW(p);
                        

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
                const unsigned int n_q_points = fe_values_coupling_test.n_quadrature_points;
                 for (unsigned int q = 0; q < n_q_points; ++q) {
                for (unsigned int i = 0; i < dofs_per_cell; i++) {
                  
                  double z;
                  if(constructed_solution == 1)
                    z = (1 + quadrature_point_omega[0]);
                  else if(constructed_solution == 2)
                    z = 1;
                  else if (constructed_solution == 3)
                    z = (1 + quadrature_point_omega[0]);
                  else 
                    z = 0;
                  local_vector(i) +=
                  beta *  fe_values_coupling_test[Potential].value(i, q) * z *fe_values_omega.JxW(p);
                }
                constraints.distribute_local_to_global(
                    local_vector, local_dof_indices_test, system_rhs);
              }
              }
#endif

#if COUPLED ||VESSEL
              // pcout << "coupled " << std::endl;
              for (unsigned int q_avag = 0; q_avag < nof_quad_points;
                   q_avag++) {
                // Quadrature weights and points
                quadrature_point_trial = quadrature_points_circle[q_avag];
             //  std::cout<< "quadrature_point_coupling "<<quadrature_point_coupling <<" quadrature_point_trial "<<quadrature_point_trial<<std::endl;
                double weight;
                double C_avag;
                if (AVERAGE) {
                  double perimeter = 2.0 * numbers::PI * radius;
                  double h_avag = perimeter / (nof_quad_points);

                  double weights_odd = 4.0 / 3.0 * h_avag;
                  double weights_even = 2.0 / 3.0 * h_avag;
                  double weights_first_last = h_avag / 3.0;

                  C_avag = 1.0 / (2.0 * numbers::PI);
             
                  if (q_avag == 0)
                    weight = 2 * weights_first_last;
                  else {

                    if (q_avag % 2 == 0)
                      weight = weights_even;
                    else
                      weight = weights_odd;
                  }
                  //weight = ((2.0 * numbers::PI * radius) / (nof_quad_points));
                } else {
                  weight = 1.0;
                  C_avag = 1.0;
                }
                //weight = 1.0;
                //C_avag = 1.0;
                weight = 1.0 / nof_quad_points;
                C_avag = 1.0;
                unsigned int n_tr;
#if TEST
              

            auto cell_trial_first = GridTools::find_active_cell_around_point(
                 cache, quadrature_point_trial, cell_start, marked_vertices);

			#if FASTER
             auto cell_trial_array = find_all_active_cells_around_point<dim, dim>(
                       mapping, triangulation, quadrature_point_trial,1e-10 ,cell_trial_first, &cache.get_vertex_to_cell_map());
             #else
                          auto cell_trial_array = GridTools::find_all_active_cells_around_point(
                       mapping, triangulation, quadrature_point_trial,1e-10 ,cell_trial_first);
             #endif



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
              auto cell_trial_tri = cellpair_trial.first;
                  typename DoFHandler<dim>::active_cell_iterator
                  cell_trial = dof_handler_Omega.begin_active(cell_trial_tri->level()),
                  endc_trial = dof_handler_Omega.end();

              for (; cell_trial != endc_trial; ++cell_trial) {
                if(cell_trial->level() == cell_trial_tri->level() && cell_trial->index() == cell_trial_tri->index())
                {
                  //std::cout<<"break"<<std::endl;
                  break;
                }
              }
                if (cell_trial == dof_handler_Omega.end())
                {
                  pcout<<"cell_trial nicht gefunden "<<cell_trial_tri<<" : "<<cell_trial<<std::endl;
                }
                //pcout<<"cellcomp trial " <<cell_trial_tri<<" : "<<cell_trial<<std::endl;
#endif
                 if (cell_trial != dof_handler_Omega.end())
                    if (cell_trial->is_locally_owned() &&
                       cell_test->is_locally_owned()) 
                     {
                     
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

                      unsigned int n_ftrial = 0;//wie viel faces der celle liegen am Punkt
                      std::vector<unsigned int> face_no_trial;
                      for (unsigned int face_no = 0;
                           face_no < GeometryInfo<dim>::faces_per_cell;
                           face_no++) {
                        typename DoFHandler<dim>::face_iterator face_trial =
                            cell_trial->face(face_no);
                        auto bounding_box = face_trial->bounding_box();
                        if(bounding_box.point_inside(
                                        quadrature_point_trial,
                                        distance_tolerance) == true)
                        {
                          n_ftrial += 1;
                          face_no_trial.push_back(face_no);
                        }
        
                      }
                      if (n_ftrial == 0) {
                        insideCell_trial = true;
                        n_ftrial = 1;
                        face_no_trial.push_back(0);

                      } else {
                        insideCell_trial = false;

                      }
                     // std::cout<<"insideCell_test " <<insideCell_test <<" insideCell_trial " << insideCell_trial<<std::endl;
                     
                   /*   std::cout<<"cell_test "<< cell_test <<" insideCell_test "<<insideCell_test <<" n_ftest "<<n_ftest<<" n_te "<<n_te<< " face_no_test ";
                      for(const auto& kk : face_no_test)
                      std::cout<<kk<<" ";
                      std::cout<<" cell_trial "<< cell_trial <<" insideCell_trial "<<insideCell_trial <<" n_ftrial "<<n_ftrial<<" n_tr "<<n_tr<<" face_no_trial ";
                      for(const auto& kk : face_no_trial)
                      std::cout<<kk<<" ";
                      std::cout <<std::endl;*/
                      
                      
                      for (unsigned int ftest = 0; ftest < n_ftest; ftest++) {
                        //std::cout<<"ftest "<<ftest<<std::endl; 
                        //std::cout<<" face_no_test[ftest] "<< face_no_test[ftest]<<std::endl; 
                        Point<dim - 1> quadrature_point_test_mapped_face =
                            mapping.project_real_point_to_unit_point_on_face(
                                cell_test, face_no_test[ftest],
                                quadrature_point_test);
                         if(face_no_test[ftest]== 2 ||face_no_test[ftest] ==3)
                            quadrature_point_test_mapped_face = {quadrature_point_test_mapped_face[1],quadrature_point_test_mapped_face[0] };
                        
                        for(unsigned int tf = 0; tf < dim -1; tf++)
                        quadrature_point_test_mapped_face[tf] = std::max(0.0,quadrature_point_test_mapped_face[tf]);

                        std::vector<Point<dim - 1>> quadrature_point_test_face =
                            {quadrature_point_test_mapped_face};
                        const Quadrature<dim - 1> my_quadrature_formula_test(
                            quadrature_point_test_face, my_quadrature_weights);
                        FEFaceValues<dim> fe_values_coupling_test_face(
                            fe_Omega, my_quadrature_formula_test,
                            update_flags_coupling);
                        fe_values_coupling_test_face.reinit(
                            cell_test, face_no_test[ftest]);
                     /* std::cout <<"quadrature_point_test(0) " <<fe_values_coupling_test_face.get_quadrature_points().size()<<
                      " v "<<fe_values_coupling_test_face.get_quadrature_points()[0]<<std::endl;*/

                      if(fe_values_coupling_test_face.get_quadrature_points()[0].distance(quadrature_point_test) > 0.0000001 && !insideCell_test)
                      {
                          std::cerr << "quadrature_point_test wrong " <<fe_values_coupling_test_face.get_quadrature_points()[0].distance(quadrature_point_test)<< std::endl;
                          throw std::runtime_error("Falsch");  
                        }
                          for (unsigned int ftrial = 0; ftrial < n_ftrial;
                             ftrial++) {

                          Point<dim - 1> quadrature_point_trial_mapped_face =
                              mapping.project_real_point_to_unit_point_on_face(
                                  cell_trial, face_no_trial[ftrial],
                                  quadrature_point_trial);

                        if( face_no_trial[ftrial] == 2 || face_no_trial[ftrial] ==3)
                            quadrature_point_trial_mapped_face = {quadrature_point_trial_mapped_face[1],quadrature_point_trial_mapped_face[0] };
                        
                        for(unsigned int tf = 0; tf < dim -1; tf++)
                        quadrature_point_trial_mapped_face[tf] = std::max(0.0,quadrature_point_trial_mapped_face[tf]);

                          std::vector<Point<dim - 1>>
                              quadrature_point_trial_face = {
                                  quadrature_point_trial_mapped_face};
                          const Quadrature<dim - 1> my_quadrature_formula_trial(
                              quadrature_point_trial_face,
                              my_quadrature_weights);
                          FEFaceValues<dim> fe_values_coupling_trial_face(
                              fe_Omega, my_quadrature_formula_trial,
                              update_flags_coupling);
                             // std::cout<<"ftrial "<<ftrial<<std::endl; 
                             // std::cout<<" face_no_trial[ftrial] "<< face_no_trial[ftrial]<<std::endl; 
                          fe_values_coupling_trial_face.reinit(
                              cell_trial, face_no_trial[ftrial]);

                              if(fe_values_coupling_trial_face.get_quadrature_points()[0].distance(quadrature_point_trial) > 0.0000001 && !insideCell_trial)
                              //if(((fe_values_coupling_trial_face.get_quadrature_points()[0][1] - quadrature_point_trial[1]) > 0.0000001)|| 
                              //((fe_values_coupling_trial_face.get_quadrature_points()[0][2] - quadrature_point_trial[2]) > 0.0000001))
                              {
                                std::cout<<fe_values_coupling_trial_face.get_quadrature_points()[0] << " vs " << quadrature_point_trial<<std::endl;
                                std::cerr << "quadrature_point_trial wrong " <<fe_values_coupling_trial_face.get_quadrature_points()[0].distance(quadrature_point_trial)<< std::endl;
                                throw std::runtime_error("Falsch");  
                            }
                              /*  std::cout <<"quadrature_point_trial(0) " <<fe_values_coupling_trial_face.get_quadrature_points().size()<<
                      " v "<<fe_values_coupling_trial_face.get_quadrature_points()[0]<<std::endl;*/
                          V_U_matrix_coupling = 0;
                          v_U_matrix_coupling = 0;
                          V_u_matrix_coupling = 0;
                          v_u_matrix_coupling = 0;

                          double psi_potential_test;
                          double psi_potential_trial;
                          // V_U_matrix_coupling
                          for (unsigned int i = 0; i < dofs_per_cell; i++) {
                            if (insideCell_test)
                              psi_potential_test =
                                  fe_values_coupling_test[Potential].value(i,
                                                                           0);
                            else
                              psi_potential_test =
                                  fe_values_coupling_test_face[Potential].value(
                                      i, 0);
                          //  std::cout<<"psi_potential_test " <<psi_potential_test<< std::endl;
                            for (unsigned int j = 0; j < dofs_per_cell; j++) {
                              if (insideCell_trial)
                                psi_potential_trial =
                                    fe_values_coupling_trial[Potential].value(
                                        j, 0);
                              else
                                psi_potential_trial =
                                    fe_values_coupling_trial_face[Potential]
                                        .value(j, 0);
                           //     std::cout<<"psi_potential_trial " <<psi_potential_trial<< std::endl;
                              V_U_matrix_coupling(i, j) +=
                                  g * psi_potential_test * psi_potential_trial *
                                  C_avag * weight  * 1 /
                                  (n_tr * n_ftrial) * 1 / (n_te * n_ftest) * fe_values_omega.JxW(p);
                            }
                          }
                          constraints.distribute_local_to_global(
                              V_U_matrix_coupling, local_dof_indices_test,
                              local_dof_indices_trial, system_matrix);
#if !VESSEL
                          //v_U_matrix_coupling
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

                          //V_u_matrix_coupling
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
                              if(q_avag == 0)
                              V_u_matrix_coupling(i, j) +=
                                  -g * psi_potential_test *
                                  fe_values_omega[Potential_omega].value(j, p) *
                                  fe_values_omega.JxW(p) * 1 /
                                  (n_tr * n_ftrial) * 1 / (n_te * n_ftest);
                                  
                            }
                          }
                          constraints.distribute_local_to_global(
                              V_u_matrix_coupling, local_dof_indices_test,
                              local_dof_indices_omega, system_matrix);

                          //v_u_matrix_coupling
                          for (unsigned int i = 0; i < dofs_per_cell_omega;
                               i++) {
                            for (unsigned int j = 0; j < dofs_per_cell_omega;
                                 j++) {
                              if(q_avag == 0)
                              v_u_matrix_coupling(i, j) +=
                                  g *
                                  fe_values_omega[Potential_omega].value(j, p) *
                                  fe_values_omega[Potential_omega].value(i, p) *
                                  fe_values_omega.JxW(p) * 1 /
                                  (n_tr * n_ftrial) * 1 / (n_te * n_ftest);
                                 
                            }
                          }
                          constraints.distribute_local_to_global(
                              v_u_matrix_coupling, local_dof_indices_omega,
                              local_dof_indices_omega, system_matrix);
#endif
                          // --------------------------cell ende
                          // --------------------
                        }//for n_ftrial
                      } //for n_ftest
                    }else //if cell_trial->is_locally_owned()
                      std::cout<<"düdüm1 - assem"<<std::endl;
                } //for cellpair_trial : cell_trial_array
                   }//for nof_quad_points circle
             
//#endif
               
#endif
}//if cell_test->is_locally_owned()
}//for cellpair : cell_test_array
}//for quadrature_points_omega
//}//if (cell_omega->is_locally_owned())
}//for cell_omega
}//if geoconfig 1D/3D

}
#endif
  // std::cout << "ende coupling loop" << std::endl;


 /* for (unsigned int i = 0; i < dof_handler_Omega.n_dofs() + dof_handler_omega.n_dofs(); i++) // dof_table.size()
  {
    // if(dof_table[i].first.first == 1 || dof_table[i].first.first == 3)
    {
      if (system_matrix.el(i, i) == 0) {
std::cout<<"ja"<<std::endl;
        system_matrix.add(i, i, 1);
      }
    }
  }*/
  
  pcout<<"Start compress " <<std::endl;
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}

template <int dim, int dim_omega>
template <int _dim>
void LDGPoissonProblem<dim, dim_omega>::dof_omega_local_2_global(
    const DoFHandler<_dim> &dof_handler_omega,//dof_handler wird in derzeitiger Implemtierung nicht mehr benötigt
    std::vector<types::global_dof_index> &local_dof_indices_omega) {

  for (unsigned int i = 0; i < local_dof_indices_omega.size(); ++i) {
    //std::cout<<   local_dof_indices_omega[i] <<" ";     
    local_dof_indices_omega[i] =   start_VectorField_omega +  local_dof_indices_omega[i];         
    //std::cout<<   local_dof_indices_omega[i] <<std::endl;     
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
    const FEValuesExtractors::Scalar &Potential,
    bool no_gradient) {
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
        if(no_gradient)
        {
        //  pcout<<"no gradient" <<std::endl;* K_inverse_values[q]
         cell_matrix(i, j) += (psi_i_field  * psi_j_field)  + (psi_j_potential* psi_i_potential ) * cell_fe.JxW(q);
         }
        else{
        cell_matrix(i, j) +=
            ((psi_i_field * K_inverse_values[q] * psi_j_field) -
             (div_psi_i_field * psi_j_potential) -
             (grad_psi_i_potential * psi_j_field)) *
            cell_fe.JxW(q);
        }
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
    Vector<double> &local_vector, const Function<_dim> &Neumann_bc_function,
     const FEValuesExtractors::Vector VectorField,
    const FEValuesExtractors::Scalar Potential) {
  const unsigned int dofs_per_cell = face_fe.dofs_per_cell;
  const unsigned int n_q_points = face_fe.n_quadrature_points;

  std::vector<double> Neumann_bc_values(n_q_points);

  Neumann_bc_function.value_list(face_fe.get_quadrature_points(),
                                 Neumann_bc_values);

  for (unsigned int q = 0; q < n_q_points; ++q) {
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      const Tensor<1, _dim> psi_i_field = face_fe[VectorField].value(i, q);
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
LDGPoissonProblem<dim, dim_omega>::compute_errors(){
  /*std::cout << "compute_errors "
            << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << std::endl;*/
  double potential_l2_error, vectorfield_l2_error, potential_l2_error_omega,
      vectorfield_l2_error_omega;
  double global_potential_l2_error_omega, global_vectorfield_l2_error_omega;

    const ComponentSelectFunction<dim> potential_mask(dim,
                                                      dim + 1);
    const ComponentSelectFunction<dim> vectorfield_mask(std::make_pair(0, dim),
                                                        dim + 1);
    std::cout<<"h_min "<<h_min<<std::endl;
    const DistanceWeight<dim> distance_weight(alpha, radius, h_min); //, radius

    const ProductFunction<dim> connected_function_potential(potential_mask,
                                                            distance_weight);
    const ProductFunction<dim> connected_function_vectorfield(vectorfield_mask,
                                                              distance_weight);


/*std::cout<<"grow< "<< triangulation.n_active_cells()<<std::endl;*/
    cellwise_errors_Q.grow_or_shrink(triangulation.n_active_cells());
   cellwise_errors_U.grow_or_shrink(triangulation.n_active_cells());
   cellwise_errors_q.grow_or_shrink(triangulation_omega.n_active_cells());
   cellwise_errors_u.grow_or_shrink(triangulation_omega.n_active_cells());

/*
  Vector<double> cellwise_errors_U(
      triangulation.n_active_cells());
  Vector<double> cellwise_errors_Q(
      triangulation.n_active_cells());
*/

    /*pcout << "triangulation.n_active_cells() " << triangulation.n_active_cells()
          << " dof_handler_Omega.n_dofs() " << dof_handler_Omega.n_dofs()
          << " dof_handler_Omega.n_locally_owned_dofs() "
          << dof_handler_Omega.n_locally_owned_dofs() << " solution size "
          << solution.size() << " mpi "
          << Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) << std::endl;*/

    const QTrapezoid<1> q_trapez;
    const QIterated<dim> quadrature(q_trapez, degree + 2);

    VectorTools::integrate_difference(
        dof_handler_Omega, solution.block(0), true_solution, cellwise_errors_U, quadrature,
        VectorTools::L2_norm, &connected_function_potential); //
   
    potential_l2_error = VectorTools::compute_global_error(
        triangulation, cellwise_errors_U, VectorTools::L2_norm);
    
    VectorTools::integrate_difference(
        dof_handler_Omega, solution.block(0), true_solution, cellwise_errors_Q, quadrature,
        VectorTools::L2_norm, &connected_function_vectorfield);

    vectorfield_l2_error = VectorTools::compute_global_error(
        triangulation, cellwise_errors_Q, VectorTools::L2_norm);


//cellwise_errors_Q.print(std::cout);

 /* Vector<double> cellwise_errors_u(
      triangulation_omega.n_active_cells());
  Vector<double> cellwise_errors_q(
      triangulation_omega.n_active_cells());*/

      const ComponentSelectFunction<dim_omega> potential_mask_omega(
          dim_omega, dim_omega + 1);
      const ComponentSelectFunction<dim_omega> vectorfield_mask_omega(
          std::make_pair(0, dim_omega), dim_omega + 1);

    
      const QTrapezoid<1> q_trapez_omega;
      const QIterated<dim_omega> quadrature_omega(q_trapez_omega, degree + 2);
   // solution.block(1).print(std::cout);
      VectorTools::integrate_difference(
          dof_handler_omega, solution_omega, true_solution_omega,
          cellwise_errors_u, quadrature_omega, VectorTools::L2_norm,
          &potential_mask_omega);
//cellwise_errors_u.print(std::cout);
      potential_l2_error_omega = VectorTools::compute_global_error(
          triangulation_omega, cellwise_errors_u, VectorTools::L2_norm);

      VectorTools::integrate_difference(
          dof_handler_omega, solution_omega, true_solution_omega,
          cellwise_errors_q, quadrature_omega, VectorTools::L2_norm,
          &vectorfield_mask_omega);

      vectorfield_l2_error_omega = VectorTools::compute_global_error(
          triangulation_omega, cellwise_errors_q, VectorTools::L2_norm);

 return std::array<double, 4>{{potential_l2_error, vectorfield_l2_error,
                                potential_l2_error_omega,
                                vectorfield_l2_error_omega}};
}

template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::solve() {
  TimerOutput::Scope t(computing_timer, "solve");
  pcout << "Solving linear system... "<<std::endl;
  Timer timer;
  //solution = system_rhs;
  TrilinosWrappers::MPI::BlockVector completely_distributed_solution(
        system_rhs);
  completely_distributed_solution = solution;
 // Quadrature<dim_omega> quadrature_omega(fe_omega.get_unit_support_points());  // Quadrature points 
  IndexSet locally_owned_dofs_Omega;
  IndexSet locally_relevant_dofs_Omega;

  IndexSet locally_owned_dofs_omega_local;
  IndexSet locally_relevant_dofs_omega_local;

  locally_owned_dofs_Omega = dof_handler_Omega.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler_Omega, locally_relevant_dofs_Omega);
  locally_owned_dofs_omega_local = dof_handler_omega.locally_owned_dofs();
  DoFTools::extract_locally_relevant_dofs(dof_handler_omega, locally_relevant_dofs_omega_local);

// Ensure constraints are correctly set and closed
AffineConstraints<double> constraints;
// Add Dirichlet boundary conditions or other necessary constraints
// constraints.add_boundary_values(...);
constraints.close();

TrilinosWrappers::MPI::Vector solution_const_Omega;
solution_const_Omega.reinit(locally_owned_dofs_Omega, MPI_COMM_WORLD);
  MappingQ1<dim> mapping_Omega;  
//VectorTools::project(mapping_Omega, dof_handler_Omega, {},  QGauss<dim>(degree +1), true_solution, solution_const_Omega);
project<dim>(mapping_Omega, dof_handler_Omega, QGauss<dim>(degree +1), true_solution, solution_const_Omega, locally_owned_dofs_Omega,degree);
// solution_const_Omega.print(std::cout);

TrilinosWrappers::MPI::Vector solution_const_omega;
solution_const_omega.reinit(locally_owned_dofs_omega_local, MPI_COMM_WORLD);
  MappingQ1<dim_omega> mapping_omega;  
/*
{
   TrilinosWrappers::SparsityPattern trilinos_sparsity(locally_owned_dofs_omega_local, MPI_COMM_WORLD);
    DoFTools::make_sparsity_pattern(dof_handler_omega, trilinos_sparsity,{},true,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
    trilinos_sparsity.compress();
TrilinosWrappers::SparseMatrix system_matrix;
system_matrix.reinit(trilinos_sparsity);
TrilinosWrappers::MPI::Vector rhs(solution_const_omega);
MatrixCreator::create_mass_matrix(mapping_omega, dof_handler_omega, QGauss<dim_omega>(degree +1), system_matrix);
VectorTools::create_right_hand_side(dof_handler_omega,QGauss<dim_omega>(degree +1), true_solution_omega, rhs);
std::cout<<"system matrix"<<std::endl;
system_matrix.print(std::cout);
std::cout<<"rhs"<<std::endl;
rhs.print(std::cout);
    // Step 3: Solve the system
    SolverControl solver_control(1000, 1e-12);
    SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control);

    TrilinosWrappers::PreconditionIdentity precondition;
   // precondition.initialize(system_matrix);

    solver.solve(system_matrix, solution_const_omega, rhs, precondition);
}
*/

project<dim_omega>(mapping_omega, dof_handler_omega, QGauss<dim_omega>(degree +1), true_solution_omega, solution_const_omega, locally_owned_dofs_omega_local,degree);
//VectorTools::project(mapping_omega, dof_handler_omega, {},  QGauss<dim_omega>(degree +1), true_solution_omega, solution_const_omega);
//std::cout<<"sol"<<std::endl;
//solution_const_omega.print(std::cout);
/*TrilinosWrappers::MPI::Vector relevant_solution(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
solution_const_omega.update_ghost_values();
solution_const_Omega.update_ghost_values();*/
double l2_norm_solution_omega = solution_const_omega.l2_norm();
double l2_norm_solution_Omega = solution_const_Omega.l2_norm();

//completely_distributed_solution = system_rhs;
//completely_distributed_solution.block(0) = solution_const_Omega;
//completely_distributed_solution.block(1) =  solution_const_omega;

completely_distributed_solution.block(0) = std::sqrt(std::pow(l2_norm_solution_Omega,2)/solution_const_Omega.locally_owned_size());
completely_distributed_solution.block(1) = std::sqrt(std::pow(l2_norm_solution_omega,2)/solution_const_omega.locally_owned_size());

//completely_distributed_solution.block(0).compress(VectorOperation::insert);
//completely_distributed_solution.block(1).compress(VectorOperation::insert);

//completely_distributed_solution.block(1).print(std::cout);
pcout<<"l2_norm_solution_omega "<<l2_norm_solution_omega<<" completely_distributed_solution.block(1) "<<completely_distributed_solution.block(1).l2_norm()<<std::endl;
pcout<<"l2_norm_solution_Omega "<<l2_norm_solution_Omega<<" completely_distributed_solution.block(0) "<<completely_distributed_solution.block(0).l2_norm()<<std::endl;
#if SOLVE_BLOCKWISE// && COUPLED
  pcout<<"solve blockwise"<<std::endl;
#if A11SCHUR
pcout<<"A11 Schur"<<std::endl;
  const InverseMatrix A_inverse(system_matrix.block(0,0));

  
  TrilinosWrappers::MPI::Vector tmp(completely_distributed_solution.block(0));

 
  TrilinosWrappers::MPI::Vector schur_rhs(system_rhs.block(1));
  A_inverse.vmult(tmp, system_rhs.block(0));
  system_matrix.block(1, 0).vmult(schur_rhs, tmp);
  schur_rhs -= system_rhs.block(1);
 // schur_rhs.print(std::cout);

 SchurComplement schur_complement(system_matrix, A_inverse, system_rhs);
  
  
 // ReductionControl solver_control1(completely_distributed_solution.block(0).locally_owned_size(), tolerance * system_rhs.l2_norm(), reduction);
  SolverControl solver_control1(std::max((int)completely_distributed_solution.block(0).locally_owned_size(),1000), tolerance);
  SolverGMRES<TrilinosWrappers::MPI::Vector > solver(solver_control1);

TrilinosWrappers::PreconditionILUT preconditioner;
  TrilinosWrappers::PreconditionILUT::AdditionalData data;
  preconditioner.initialize(system_matrix.block(1, 1), data);

  solver.solve(schur_complement, completely_distributed_solution.block(1),schur_rhs, preconditioner);
  pcout<<"Schur complete "<<std::endl;

  system_matrix.block(0, 1).vmult(tmp, completely_distributed_solution.block(1));
  tmp *= -1;
  tmp += system_rhs.block(0);
 
  A_inverse.vmult(completely_distributed_solution.block(0), tmp);

 // A_inverse.vmult(completely_distributed_solution.block(0), system_rhs.block(0));//unkoppled
    pcout << "Number of iterations: " << solver_control1.last_step() << std::endl;
#else
pcout<<"A22 Schur"<<std::endl;
const InverseMatrix A_inverse(system_matrix.block(1,1));

  
  TrilinosWrappers::MPI::Vector tmp(completely_distributed_solution.block(1));

 
  TrilinosWrappers::MPI::Vector schur_rhs(system_rhs.block(0));
  A_inverse.vmult(tmp, system_rhs.block(1));
  system_matrix.block(0, 1).vmult(schur_rhs, tmp);
  schur_rhs -= system_rhs.block(0);
 // schur_rhs.print(std::cout);

 SchurComplement_A_22 schur_complement(system_matrix, A_inverse, system_rhs);


  //ReductionControl solver_control1(completely_distributed_solution.block(0).locally_owned_size(), tolerance * system_rhs.l2_norm(), reduction);
  SolverControl solver_control1(std::max((int)completely_distributed_solution.block(0).locally_owned_size(),1000), tolerance);
  //SolverGMRES<TrilinosWrappers::MPI::Vector >::AdditionalData datasolver(true);
 SolverGMRES<TrilinosWrappers::MPI::Vector > solver(solver_control1);


TrilinosWrappers::PreconditionILU preconditioner;
  TrilinosWrappers::PreconditionILU::AdditionalData data;
  preconditioner.initialize(system_matrix.block(0, 0), data);

  solver.solve(schur_complement, completely_distributed_solution.block(0),schur_rhs, preconditioner);
  pcout<<"Schur complete "<<std::endl;

  system_matrix.block(1, 0).vmult(tmp, completely_distributed_solution.block(0));
  tmp *= -1;
  tmp += system_rhs.block(1);
 //if(COUPLED)
  A_inverse.vmult(completely_distributed_solution.block(1), tmp);
//else
  //A_inverse.vmult(completely_distributed_solution.block(0), system_rhs.block(0));//unkoppled
pcout << "Number of iterations: " << solver_control1.last_step() << std::endl;
#endif
#else 
pcout<<"solve full"<<std::endl;


// Set up solver control
//ReductionControl solver_control22(dof_handler_Omega.n_locally_owned_dofs(), tolerance * system_rhs.l2_norm(), reduction);
SolverControl solver_control22(std::max((int)dof_handler_Omega.n_locally_owned_dofs(),1000), tolerance );//



if(geo_conf == GeometryConfiguration::TwoD_ZeroD || COUPLED==0)
//if(true)
{
  pcout<<"GeometryConfiguration::TwoD_ZeroD || COUPLED==0"<<std::endl;
TrilinosWrappers::PreconditionILU preconditioner_block_0;
TrilinosWrappers::PreconditionILU preconditioner_block_1;//PreconditionILU  PreconditionBlockJacobi
// Initialize the preconditioners with the appropriate blocks of the matrix
preconditioner_block_0.initialize(system_matrix.block(0, 0));  // ILU for block (0,0)
preconditioner_block_1.initialize(system_matrix.block(1, 1));  // ILU for block (1,1)
  SolverGMRES<TrilinosWrappers::MPI::Vector> solver(solver_control22);
  solver.solve(system_matrix.block(0,0), completely_distributed_solution.block(0), system_rhs.block(0),preconditioner_block_0);
  pcout<<"Solve Omega done"<<std::endl;
  TrilinosWrappers::PreconditionIdentity precondition;
  solver.solve(system_matrix.block(1,1), completely_distributed_solution.block(1), system_rhs.block(1),precondition);
 pcout<<"Solve omega done"<<std::endl;
}
else
{
  // Solve the system using the block preconditioner
  // Preconditioners for each block
TrilinosWrappers::PreconditionILU preconditioner_block_0;
TrilinosWrappers::PreconditionILU preconditioner_block_1;//PreconditionILU  PreconditionBlockJacobi
// Initialize the preconditioners with the appropriate blocks of the matrix
preconditioner_block_0.initialize(system_matrix.block(0, 0));  // ILU for block (0,0)
preconditioner_block_1.initialize(system_matrix.block(1, 1));  // ILU for block (1,1)
 SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control22);
 BlockPreconditioner block_preconditioner(preconditioner_block_0, preconditioner_block_1);
 solver.solve(system_matrix, completely_distributed_solution, system_rhs,block_preconditioner);
}

   pcout << "Number of iterations: " << solver_control22.last_step() << std::endl;
#endif
constraints.distribute(completely_distributed_solution);
solution = completely_distributed_solution;

 /*solution = system_rhs;
 SparseDirectUMFPACK A_direct;
 A_direct.solve(system_matrix, solution);*/

    

  
  timer.stop();

  pcout << "done (" << timer.cpu_time() << "s)" << std::endl;


  
  solution_omega.reinit(dof_handler_omega.n_dofs());
  for (unsigned int i = 0; i < dof_handler_omega.n_dofs(); i++) {
    types::global_dof_index dof_index = start_VectorField_omega + i;
      solution_omega[i] = solution[dof_index];
  }

#if 0// USE_MPI_ASSEMBLE
  // solution_omega.compress(VectorOperation::add);//TODO scauen was es noc fpr
#endif
}

template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::output_results() const {
  pcout << "Output_result" << std::endl;
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
  std::string ref_p_array = "_r_" + Utilities::int_to_string(n_refine,2) + "_p_" + Utilities::int_to_string(degree,2);
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler_Omega);
  
  data_out.add_data_vector(solution.block(0), solution_names); //, DataOut<dim>::type_cell_data  
 
  Vector<float>   subdomain(triangulation.n_active_cells());
  for (unsigned int i=0; i<subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
  
  data_out.add_data_vector(subdomain,"subdomain");
  data_out.build_patches(degree);
    const std::string filename = ("solution"  + ref_p_array  +"."+
                                  Utilities::int_to_string(
                                    triangulation.locally_owned_subdomain(),4));
  std::ofstream output((folder_name + filename + ".vtu").c_str());
  data_out.write_vtu(output);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
      {
        std::vector<std::string>    filenames;
        for (unsigned int i=0;
             i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
             i++)
          {
            filenames.push_back("solution" + ref_p_array  +"."+
                                Utilities::int_to_string(i,4) +
                                ".vtu");
          }
        std::ofstream master_output(folder_name +"solution" + ref_p_array+".pvtu");
        data_out.write_pvtu_record(master_output, filenames);
      }

 // ------analytical solution--------
/*
pcout << "analytical solution" << std::endl;
 DoFHandler<dim_omega> dof_handler_Lag(triangulation_omega);
  FESystem<dim_omega> fe_Lag(FESystem<dim_omega>(FE_DGQ<dim_omega>(degree), dim_omega),
                       FE_DGQ<dim_omega>(degree));
  dof_handler_Lag.distribute_dofs(fe_Lag);
  TrilinosWrappers::MPI::Vector solution_const;
  solution_const.reinit(dof_handler_Lag.locally_owned_dofs(), MPI_COMM_WORLD);

  VectorTools::interpolate(dof_handler_Lag, true_solution_omega, solution_const);
 solution_const.print(std::cout);
std::cout << "solution_const l2 " << solution_const.l2_norm()<<" "<< solution_const.l1_norm()<<std::endl;
*/

 {
  DoFHandler<dim> dof_handler_Lag(triangulation);
  FESystem<dim> fe_Lag(FESystem<dim>(FE_DGQ<dim>(degree), dim),
                       FE_DGQ<dim>(degree));
  dof_handler_Lag.distribute_dofs(fe_Lag);
  TrilinosWrappers::MPI::Vector solution_const;
  solution_const.reinit(dof_handler_Lag.locally_owned_dofs(), MPI_COMM_WORLD);

  VectorTools::interpolate(dof_handler_Lag, true_solution, solution_const);
  //solution_const.print(std::cout);
 

  DataOut<dim> data_out_const;
  data_out_const.attach_dof_handler(dof_handler_Lag);
  data_out_const.add_data_vector(solution_const, solution_names); //

  data_out_const.build_patches(degree);
  const std::string filename_const = ("solution_const"    + ref_p_array  +"."+
                                  Utilities::int_to_string(
                                    triangulation.locally_owned_subdomain(),4));
  std::ofstream output_const((folder_name+ filename_const + ".vtu").c_str());
  data_out_const.write_vtu(output_const);


if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
      {
        std::vector<std::string>    filenames;
        for (unsigned int i=0;
             i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
             i++)
          {
            filenames.push_back("solution_const"  + ref_p_array  +"."+
                                Utilities::int_to_string(i,4) +
                                ".vtu");
          }
        std::ofstream master_output(folder_name + "solution_const.pvtu");
        data_out_const.write_pvtu_record(master_output, filenames);
      }
 }
 //omega
 {
  DoFHandler<dim_omega> dof_handler_Lag(triangulation_omega);
  FESystem<dim_omega> fe_Lag(FESystem<dim_omega>(FE_DGQ<dim_omega>(degree), dim_omega),FE_DGQ<dim_omega>(degree));
  dof_handler_Lag.distribute_dofs(fe_Lag);
  TrilinosWrappers::MPI::Vector solution_const;
  solution_const.reinit(dof_handler_Lag.locally_owned_dofs(), MPI_COMM_WORLD);

  VectorTools::interpolate(dof_handler_Lag, true_solution_omega, solution_const);
  //solution_const.print(std::cout);
 

  DataOut<dim_omega> data_out_const;
  data_out_const.attach_dof_handler(dof_handler_Lag);
  std::vector<std::string> solution_names_omega;
  solution_names_omega.push_back("u");
  solution_names_omega.push_back("q");
  data_out_const.add_data_vector(solution_const, solution_names_omega); //

  data_out_const.build_patches(degree);
  const std::string filename_const = ("solution_omega_const"    + ref_p_array  +"."+
                                  Utilities::int_to_string(
                                    triangulation_omega.locally_owned_subdomain(),4));
  std::ofstream output_const((folder_name+ filename_const + ".vtu").c_str());
  data_out_const.write_vtu(output_const);


if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
      {
        std::vector<std::string>    filenames;
        for (unsigned int i=0;
             i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
             i++)
          {
            filenames.push_back("solution_omega_const"  + ref_p_array  +"."+
                                Utilities::int_to_string(i,4) +
                                ".vtu");
          }
        std::ofstream master_output(folder_name + "solution_omega_const.pvtu");
        data_out_const.write_pvtu_record(master_output, filenames);
      }
 }
 
//----------cell_wise error ---------------

{
    DataOut<dim> data_out_error;
    data_out_error.attach_triangulation(triangulation);
    data_out_error.add_data_vector(cellwise_errors_Q, "Q");
    data_out_error.add_data_vector(cellwise_errors_U, "U");
    data_out_error.build_patches();


  const std::string filename_error = ("error" +  ref_p_array + "."  +
                                  Utilities::int_to_string(
                                    triangulation.locally_owned_subdomain(),4));
  std::ofstream output_error((folder_name + filename_error + ".vtu").c_str());
  data_out_error.write_vtu(output_error);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
      {
        std::vector<std::string>    filenames;
        for (unsigned int i=0;
             i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
             i++)
          {
            filenames.push_back("error"  + ref_p_array  +"."+
                                Utilities::int_to_string(i,4) +
                                ".vtu");
          }
        std::ofstream master_output(folder_name + "error"  + ref_p_array  + "."  +"pvtu");
        data_out_error.write_pvtu_record(master_output, filenames);
      }
 }
// omega error
{
  DataOut<dim_omega> data_out_error;
    data_out_error.attach_triangulation(triangulation_omega);
    data_out_error.add_data_vector(cellwise_errors_q, "q");
    data_out_error.add_data_vector(cellwise_errors_u, "u");
    data_out_error.build_patches();


  const std::string filename_error = ("error_omega" +  ref_p_array + "."  +
                                  Utilities::int_to_string(
                                    triangulation_omega.locally_owned_subdomain(),4));
  std::ofstream output_error((folder_name + filename_error + ".vtu").c_str());
  data_out_error.write_vtu(output_error);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
      {
        std::vector<std::string>    filenames;
        for (unsigned int i=0;
             i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
             i++)
          {
            filenames.push_back("error_omega"  + ref_p_array  +"."+
                                Utilities::int_to_string(i,4) +
                                ".vtu");
          }
        std::ofstream master_output(folder_name + "error_omega"  + ref_p_array  + "."  +"pvtu");
        data_out_error.write_pvtu_record(master_output, filenames);
      }
}
   //-----omega-----------
 // std::cout << "omega solution" << std::endl;

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
    const std::string filename_omega = ("solution_omega"    + ref_p_array  +"."+
                                  Utilities::int_to_string(
                                    triangulation_omega.locally_owned_subdomain(),4));
  std::ofstream output_omega((folder_name + filename_omega + ".vtu").c_str());
  data_out_omega.write_vtu(output_omega);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
      {
        std::vector<std::string>    filenames;
        for (unsigned int i=0;
             i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
             i++)
          {
            filenames.push_back("solution_omega"  + ref_p_array  +"."+
                                Utilities::int_to_string(i,4) +
                                ".vtu");
          }
        std::ofstream master_output(folder_name +"solution_omega.pvtu");
        data_out_omega.write_pvtu_record(master_output, filenames);
      }




/*
//Boundary
DataPostprocessors::BoundaryIds<dim> boundary_ids;
DataOutFaces<dim> data_out_faces;
FE_Q<dim>         dummy_fe(1);
 
DoFHandler<dim>   dummy_dof_handler(triangulation);
dummy_dof_handler.distribute_dofs(dummy_fe);
 
Vector<double> dummy_solution (dummy_dof_handler.n_dofs());
 
data_out_faces.attach_dof_handler(dummy_dof_handler);
data_out_faces.add_data_vector(dummy_solution, boundary_ids);
data_out_faces.build_patches();
 
 const std::string filename_faces = ("boundary_ids"    + ref_p_array  +"."+
                                  Utilities::int_to_string(
                                    triangulation.locally_owned_subdomain(),4));
std::ofstream out((folder_name + filename_faces + ".vtu").c_str());
data_out_faces.write_vtu(out);

if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 )
  {
    std::vector<std::string>    filenames;
    for (unsigned int i=0;
          i < Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
          i++)
      {
        filenames.push_back("boundary_ids"  + ref_p_array  +"."+
                            Utilities::int_to_string(i,4) +
                            ".vtu");
      }
    std::ofstream master_output(folder_name + "boundary_ids.pvtu");
    data_out_faces.write_pvtu_record(master_output, filenames);
  }

*/


}
template <int dim, int dim_omega>
  unsigned int LDGPoissonProblem<dim, dim_omega>::cell_weight(
    const typename parallel::distributed::Triangulation<dim>::cell_iterator //parallel::distributed::
                    &cell, const typename parallel::distributed::Triangulation<dim>::CellStatus status) const
  {
    const std::vector<Point<dim>> &vertices = triangulation.get_vertices();
  for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
  {
      // If the cell has the vertex with the given index
      if (bbox.point_inside(vertices[cell->vertex_index(v)]))
      {
        std::cout<<"hall "<<vertices[cell->vertex_index(v)]<<std::endl;
          return 10;
      }
  }
  return 1;
}
template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::memory_consumption(std::string _name) {

  Utilities::System::MemoryStats mem_stats;
  Utilities::System::get_memory_stats(mem_stats);
   
 struct rusage usage;
 getrusage(RUSAGE_SELF, &usage);
 double peak_memory = usage.ru_maxrss / 1024.0;
 double max_memory;
 MPI_Reduce(&peak_memory, &max_memory, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

 size_t memoryUsed = getCurrentRSS()/ (1024.0* 1024.0);      
 pcout<< "---------------------------------------------------" <<std::endl
 << "| " <<_name<< " Rank " << rank_mpi<<std::endl
  <<"| VmPeak: " << mem_stats.VmPeak / 1024.0 << " MB" <<", "
  << "VmSize: " << mem_stats.VmSize / 1024.0 << " MB" <<", "
  << "VmHWM: " << mem_stats.VmHWM / 1024.0 << " MB" <<", "
  << "VmRSS: " << mem_stats.VmRSS / 1024.0 << " MB" << std::endl
  << "| Peak Memory Usage: " << peak_memory << " MB" << ", "
  << "Peak Memory Usage Across All Ranks: " << max_memory << " MB" << std::endl
  << "memoryUsed: " << memoryUsed<< " MB" << std::endl
  << "-------------------------------------------------------------" <<std::endl;

}
template <int dim, int dim_omega>
std::array<double, 4> LDGPoissonProblem<dim, dim_omega>::run() {
  pcout << "******************* REFINE " << n_refine << "  DEGREE  " << degree << " ***********************" <<std::endl;
  dimension_gap = dim - dim_omega;
  pcout << "geometric configuration "<<geo_conf <<"<< dim_Omega: "<< dim <<", dim_omega: "<<dim_omega<< " -> dimension_gap "<<dimension_gap<<std::endl; 
rank_mpi = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  penalty =penalty_sigma ;
  pcout<<"penalty "<<penalty<<std::endl;
 // memory_consumption("start");
  make_grid();
 // memory_consumption("after  make_grid");
 make_dofs();
 //memory_consumption("after make_dofs()");
  assemble_system();
  //memory_consumption("after  assemble_system()");

  marked_vertices.clear();


   malloc_trim(0);
  solve();
 // memory_consumption("after solve()");






  std::array<double, 4> results_array = compute_errors();
  output_results();
  //std::array<double, 4> results_array;
  return results_array;
}


int main(int argc, char *argv[]) {
  //std::cout << "USE_MPI_ASSEMBLE " << USE_MPI_ASSEMBLE << std::endl;
  std::cout << "Using Deal.II version: " 
          << DEAL_II_PACKAGE_VERSION << std::endl;
#if 1
  deallog.depth_console(0);

  Utilities::MPI::MPI_InitFinalize mpi_initialization(
      argc, argv,
      numbers::invalid_unsigned_int); //

  int num_processes = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  // Get the rank_mpi of the process
  int rank_mpi = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

  // Print the number of processes and the rank_mpi of the current process
  if (rank_mpi == 0) {
    std::cout << "Number of MPI processes: " << num_processes << std::endl;
  }

  //std::cout << "This is MPI process " << rank_mpi << std::endl;

#endif

  /*
  Parameters parameters;
      parameters.radius = 0.01;
      parameters.lumpedAverage = true;
  
  LDGPoissonProblem<dimension_Omega, 1> LDGPoissonCoupled_s(0,3, parameters);
    std::array<double, 4> arr = LDGPoissonCoupled_s.run();
    std::cout << rank_mpi << " Result_ende: U " << arr[0] << " Q " << arr[1] << " u "
              << arr[2] << " q " << arr[3] << std::endl;
    return 0;
  */
  //std::cout << "dimension_Omega " << dimension_Omega << std::endl;


  std::vector<std::array<double, 4>> result_scenario;
  std::vector<std::string> scenario_names;


  for (unsigned int rad = 0; rad < n_r; rad++) {
    for (unsigned int LA = 0; LA < n_LA; LA++) {

      std::string LA_string = lumpedAverages[LA] ? "true" : "false";
      std::string radius_string = std::to_string(radii[rad]);
      std::string D_string = std::to_string(D);
      std::string omega_on_face_string = is_omega_on_face ? "true" : "false";
      std::string coupled_string = COUPLED==1 ? "true" : "false";
      std::string gradedMesh_string = GRADEDMESH ==1 ? "true" : "false";
      std::string paperSolution_string = PAPER_SOLUTION ==1 ? "true" : "false";
      std::string vessel_string = VESSEL ==1 ? "true" : "false";
      std::string solution_linear_string = SOLUTION1_LINEAR ==1 ? "true" : "false";

      std::string name =  "_finalResults_cons_sol_" + std::to_string(constructed_solution) + "_geoconfig_" + std::to_string(geo_conf) + 
      "_gradedMesh_" + gradedMesh_string + "_coupled_" + coupled_string + "_paper_solution_" + paperSolution_string +"_solution_linear_" + solution_linear_string +
       "_vessel_" + vessel_string +  "_omegaonface_" + omega_on_face_string +  "_LA_" + LA_string + 
       "_rad_" + radius_string + "_D_" + D_string + "_penalty_" + std::to_string(penalty_sigma);
      
      std::string folderName =name +"/";
     std::cout<<folderName<<std::endl;
      std::string command = "mkdir -p " + folderName;
      if (system(command.c_str()) == 0) {
        if(rank_mpi == 0)
        std::cout << "Folder created successfully." << std::endl;
    } else {
        std::cerr << "Error: Could not create folder." << std::endl;
    }

      scenario_names.push_back(name);

      

      Parameters parameters;


      parameters.radius = radii[rad];
      parameters.lumpedAverage = lumpedAverages[LA];
       parameters.folder_name = folderName;
     // const unsigned int p_degree[2] = {0,1};
      
      constexpr unsigned int p_degree_size =
          sizeof(p_degree) / sizeof(p_degree[0]);
 //   const unsigned int refinement[3] = {3,4,5};
    
      constexpr unsigned int refinement_size =
          sizeof(refinement) / sizeof(refinement[0]);

      std::array<double, 4> results[p_degree_size][refinement_size];
      double max_diameter[p_degree_size][refinement_size];
      double max_diameter_omega[p_degree_size][refinement_size];
      double nof_cells[p_degree_size][refinement_size];
      double nof_cells_omega[p_degree_size][refinement_size];

      std::vector<std::string> solution_names = {"U_Omega", "Q_Omega",
                                                 "u_omega", "q_omega"};
                                         
      for (unsigned int r = 0; r < refinement_size; r++) {
        for (unsigned int p = 0; p < p_degree_size; p++) {
           malloc_trim(0);
          LDGPoissonProblem<dimension_Omega, 1> LDGPoissonCoupled =
              LDGPoissonProblem<dimension_Omega, 1>(p_degree[p], refinement[r],
                                                    parameters);
          std::array<double, 4> arr;
          bool is_not_failed =true;
          try
          {
            arr = LDGPoissonCoupled.run();
            is_not_failed = true;
          }
          catch(const std::exception& e)
          {
           std::cout  << e.what() << std::endl;
           arr = {42,42,42,42};
           is_not_failed = false;
          }
          
/* if(rank_mpi == 0)
 {
                  struct rusage usage;
          getrusage(RUSAGE_SELF, &usage);
          double peak_memory = usage.ru_maxrss / 1024.0;
          double max_memory;
          MPI_Reduce(&peak_memory, &max_memory, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
           size_t memoryUsed = getCurrentRSS()/ (1024.0* 1024.0);
      std::cout<< "---------------------------------------------------" <<std::endl
        << "| Peak Memory Usage: " << peak_memory << " MB" << ", "
        << "Peak Memory Usage Across All Ranks: " << max_memory << " MB" << std::endl
          << "Memory used by the process: " << memoryUsed << " MB" << std::endl
        << "-------------------------------------------------------------" <<std::endl;

          std::cout << rank_mpi << " Result_ende: U " << arr[0] << " Q " << arr[1]
                    << " u " << arr[2] << " q " << arr[3] << std::endl;
}*/
           std::cout << rank_mpi << " Result_ende: U " << arr[0] << " Q " << arr[1]
                    << " u " << arr[2] << " q " << arr[3] << std::endl;

          results[p][r] = arr;
          max_diameter[p][r] = LDGPoissonCoupled.max_diameter;
          nof_cells[p][r] = LDGPoissonCoupled.nof_cells;
          max_diameter_omega[p][r] = LDGPoissonCoupled.max_diameter_omega;
          nof_cells_omega[p][r] = LDGPoissonCoupled.nof_cells_omega;
        

         if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && is_not_failed) {
            
          std::ofstream csvfile_unsrtd;
	  std::string filename = "cvg_res_unsrtd" + name + "_r_" + std::to_string(refinement[r]) + "_p_" + std::to_string(p_degree[p]);
	  csvfile_unsrtd.open(folderName + filename + ".csv");
            
            csvfile_unsrtd<<name<<";r "<<refinement[r]<<";h " <<max_diameter[r]<<";#c "<<nof_cells[r]<< ";p "<<p_degree[p]<< ";U " << arr[0] << ";Q " << arr[1]
                    << ";u " << arr[2] << ";q " << arr[3] << "; \n";
            csvfile_unsrtd.close();
            std::cout<<"file written"<<std::endl;
          }
        
        }
      }

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
        
        // std::cout << "--------" << std::endl;
        std::ofstream myfile;
        std::ofstream csvfile;
#if COUPLED
        std::string filename = "convergence_results_test_coupled" + name;
        myfile.open(folderName + filename + ".txt");
        csvfile.open(folderName + filename  + ".csv");
#else
        std::string filename = "convergence_results_uncoupled" + name;
        myfile.open(folderName + filename  + ".txt");
        csvfile.open(folderName + filename + ".csv");
#endif
        for (unsigned int f = 0; f < solution_names.size(); f++) {
          myfile << solution_names[f] << "\n";
          csvfile << solution_names[f] << "\n";
          std::cout <<solution_names[f] << "\n";

          myfile <<"refinement;";
          csvfile <<"refinement;";
           std::cout <<"refinement;";

          for (unsigned int p = 0; p < p_degree_size; p++) {
            myfile <<"error p="<< p_degree[p] << ";"<<  "diameter h;"<< "#cells;"<<"error;"<<"convergence_rate;";
            csvfile <<"error p="<< p_degree[p] << ";"<< "diameter h;"<< "#cells;"<<"error;"<<"convergence_rate;";
            std::cout <<"error p="<< p_degree[p] << ";"<< "diameter h;"<< "#cells;"<<"error;"<<"convergence_rate;";
          }
          myfile << "\n";
          csvfile << "\n";
          std::cout << "\n";
          for (unsigned int r = 0; r < refinement_size; r++) {    
              myfile << refinement[r] << ";";
              csvfile << refinement[r] << ";";
              std::cout << refinement[r] <<";";        
            for (unsigned int p = 0; p < p_degree_size; p++) {
              const double error = results[p][r][f];

              if(f < 2 )
              {
              myfile  << max_diameter[p][r] << ";" <<nof_cells[p][r]<< ";";
              csvfile << max_diameter[p][r]  << ";" <<nof_cells[p][r]<< ";";
              std::cout << max_diameter[p][r] << ";" <<nof_cells[p][r]<< ";";
              }
              else
              {
              myfile  << max_diameter_omega[p][r] << ";" <<nof_cells_omega[p][r]<< ";";
              csvfile << max_diameter_omega[p][r]  << ";" <<nof_cells_omega[p][r]<< ";";
              std::cout << max_diameter_omega[p][r] << ";" <<nof_cells_omega[p][r]<< ";";
              }

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
        command = "cp -r " + folderName + " /mnt/c/Users/maxro/Downloads/";
        std::cout << command << std::endl;
        if (system(command.c_str()) == 0) {
          if(rank_mpi == 0)
          std::cout << "Folder copy successfully." << std::endl;
      } else {
          std::cerr << "Error: Could not copy folder." << std::endl;
      }
      }
    }
  }


  return 0;
}



