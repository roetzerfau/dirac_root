
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

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "Functions.cc"

using namespace dealii;
#define USE_MPI_ASSEMBLE 1
#define BLOCKS 1
#define SOLVE_BLOCKWISE 1
#define FASTER 1 //nur verfügbar bei der aktuellsten dealii version
#define CYLINDER 0
#define A11SCHUR 0


//Geometrie
//case 1: 2D/0Dv-> im hintergrund iwrd trotzdem noch 1D problem gelöst
//case 2: 2D/1D 
//case 3: 3D/1D



constexpr bool no_gradient = false;

const FEValuesExtractors::Vector VectorField_omega(0);
const FEValuesExtractors::Scalar Potential_omega(1);

const FEValuesExtractors::Vector VectorField(0);
const FEValuesExtractors::Scalar Potential(dimension_Omega);


const double extent = 1;
const double half_length = std::sqrt(0.5);//0.5
const double distance_tolerance = 10;
const unsigned int N_quad_points = 3;
const double reduction = 1e-8;
const double tolerance = 1e-8;

struct Parameters {
  double radius;
  bool lumpedAverage;
};

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
    ReductionControl solver_control(src.size(), tolerance * src.l2_norm(), reduction);
    TrilinosWrappers::SolverDirect solver(solver_control);
    solver.initialize(*matrix);
    solver.solve(dst,src);
    }
    else{
TrilinosWrappers::PreconditionILU preconditioner;
  TrilinosWrappers::PreconditionILU::AdditionalData data;
  preconditioner.initialize(*matrix, data);

    ReductionControl solver_control(matrix->local_size(), tolerance * src.l2_norm(), reduction);//, 1e-7 * src.l2_norm());
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
       IndexSet set1,set2;
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
      , tmp1(block_vector.block(1))
      , tmp2(block_vector.block(1))
      , tmp3(block_vector.block(0))
      , tmp4(block_vector.block(0))

      {
      }

      void vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const
      {
   


      system_matrix->block(1,0).vmult(tmp1, src);
    
      A_inverse->vmult(tmp2, tmp1);
      system_matrix->block(0,1).vmult(tmp3, tmp2);
     
       system_matrix->block(0, 0).vmult(tmp4, src);
       dst = tmp3- tmp4;
  
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
                           const FEValuesExtractors::Scalar &Potential,
                           bool no_gradient);
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
  void dof_omega_local_2_global(
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
  
  //unsigned int nof_degrees;
  unsigned int dimension_gap;


  int rank_mpi;
  enum { Dirichlet, Neumann };

  // parameters
  double radius;
  double g;
  bool lumpedAverage;


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

  const RightHandSide<dim> rhs_function;
  const KInverse<dim> K_inverse_function;
  const DirichletBoundaryValues<dim> Dirichlet_bc_function;
  const NeumannBoundaryValues<dim> Neumann_bc_function;
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
  const UpdateFlags update_flags_coupling = update_values;

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
  pcout<<"make grid"<<std::endl;
  Point<dim> corner1, corner2;
double margin = 1.0;
double h = 0;
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
      Point<dim>(2 * half_length, -half_length + offset, -half_length + offset);
   p2 =
      Point<dim>(0, half_length + offset, half_length + offset);
}
 if (dim == 2) {
   p1 =
      Point<dim>(-half_length + offset, -half_length + offset);	
   p2 =
      Point<dim>(half_length + offset, half_length + offset);

 }
 GridGenerator::hyper_rectangle(triangulation, p1, p2);
#endif
pcout<<"refined++++++"<<std::endl;
triangulation.refine_global(n_refine);
pcout<<"refined"<<std::endl;

GridOut grid_out;
std::ofstream out("grid_Omega.vtk"); // Choose your preferred filename and format
grid_out.write_vtk(triangulation, out);

const std::vector<Point<dim>> &vertices = triangulation.get_vertices();

SparsityPattern cell_connection_graph;
DynamicSparsityPattern connectivity;
std::vector<unsigned int> cells_inside_box;

int num_processes = dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
bool is_shared_triangulation = num_processes > 1 ? true : false;
bool is_repartioned =  is_shared_triangulation && (geo_conf == GeometryConfiguration::TwoD_OneD ||geo_conf == GeometryConfiguration::ThreeD_OneD);
pcout<<"is_shared_triangulation "<<  is_shared_triangulation<<" is_repartioned "<<is_repartioned<<std::endl;
if(is_repartioned)
GridTools::get_face_connectivity_of_cells(triangulation,connectivity);

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
      //std::cout<<"p "<<vertices[cell->vertex_index(v)]<<std::endl;
    if (bbox.point_inside(vertices[cell->vertex_index(v)]))
    {
     cell_is_inside_box = true;
    // cell_start = cell;
    }
    }
    if(cell_is_inside_box)
      cells_inside_box.push_back(cell_number);
   

}

    if (cell->is_locally_owned())
    {
  
    double cell_diameter = cell->diameter(); 
    
    if (cell_diameter > max_diameter) {
      max_diameter = cell_diameter;
    }

    for (unsigned int face_no = 0; face_no < GeometryInfo<dim>::faces_per_cell;
         face_no++) {
      Point<dim> p = cell->face(face_no)->center();
      if (cell->face(face_no)->at_boundary()) {
       
        if(((p[0] == 0 || p[0] ==2 * half_length) && geo_conf == GeometryConfiguration::ThreeD_OneD && constructed_solution == 3) ){
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
if(is_repartioned)
{
  for(unsigned int row : cells_inside_box)
    connectivity.add_entries(row, cells_inside_box.begin(), cells_inside_box.end());
  cell_connection_graph.copy_from(connectivity);
  pcout<<"los "<< dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)<<std::endl;
 GridTools::partition_triangulation(dealii::Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),cell_connection_graph,triangulation, SparsityTools::Partitioner::zoltan );
}  
  




 pcout << " Memory consumption of triangulation: "
               << triangulation.memory_consumption() / (1024.0 * 1024.0 * 1024.0) // Convert to MB
	              << " GB" << std::endl;
		         
			     unsigned int global_active_cells = triangulation.n_global_active_cells();
			       
				     pcout << "Total number of active cells (global): " << global_active_cells << std::endl;

				         pcout<<"Memory DofHandler "<< dof_handler_Omega.memory_consumption()/ (1024.0 * 1024.0 * 1024.0)<<std::endl;







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

  triangulation_omega.refine_global(n_refine);

 typename DoFHandler<dim_omega>::active_cell_iterator
        cell_omega = dof_handler_omega.begin_active(),
        endc_omega = dof_handler_omega.end();
  for (; cell_omega != endc_omega; ++cell_omega) {
    if (cell_omega->is_locally_owned())
    for (unsigned int face_no = 0;
         face_no < GeometryInfo<dim_omega>::faces_per_cell; face_no++) {
      if (cell_omega->face(face_no)->at_boundary())
        cell_omega->face(face_no)->set_boundary_id(Dirichlet);
    }
  }

GridOut grid_out_omega;
std::ofstream out_omega("grid_omega.vtk"); // Choose your preferred filename and format
grid_out_omega.write_vtk(triangulation_omega, out_omega);


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
     marked_vertices[i] = true;
}
#endif

}

template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::make_dofs() {
  TimerOutput::Scope t(computing_timer, "setup");

  dof_handler_Omega.distribute_dofs(fe_Omega);
  const unsigned int dofs_per_cell = fe_Omega.dofs_per_cell;
  pcout << "dofs_per_cell " << dofs_per_cell << std::endl;
 // DoFRenumbering::component_wise(dof_handler_Omega); //uncomment for unput result

  dof_handler_omega.distribute_dofs(fe_omega);
  const unsigned int dofs_per_cell_omega = fe_omega.dofs_per_cell;
  pcout << "dofs_per_cell_omega " << dofs_per_cell_omega << std::endl;
  //DoFRenumbering::component_wise(dof_handler_omega); //TODO nochmal kontrollieren


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
        << n_vector_field_Omega << " + " << n_potential_Omega << ")"<<std::endl;
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
  locally_relevant_dofs_Omega;
  DoFTools::extract_locally_relevant_dofs(dof_handler_Omega, locally_relevant_dofs_Omega);

  locally_owned_dofs_omega_local = dof_handler_omega.locally_owned_dofs();
  // if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0 )
   //locally_owned_dofs_omega_local.clear();
   
  /*
  locally_owned_dofs_omega_global.set_size(locally_owned_dofs_omega_local.size());
  locally_owned_dofs_omega_global.add_indices(locally_owned_dofs_omega_local,  dof_handler_Omega.n_dofs());
*/
  locally_relevant_dofs_omega_local;
  DoFTools::extract_locally_relevant_dofs(dof_handler_omega, locally_relevant_dofs_omega_local);
  /*
  locally_relevant_dofs_omega_global.set_size(locally_relevant_dofs_omega_local.size());
  locally_relevant_dofs_omega_global.add_indices(locally_relevant_dofs_omega_local,  dof_handler_Omega.n_dofs());
*/


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
#if COUPLED
  {
    // coupling

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
    unsigned int nof_quad_points;
    bool AVERAGE = radius != 0 && !lumpedAverage;
    pcout << "AVERAGE (use circel) " << AVERAGE << " radius "<<radius << " lumpedAverage "<<lumpedAverage<<std::endl;
    // weight
    if (AVERAGE) {
      nof_quad_points = N_quad_points;
    } else {
      nof_quad_points = 1;
    }
    pcout<<"nof_quad_points "<<nof_quad_points<<std::endl;
    typename DoFHandler<dim_omega>::active_cell_iterator
        cell_omega = dof_handler_omega.begin_active(),
        endc_omega = dof_handler_omega.end();

    for (; cell_omega != endc_omega; ++cell_omega) {
      //if (cell_omega->is_locally_owned())
      {
      fe_values_omega.reinit(cell_omega);
      cell_omega->get_dof_indices(local_dof_indices_omega);
      dof_omega_local_2_global(dof_handler_omega, local_dof_indices_omega);

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
       // pcout<<"quadrature_point_test "<<quadrature_point_test<<std::endl;
#if TEST
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

#else
        auto cell_test = GridTools::find_active_cell_around_point(
            dof_handler_Omega, quadrature_point_test);
#endif

        {
#if TEST
          auto cell_test_tri = cellpair.first;
         typename DoFHandler<dim>::active_cell_iterator
        cell_test = dof_handler_Omega.begin_active();
        std::advance(cell_test, cell_test_tri->index());
        cell_start =cell_test;  
#endif

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
              cell_trial = dof_handler_Omega.begin_active();
              std::advance(cell_trial, cell_trial_tri->index());
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
                  //throw std::runtime_error("cell coupling error");
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
#endif
  pcout<<"start to compress"<<std::endl;
  sp_block.compress();
  

   pcout<<"Sparsity "  <<sp_block.n_rows()<<" "<<sp_block.n_cols()<<" n_nonzero_elements " <<sp_block.n_nonzero_elements()<<std::endl;
  // pcout<<"sparsity memory "<<sp_block.memory_consumption()<<std::endl;
   pcout<<"start reinit"<<std::endl;
  system_matrix.reinit(sp_block);
  pcout<<"system_matrix.reinit"<<std::endl;
  solution.reinit(locally_relevant_dofs_block,  MPI_COMM_WORLD);
   pcout<<"solution.reinit"<<std::endl;
  system_rhs.reinit(locally_owned_dofs_block, locally_relevant_dofs_block,  MPI_COMM_WORLD, true);
   pcout<<"system_rhs.reinit"<<std::endl;

   std::cout<<rank_mpi<<" memory system_matrix "<<system_matrix.memory_consumption()/ (1024.0 * 1024.0 * 1024.0)<<" memory system_rhs "<<system_rhs.memory_consumption()/ (1024.0 * 1024.0 * 1024.0)<<std::endl;
  pcout<<"Ende setup dof"<<std::endl;
}

template <int dim, int dim_omega>
void LDGPoissonProblem<dim, dim_omega>::assemble_system() {
  TimerOutput::Scope t(computing_timer, "assembly");
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
                            Potential, no_gradient);

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
              //assemble_Neumann_boundary_terms(fe_face_values, local_matrix,
                //                              local_vector,
                  //                            Neumann_bc_function);
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
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);

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
                          Potential_omega, no_gradient);

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
           dof_omega_local_2_global(dof_handler_omega,
                              local_neighbor_dof_indices_omega);

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
    std::cout << "2D/0D" << std::endl;
    Point<dim> quadrature_point_test(y_l, z_l);
    std::vector<types::global_dof_index> local_dof_indices_test(dofs_per_cell);
    // test function
    std::vector<double> my_quadrature_weights = {1};
    unsigned int n_te;
#if TEST
    auto cell_test_array = GridTools::find_all_active_cells_around_point(
        mapping, dof_handler_Omega, quadrature_point_test, 1e-10, marked_vertices);
    n_te = cell_test_array.size();
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


      if (cell_test != dof_handler_Omega.end())
        if (cell_test->is_locally_owned())
        {
          
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

            for (unsigned int i = 0; i < dofs_per_cell; i++) {
            

              local_vector(i) +=
                  fe_values_coupling_test[Potential].value(i, 0); //
            }
           
            constraints.distribute_local_to_global(
                local_vector, local_dof_indices_test, system_rhs);
          }
        }
    }
  }
  if (geo_conf == GeometryConfiguration::TwoD_OneD || geo_conf == GeometryConfiguration::ThreeD_OneD) {
    std::cout<<"2D/1D  3D/1D"<<std::endl;
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
    pcout << "AVERAGE " << AVERAGE << " radius "<<radius << " lumpedAverage "<<lumpedAverage<<std::endl;

    // weight
    if (AVERAGE) {
      nof_quad_points = N_quad_points;
    } else {
      nof_quad_points = 1;
    }

    cell_omega = dof_handler_omega.begin_active();
    endc_omega = dof_handler_omega.end();

    for (; cell_omega != endc_omega; ++cell_omega) {
      //if (cell_omega->is_locally_owned())
      {
      fe_values_omega.reinit(cell_omega);
      cell_omega->get_dof_indices(local_dof_indices_omega);
      dof_omega_local_2_global(dof_handler_omega, local_dof_indices_omega);

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
       // pcout << "cell_test_array " << cell_test_array.size() << std::endl;
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
              cell_test = dof_handler_Omega.begin_active();
              std::advance(cell_test, cell_test_tri->index());
              cell_start = cell_test;
#endif

#if 1// USE_MPI_ASSEMBLE
          if (cell_test != dof_handler_Omega.end())
            if (cell_test->is_locally_owned())
#endif
            {
              
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

#if !COUPLED
            //  std::cout << "not coupled" << std::endl;
              //-------------face -----------------
           
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
                 for (unsigned int q = 0; q < n_q_points; ++q) {
                for (unsigned int i = 0; i < dofs_per_cell; i++) {
                  local_vector(i) +=
                      fe_values_coupling_test[Potential].value(i, 0) *
                      (1 + quadrature_point_omega[0]) * fe_values_omega.JxW(p);
                }
                constraints.distribute_local_to_global(
                    local_vector, local_dof_indices_test, system_rhs);
              }
              }
#endif

#if COUPLED
              // std::cout << "coupled " << std::endl;
              for (unsigned int q_avag = 0; q_avag < nof_quad_points;
                   q_avag++) {
                // Quadrature weights and points
                quadrature_point_trial = quadrature_points_circle[q_avag];
              
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
                  weight = ((2.0 * numbers::PI * radius) / (nof_quad_points));
                } else {
                  weight = 1;
                  C_avag = 1;
                }
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
              cell_trial = dof_handler_Omega.begin_active();
              std::advance(cell_trial, cell_trial_tri->index());
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

                      }

                     

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
        }
        // std::cout<<std::endl;
      }
    }
  }

}
#endif
  // std::cout << "ende coupling loop" << std::endl;


  /*for (unsigned int i = 0; i < dof_handler_Omega.n_dofs() + dof_handler_omega.n_dofs(); i++) // dof_table.size()
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
          pcout<<"no gradient" <<std::endl;
         cell_matrix(i, j) += (psi_i_field * K_inverse_values[q] * psi_j_field)  + (psi_j_potential* psi_i_potential ) * cell_fe.JxW(q);
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
  double global_potential_l2_error_omega, global_vectorfield_l2_error_omega;

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

/*
    DataOut<dim> data_out;
    data_out.attach_triangulation(triangulation);
    data_out.add_data_vector(cellwise_errors_Q, "Q");
    data_out.add_data_vector(cellwise_errors_U, "U");
    data_out.build_patches();
    std::ofstream output("error.vtk");
    data_out.write_vtk(output);
*/


      const ComponentSelectFunction<dim_omega> potential_mask_omega(
          dim_omega, dim_omega + 1);
      const ComponentSelectFunction<dim_omega> vectorfield_mask_omega(
          std::make_pair(0, dim_omega), dim_omega + 1);
      Vector<double> cellwise_errors_u(
          triangulation_omega.n_active_cells());
      Vector<double> cellwise_errors_q(
          triangulation_omega.n_active_cells());
    
      const QTrapezoid<1> q_trapez_omega;
      const QIterated<dim_omega> quadrature_omega(q_trapez_omega, degree + 2);
   // solution.block(1).print(std::cout);
      VectorTools::integrate_difference(
          dof_handler_omega, solution_omega, true_solution_omega,
          cellwise_errors_u, quadrature_omega, VectorTools::L2_norm,
          &potential_mask_omega);
cellwise_errors_u.print(std::cout);
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
solution = system_rhs;

        TrilinosWrappers::MPI::BlockVector completely_distributed_solution(
        system_rhs);
  completely_distributed_solution = solution;


#if SOLVE_BLOCKWISE
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
  
  
  ReductionControl solver_control1(completely_distributed_solution.block(0).locally_owned_size(), tolerance * system_rhs.l2_norm(), reduction);
  SolverGMRES<TrilinosWrappers::MPI::Vector > solver(solver_control1);

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


  ReductionControl solver_control1(completely_distributed_solution.block(0).locally_owned_size(), tolerance * system_rhs.l2_norm(), reduction);
  SolverGMRES<TrilinosWrappers::MPI::Vector > solver(solver_control1);
 

TrilinosWrappers::PreconditionILU preconditioner;
  TrilinosWrappers::PreconditionILU::AdditionalData data;
  preconditioner.initialize(system_matrix.block(0, 0), data);

  solver.solve(schur_complement, completely_distributed_solution.block(0),schur_rhs, preconditioner);
  pcout<<"Schur complete "<<std::endl;

  system_matrix.block(1, 0).vmult(tmp, completely_distributed_solution.block(0));
  tmp *= -1;
  tmp += system_rhs.block(1);
 
  A_inverse.vmult(completely_distributed_solution.block(1), tmp);

 // A_inverse.vmult(completely_distributed_solution.block(0), system_rhs.block(0));//unkoppled

#endif
#else 
pcout<<"solve full"<<std::endl;
// Preconditioners for each block
TrilinosWrappers::PreconditionILU preconditioner_block_0;
TrilinosWrappers::PreconditionILU preconditioner_block_1;//PreconditionILU  PreconditionBlockJacobi

// Initialize the preconditioners with the appropriate blocks of the matrix
preconditioner_block_0.initialize(system_matrix.block(0, 0));  // ILU for block (0,0)
preconditioner_block_1.initialize(system_matrix.block(1, 1));  // ILU for block (1,1)

// Set up solver control
ReductionControl solver_control22(dof_handler_Omega.n_locally_owned_dofs(), tolerance * system_rhs.l2_norm(), reduction);
SolverGMRES<TrilinosWrappers::MPI::BlockVector> solver(solver_control22);


// Create the block preconditioner
BlockPreconditioner block_preconditioner(preconditioner_block_0, preconditioner_block_1);

// Solve the system using the block preconditioner
solver.solve(system_matrix, completely_distributed_solution, system_rhs,block_preconditioner);
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
std::array<double, 4> LDGPoissonProblem<dim, dim_omega>::run() {
  pcout << "n_refine " << n_refine << "  degree " << degree << std::endl;
  dimension_gap = dim - dim_omega;
  pcout << "geometric configuration "<<geo_conf <<"<< dim_Omega: "<< dim <<", dim_omega: "<<dim_omega<< " -> dimension_gap "<<dimension_gap<<std::endl; 
rank_mpi = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  penalty = 5;
  make_grid();
  make_dofs();
  assemble_system();
  solve();
  output_results();
  std::array<double, 4> results_array= compute_errors();
  return results_array;
}

int main(int argc, char *argv[]) {
  //std::cout << "USE_MPI_ASSEMBLE " << USE_MPI_ASSEMBLE << std::endl;
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
  std::cout << "dimension_Omega " << dimension_Omega << std::endl;
  const unsigned int n_r = 1;
  const unsigned int n_LA = 1;
  double radii[n_r] = {  0.01};
  bool lumpedAverages[n_LA] = {true} ;//TODO bei punkt wuelle noch berücksichtnge
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
     // const unsigned int p_degree[2] = {0,1};
      const unsigned int p_degree[1] = {1};
      constexpr unsigned int p_degree_size =
          sizeof(p_degree) / sizeof(p_degree[0]);
 //   const unsigned int refinement[3] = {3,4,5};
    const unsigned int refinement[2] = {3,4};

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
          

          std::cout << rank_mpi << " Result_ende: U " << arr[0] << " Q " << arr[1]
                    << " u " << arr[2] << " q " << arr[3] << std::endl;
                
          if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
            
          std::ofstream csvfile_unsrtd;
	  std::string filename = "cvg_res_unsrtd" + name + "_r_" + std::to_string(refinement[r]) + "_p_" + std::to_string(p_degree[p]);
	  csvfile_unsrtd.open(filename + ".csv");
            
            csvfile_unsrtd<<name<<";r "<<refinement[r]<<";p "<<p_degree[p]<< ";U " << arr[0] << ";Q " << arr[1]
                    << ";u " << arr[2] << ";q " << arr[3] << "; \n";
            csvfile_unsrtd.close();
          }
          results[p][r] = arr;
          max_diameter[r] = LDGPoissonCoupled.max_diameter;
        }
      }

      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
        
        // std::cout << "--------" << std::endl;
        std::ofstream myfile;
        std::ofstream csvfile;
#if COUPLED
        std::string filename = "convergence_results_test_coupled" + name;
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




