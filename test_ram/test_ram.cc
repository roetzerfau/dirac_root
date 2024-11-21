
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

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <malloc.h>



using namespace dealii;

template <int dim, int dim_omega> class TestRam {

public:
  TestRam(const unsigned int degree, const unsigned int n_refine);

  ~TestRam();

 void run();
  double max_diameter;
private:
  void make_grid();

  void make_dofs();


  template <int _dim>
  void dof_omega_local_2_global(
      const DoFHandler<_dim> &dof_handler_Omega,
      std::vector<types::global_dof_index> &local_dof_indices_omega);

  const unsigned int degree;
  const unsigned int n_refine;

  double h_max;
  double h_min;
  
  //unsigned int nof_degrees;
  unsigned int dimension_gap;


  int rank_mpi;
  
  std::string folder_name;



  parallel::shared::Triangulation<dim> triangulation;
  GridTools::Cache<dim, dim> cache;

  BoundingBox<dim> bbox;

  TrilinosWrappers::BlockSparseMatrix system_matrix;
  TrilinosWrappers::MPI::BlockVector solution;
  TrilinosWrappers::MPI::BlockVector system_rhs;

  FESystem<dim> fe_Omega;
  DoFHandler<dim> dof_handler_Omega;

  parallel::shared::Triangulation<dim_omega> triangulation_omega;
  FESystem<dim_omega> fe_omega;
  DoFHandler<dim_omega> dof_handler_omega;


 AffineConstraints<double> constraints;

  IndexSet locally_owned_dofs_Omega;
  IndexSet locally_relevant_dofs_Omega;

  IndexSet locally_owned_dofs_omega_local;
  IndexSet locally_relevant_dofs_omega_local;


  ConditionalOStream pcout;
  TimerOutput computing_timer;

    unsigned int start_VectorField_omega;
  unsigned int start_Potential_omega;
  unsigned int start_Potential_Omega;


};

template <int dim, int dim_omega>
TestRam<dim, dim_omega>::TestRam(const unsigned int degree, const unsigned int n_refine)
    : degree(degree), n_refine(n_refine),
      triangulation(MPI_COMM_WORLD),
      cache(triangulation),
      triangulation_omega(MPI_COMM_WORLD),
      fe_Omega(FESystem<dim>(FE_DGQ<dim>(degree), dim), FE_DGQ<dim>(degree)),
      dof_handler_Omega(triangulation),
      fe_omega(FESystem<dim_omega>(FE_DGQ<dim_omega>(degree), dim_omega),
               FE_DGQ<dim_omega>(degree)),
      dof_handler_omega(triangulation_omega),
      pcout(std::cout),
      computing_timer(pcout, TimerOutput::summary, TimerOutput::wall_times) {}


template <int dim, int dim_omega>
TestRam<dim, dim_omega>::~TestRam() {
  dof_handler_Omega.clear();
  dof_handler_omega.clear();
}

template <int dim, int dim_omega>
void TestRam<dim, dim_omega>::make_grid() {
  TimerOutput::Scope t(computing_timer, "make grid");
  pcout<<"make grid"<<std::endl;

  double offset = 0.0;
  double half_length =  std::sqrt(0.5);
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
pcout<<"refined++++++"<<std::endl;
triangulation.refine_global(n_refine);
pcout<<"refined"<<std::endl;

 //pcout << "Memory consumption of triangulation: "
      //         << triangulation.memory_consumption() / (1024.0 * 1024.0 * 1024.0) // Convert to MB
	   //          << " GB" << std::endl;
		         
			     unsigned int global_active_cells = triangulation.n_global_active_cells();
			       
				     pcout << "Total number of active cells (global): " << global_active_cells << std::endl;
		

//---------------omega-------------------------
    if(dim == 2)
    GridGenerator::hyper_cube(triangulation_omega, -half_length ,  half_length);
    if(dim == 3)
    GridGenerator::hyper_cube(triangulation_omega,0 ,  2*half_length);

  triangulation_omega.refine_global(n_refine);

Utilities::System::MemoryStats mem_stats;
Utilities::System::get_memory_stats(mem_stats);
  pcout << "Memory Statistics Grid:" << std::endl
<<"VmPeak: " << mem_stats.VmPeak / 1024.0 << " MB" << std::endl 
<< "VmSize: " << mem_stats.VmSize / 1024.0 << " MB" << std::endl
<< "VmHWM: " << mem_stats.VmHWM / 1024.0 << " MB" << std::endl
<< "VmRSS: " << mem_stats.VmRSS / 1024.0 << " MB" << std::endl;
}

template <int dim, int dim_omega>
void TestRam<dim, dim_omega>::make_dofs() {
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


  //std::cout<<"Memory DofHandler "<< dof_handler_Omega.memory_consumption()/ (1024.0 * 1024.0 * 1024.0)<<" GB"<<std::endl;
  constraints.clear();
  constraints.close();
  unsigned int n_dofs_total = dof_handler_Omega.n_dofs() + dof_handler_omega.n_dofs();


  locally_owned_dofs_Omega = dof_handler_Omega.locally_owned_dofs();
  
  DoFTools::extract_locally_relevant_dofs(dof_handler_Omega, locally_relevant_dofs_Omega);
  //std::cout<<"Memory locally_owned_dofs_Omega "<< locally_owned_dofs_Omega.memory_consumption()/ (1024.0 * 1024.0 * 1024.0) // Convert to MB
	  //            << " GB" << std::endl;
  //std::cout<<"Memory locally_relevant_dofs_Omega "<< locally_relevant_dofs_Omega.memory_consumption()/ (1024.0 * 1024.0 * 1024.0) // Convert to MB
	  //            << " GB" << std::endl;
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

Utilities::System::MemoryStats mem_stats;
Utilities::System::get_memory_stats(mem_stats);
pcout << "Memory Statistics 1:" << std::endl
  <<"cpu_load "<<Utilities::System::get_cpu_load()<<std::endl
<<"VmPeak: " << mem_stats.VmPeak / 1024.0 << " MB" << std::endl 
<< "VmSize: " << mem_stats.VmSize / 1024.0 << " MB" << std::endl
<< "VmHWM: " << mem_stats.VmHWM / 1024.0 << " MB" << std::endl
<< "VmRSS: " << mem_stats.VmRSS / 1024.0 << " MB" << std::endl;


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
  DoFTools::make_flux_sparsity_pattern(dof_handler_Omega, sp_block.block(0,0),constraints,false,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));//,  constraints,false,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD));
 // std::cout<<"sparsity memory flx block(0, 0)"<<sp_block.memory_consumption()/ (1024.0 * 1024.0 * 1024.0) // Convert to MB
	//              << " GB" << std::endl;
  DoFTools::make_flux_sparsity_pattern(dof_handler_omega, sp_block.block(1,1),constraints,false,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) );
 // std::cout <<"sparsity memory flx block(1, 1)"<<sp_block.memory_consumption()/ (1024.0 * 1024.0 * 1024.0) // Convert to MB
	//              << " GB" << std::endl;
  sp_block.collect_sizes();
  //malloc_trim(0);  // Force memory release

  pcout<<"start to compress"<<std::endl;

 /*   Utilities::System::get_memory_stats(mem_stats);
  std::cout << "Memory Statistics before compress sparse:" << std::endl
  <<"cpu_load "<<Utilities::System::get_cpu_load()<<std::endl
<<"VmPeak: " << mem_stats.VmPeak / 1024.0 << " MB" << std::endl 
<< "VmSize: " << mem_stats.VmSize / 1024.0 << " MB" << std::endl
<< "VmHWM: " << mem_stats.VmHWM / 1024.0 << " MB" << std::endl
<< "VmRSS: " << mem_stats.VmRSS / 1024.0 << " MB" << std::endl;*/
  sp_block.compress();

 /* Utilities::System::get_memory_stats(mem_stats);
    std::cout << "Memory Statistics after compress sparse:" << std::endl
  <<"cpu_load "<<Utilities::System::get_cpu_load()<<std::endl
<<"VmPeak: " << mem_stats.VmPeak / 1024.0 << " MB" << std::endl 
<< "VmSize: " << mem_stats.VmSize / 1024.0 << " MB" << std::endl
<< "VmHWM: " << mem_stats.VmHWM / 1024.0 << " MB" << std::endl
<< "VmRSS: " << mem_stats.VmRSS / 1024.0 << " MB" << std::endl;*/


   pcout<<"Sparsity "  <<sp_block.n_rows()<<" "<<sp_block.n_cols()<<" n_nonzero_elements " <<sp_block.n_nonzero_elements()<<std::endl;
  // pcout<<"sparsity memory "<<sp_block.memory_consumption()<<std::endl;
   pcout<<"start reinit"<<std::endl;
  system_matrix.reinit(sp_block);
  pcout<<"system_matrix.reinit"<<std::endl;
  solution.reinit(locally_relevant_dofs_block,  MPI_COMM_WORLD);
   pcout<<"solution.reinit"<<std::endl;
  system_rhs.reinit(locally_owned_dofs_block, locally_relevant_dofs_block,  MPI_COMM_WORLD, true);
   pcout<<"system_rhs.reinit"<<std::endl;

 //  std::cout<<rank_mpi<<" memory system_matrix "<<system_matrix.memory_consumption()/ (1024.0 * 1024.0 * 1024.0)<<" memory system_rhs "<<system_rhs.memory_consumption()/ (1024.0 * 1024.0 * 1024.0)<<std::endl;
  pcout<<"Ende setup dof"<<std::endl;

      Utilities::System::get_memory_stats(mem_stats);
  pcout << "Memory Statistics Ende setup dof:" << std::endl
  <<"cpu_load "<<Utilities::System::get_cpu_load()<<std::endl
<<"VmPeak: " << mem_stats.VmPeak / 1024.0 << " MB" << std::endl 
<< "VmSize: " << mem_stats.VmSize / 1024.0 << " MB" << std::endl
<< "VmHWM: " << mem_stats.VmHWM / 1024.0 << " MB" << std::endl
<< "VmRSS: " << mem_stats.VmRSS / 1024.0 << " MB" << std::endl;
//std::cout<<"malloc_trim "<<malloc_trim(0)<<std::endl;
}

template <int dim, int dim_omega>
void TestRam<dim, dim_omega>::run() {
  pcout << "n_refine " << n_refine << "  degree " << degree << std::endl;
  dimension_gap = dim - dim_omega;
  pcout <<"dim_Omega: "<< dim <<", dim_omega: "<<dim_omega<< " -> dimension_gap "<<dimension_gap<<std::endl; 
rank_mpi = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  
  make_grid();
  make_dofs();
}

int main(int argc, char *argv[]) {
  //std::cout << "USE_MPI_ASSEMBLE " << USE_MPI_ASSEMBLE << std::endl;

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


/*Utilities::System::MemoryStats mem_stats;
Utilities::System::get_memory_stats(mem_stats);
  std::cout << "Memory Statistics Begin:" << std::endl
  <<"cpu_load "<<Utilities::System::get_cpu_load()<<std::endl
<<"VmPeak: " << mem_stats.VmPeak / 1024.0 << " MB" << std::endl 
<< "VmSize: " << mem_stats.VmSize / 1024.0 << " MB" << std::endl
<< "VmHWM: " << mem_stats.VmHWM / 1024.0 << " MB" << std::endl
<< "VmRSS: " << mem_stats.VmRSS / 1024.0 << " MB" << std::endl;
  */

  
  TestRam<3, 1> myTestRam(1,7);
    myTestRam.run();
   
  

  //std::cout << "dimension_Omega " << dimension_Omega << std::endl;


  
 /* Utilities::System::get_memory_stats(mem_stats);
  std::cout << "Memory Statistics End:" << std::endl
  <<"cpu_load "<<Utilities::System::get_cpu_load()<<std::endl
<<"VmPeak: " << mem_stats.VmPeak / 1024.0 << " MB" << std::endl 
<< "VmSize: " << mem_stats.VmSize / 1024.0 << " MB" << std::endl
<< "VmHWM: " << mem_stats.VmHWM / 1024.0 << " MB" << std::endl
<< "VmRSS: " << mem_stats.VmRSS / 1024.0 << " MB" << std::endl;*/
  return 0;
}




