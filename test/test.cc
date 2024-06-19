/*#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/numerics/vector_tools.h>
//#include <deal.II/fe/mapping_fe.h>

#include <iostream>
#include <fstream>

using namespace dealii;

template <int dim>
void map_dofs_to_support_points(const DoFHandler<dim> &dof_handler,
                                std::vector<Point<dim>> &support_points)
{
    // Step 1: Get the finite element associated with the DoFHandler
    const FiniteElement<dim> &fe = dof_handler.get_fe();
    
    // Step 2: Check if the finite element has support points
    if (fe.n_support_points() == 0)
    {
        std::cerr << "The finite element does not have support points." << std::endl;
        return;
    }

    // Step 3: Create a vector of Point<dim> to store the support points
    support_points.resize(dof_handler.n_dofs());

    // Step 4: Get the mapping from DoFs to support points
    std::vector<Point<dim>> unit_support_points(fe.n_support_points());
    fe.get_unit_support_points(unit_support_points);

    // Step 5: Create a vector to hold the support points for one cell
    std::vector<Point<dim>> cell_support_points(fe.n_dofs_per_cell());

    // Step 6: Iterate over all active cells
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
    for (; cell != endc; ++cell)
    {
        cell->get_dof_indices(local_dof_indices);

        // Map the unit support points to real support points in the cell
        MappingQ1<dim> mapping;
        mapping.transform_real_to_unit_cell(cell, unit_support_points, cell_support_points);

        // Assign the cell support points to the global support points
        for (unsigned int i = 0; i < local_dof_indices.size(); ++i)
        {
            support_points[local_dof_indices[i]] = cell_support_points[i];
        }
    }
}

int main()
{
    const unsigned int dim = 2;
    Triangulation<dim> triangulation;
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(2);

    FE_Q<dim> fe(1);
    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    std::vector<Point<dim>> support_points;
    map_dofs_to_support_points(dof_handler, support_points);

    // Output the support points
    for (unsigned int i = 0; i < support_points.size(); ++i)
    {
        std::cout << "DoF " << i << ": " << support_points[i] << std::endl;
    }

    return 0;
}
*/

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


#include <iostream>
#include <vector>

using namespace dealii;

template <int dim>
void map_dofs_to_support_points(const Mapping<dim> &mapping,
                                const DoFHandler<dim> &dof_handler,
                                std::vector<Point<dim>> &support_points)
{

    QGauss<dim> quadrature_formula(0 + 2);
    FEValues<dim> fe_values(dof_handler.get_fe(), quadrature_formula,
                                update_values | update_gradients |
                                    update_quadrature_points |
                                    update_JxW_values);


    //const FiniteElement<dim> &fe = dof_handler.get_fe();
   const FESystem<dim> &fe = dof_handler.get_fe();
   std::cout<<dof_handler.get_fe().base_element(0).get_name()<<std::endl;
    std::cout<<dof_handler.get_fe().base_element(1).get_name()<<std::endl;

    
    const FEValuesExtractors::Vector eins(0);
    const FEValuesExtractors::Scalar zwei(2);
    ComponentMask mask = dof_handler.get_fe().component_mask (eins);
    const FiniteElement<dim> &fe__= dof_handler.get_fe().get_sub_fe(mask);

     //const unsigned int component_i  = fe.system_to_component_index(i).first;
      
   // const auto &fe = dof_handler.get_fe_collection();
    //Assert(fe.has_support_points(), ExcNotImplemented());

    // Get the unit support points from the finite element
    std::vector<Point<dim>> unit_support_points_general(fe.n_dofs_per_cell());
    std::vector<Point<dim>> unit_support_points_FE_Q(fe.n_dofs_per_cell());
    std::vector<Point<dim>> unit_support_points_RT(fe.n_dofs_per_cell());

    unit_support_points_RT =  dof_handler.get_fe().base_element(0).get_generalized_support_points();
    unit_support_points_FE_Q =  dof_handler.get_fe().base_element(1).get_unit_support_points();
    unit_support_points_general =  dof_handler.get_fe().get_generalized_support_points();
    // Resize the support points vector to hold all DoF support points
    support_points.resize(dof_handler.n_dofs());
    std::cout<<"dof_handler.n_dofs() "<<dof_handler.n_dofs()<<std::endl;
    // Loop over all active cells and get the support points
    std::vector<Point<dim>> cell_support_points(fe.n_dofs_per_cell());
    std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());

   // for (const auto i : fe_values.dof_indices())
   /*for(unsigned int i = 0; i < fe.n_dofs_per_cell(); i++)
   {

    const unsigned int base =
  dof_handler.get_fe().system_to_base_index(i).first.first;
const unsigned int multiplicity =
  dof_handler.get_fe().system_to_base_index(i).first.second;
const unsigned int within_base_  =
  dof_handler.get_fe().system_to_base_index(i).second; // same as above
    std::cout<<i <<" "<<base <<" "<< multiplicity<<" "<< within_base_ ;
     std::cout<<std::endl;
   }
   */



   /* const unsigned int component =
  dof_handler.get_fe().system_to_component_index(i).first;
const unsigned int within_base =
  dof_handler.get_fe().system_to_component_index(i).second;
*/


/*
const unsigned int component =
  fe_basis.system_to_component_index(i).first;
const unsigned int within_base =
  fe_basis.system_to_component_index(i).second;

const unsigned int base =
  fe_basis.system_to_base_index(i).first.first;
const unsigned int multiplicity =
  fe_basis.system_to_base_index(i).first.second;
const unsigned int within_base_  =
  fe_basis.system_to_base_index(i).second; // same as above
*/
std::vector<std::pair< std::pair< unsigned int, unsigned int >, unsigned int >> dof_table(dof_handler.n_dofs());
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
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
            
                

            for(unsigned int i = 0; i < unit_support_points_FE_Q.size(); i++)
                cell_support_points[i] = mapping.transform_unit_to_real_cell(cell, unit_support_points_FE_Q[i]);
            unsigned int comp;
            if(base == 1)
            {
             comp = dof_handler.get_fe().base_element(base).system_to_component_index(within_base_).first;
             support_points[local_dof_indices[i]] = cell_support_points[within_base_];
            }
            std::cout<<local_dof_indices[i]<< " "<< i <<" point "<<support_points[local_dof_indices[i]]<<" base "<<base <<" "<< multiplicity<<" "<< within_base_ <<" comp " <<comp<<std::endl;
        }
        std::cout<<"----------"<<std::endl;
    }



std::cout<<"Start loop ----------"<<std::endl;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {

        fe_values.reinit(cell);

        for (unsigned int l = 0; l < cell->n_lines(); l++) {
            std::cout<<"line------------";
        const typename DoFHandler<dim>::active_line_iterator line =
          cell->line(l);

          std::vector<types::global_dof_index> local_dof_indices(
          fe.n_dofs_per_line() + fe.n_dofs_per_vertex() * 2);

            line->get_dof_indices(local_dof_indices);
            // std::cout<<"fe.n_dofs_per_line() "<<fe.n_dofs_per_line()<< "  fe.n_dofs_per_vertex() "<< fe.n_dofs_per_vertex()<<std::endl;
             for(types::global_dof_index ind: local_dof_indices)
            std::cout<<ind<<" ";

        }
        std::cout<<std::endl;

        cell->get_dof_indices( local_dof_indices);
        std::cout<<"fe.n_dofs_per_cell() "<< fe.n_dofs_per_cell() << " unit_support_points_FE_Q.size() "<<unit_support_points_FE_Q.size()<<
        " unit_support_points_RT.size() "<<unit_support_points_RT.size() <<
        " unit_support_points_general.size() "<<unit_support_points_general.size()<<std::endl;

        for(unsigned int i = 0; i < unit_support_points_FE_Q.size(); i++)
            cell_support_points[i] = mapping.transform_unit_to_real_cell(cell, unit_support_points_FE_Q[i]);

        for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
        {
            std::cout<<local_dof_indices[i]<< " ";
            const unsigned int base = dof_handler.get_fe().system_to_base_index(i).first.first;
            const unsigned int within_base_  = dof_handler.get_fe().system_to_base_index(i).second;
            if(base == 1)
            {
             support_points[local_dof_indices[i]] = cell_support_points[within_base_];
            }

        }
        std::cout<<std::endl;
    }


}

int main()
{
    const unsigned int dim = 2;
    constexpr unsigned int nof_scalar_fields{2};


    Triangulation<dim> triangulation;
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(1);
    unsigned int degree = 0;
    //FE_DGQ<dim> fe(degree);
    
    
    
    
    /*FE_RaviartThomas<dim> fe(degree);
    //convert_generalized_support_point_values_to_dof_values()
    std::vector<Point<dim>> unit_support_points(fe.n_dofs_per_cell());
    unit_support_points =  fe.get_generalized_support_points();
    std::cout<<"fe.n_dofs_per_cell() "<< fe.n_dofs_per_cell() <<" unit_support_points.size "<<unit_support_points.size()<<std::endl;
    for(Point<dim> p : unit_support_points)
    {
        std::cout<<p<<std::endl;
    }*/

   std::cout<<"------------------------------------------"<<std::endl;
    
    
    //FESystem<dim> fe(FE_RaviartThomas<dim>(degree) ^ nof_scalar_fields);//, FE_DGQ<dim>(degree) ^ nof_scalar_fields);//;//
   //  FESystem<dim> fe(FE_DGQ<dim>(degree) ^ nof_scalar_fields);//, 
     FESystem<dim> fe(FESystem<dim>(FE_RaviartThomas<dim>(degree), nof_scalar_fields), FESystem<dim>(FE_DGQ<dim>(degree), nof_scalar_fields));//, 
    // std::cout<<fe.get_name()<<std::endl;

     //std::cout<<fe.base_element(0).get_name()<<std::endl;
     //std::cout<<fe.base_element(1).get_name()<<std::endl;

    // std::cout<<fe.get_sub_fe(0,2).get_name()<<std::endl;
    // std::cout<<fe.get_sub_fe(2,2).get_name()<<std::endl;

    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);
    DoFRenumbering::component_wise(dof_handler); //new

    const std::vector<types::global_dof_index> dofs_per_component =
    DoFTools::count_dofs_per_fe_component(dof_handler);
  
    for(unsigned int i = 0; i < dofs_per_component.size(); i++)
     std::cout<<"dofs_per_component " <<dofs_per_component[i]<<std::endl;
    const unsigned int n_u = dofs_per_component[0] * nof_scalar_fields,
                      n_p = dofs_per_component[dofs_per_component.size()-1] * nof_scalar_fields;
  std::cout << "Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "Total number of cells: " << triangulation.n_cells()
            << std::endl
            << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << " (" << n_u << '+' << n_p << ')' << std::endl;






    QGauss<dim> quadrature_formula(degree + 1);
    FEValues<dim> fe_values(fe, quadrature_formula,
                                update_values | update_gradients |
                                    update_quadrature_points |
                                    update_JxW_values);
    //MappingQ1<dim> mapping;
    std::vector<Point<dim>> support_points;
    std::vector<Point<dim>> support_points2(dof_handler.n_dofs());
    map_dofs_to_support_points(fe_values.get_mapping(), dof_handler, support_points);
   // DoFTools::map_dofs_to_support_points(fe_values.get_mapping(), dof_handler, support_points2);
    // Output the support points
    for (unsigned int i = 0; i < support_points.size(); ++i)
    {   
        bool istheSame = support_points2[i] == support_points[i];
        std::cout << "DoF " << i << ": " << support_points[i] <<std::endl;
        // " v "<< istheSame<< " c " << support_points2[i]<<std::endl;
       // if(!istheSame)
       // std::cout<<"aggggggggggg"<<std::endl;
    }

    return 0;
    
}
