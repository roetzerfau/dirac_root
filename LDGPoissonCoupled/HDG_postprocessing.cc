

using namespace dealii;

template <int dim> class TrueSolutionPotential : public Function<dim> {
public:
  TrueSolutionPotential() : Function<dim>() {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override;
  virtual double value(const Point<dim> & p,
      const unsigned int component = 0) const override;
private:
    const TrueSolution<dim> true_solution;
};





template <int dim>
double post_process(const parallel::shared::Triangulation<dim> &triangulation, unsigned int degree, const UpdateFlags &update_flags,
    const FESystem<dim> &fe_local,  const DoFHandler<dim> &dof_handler_local,
    const TrilinosWrappers::MPI::Vector solution_local,
    const FEValuesExtractors::Vector VectorField, const FEValuesExtractors::Scalar Potential,
    double alpha, double radius, double h_min)
{
    std::cout<<"HDG post process"<<std::endl;
    //Vector<double> u_star(dof_handler_u_star.n_dofs());

    QGauss<dim> quadrature_formula(degree + 3);


    FE_DGQ<dim> fe_u_star(degree + 1);
    DoFHandler<dim> dof_handler_u_star(triangulation);
    dof_handler_u_star.distribute_dofs(fe_u_star);
    FEValues<dim> fe_values_u_star(fe_u_star, quadrature_formula, update_flags);
    Vector<double>  solution_u_star(dof_handler_u_star.n_dofs());

    FEValues<dim> fe_values_local(fe_local, quadrature_formula, update_flags);





     typename DoFHandler<dim>::active_cell_iterator cell_u_star = dof_handler_u_star
                                                              .begin_active(),
                                                   endc_u_star = dof_handler_u_star.end();


    std::vector<double> u_values(quadrature_formula.size());
    std::vector<Tensor<1, dim>> q_values(quadrature_formula.size());

  //  u_values(quadrature_formula.size());
   // q_values(quadrature_formula.size());

   // Vector<double> solution_local;
   // solution_local.reinit(dof_handler_local.n_dofs());

    for (; cell_u_star != endc_u_star; ++cell_u_star) {

#if USE_MPI_ASSEMBLE
      if (cell_u_star ->is_locally_owned())
#endif
      {



    const typename DoFHandler<dim>::active_cell_iterator cell_local =
      cell_u_star->as_dof_handler_iterator(dof_handler_local);
 
    fe_values_local.reinit(cell_local);
    fe_values_u_star.reinit(cell_u_star);


 
    const unsigned int dofs_per_cell = fe_values_u_star.dofs_per_cell;
    const unsigned int n_q_points = fe_values_u_star.n_quadrature_points;
  //  std::cout<<"a "<<quadrature_formula.size()<<" b "<<n_q_points<<std::endl;
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double> cell_vector(dofs_per_cell);
    Vector<double> cell_solution_u_star(dofs_per_cell);
    cell_matrix = 0;
    cell_vector = 0;

 
    fe_values_local[Potential].get_function_values(solution_local,
                                                        u_values);
    fe_values_local[VectorField].get_function_values(solution_local,
                                                        q_values);

//TODO K inverse unequal Id
  for (unsigned int q = 0; q < n_q_points; ++q) {
    for (unsigned int i = 1; i < dofs_per_cell; ++i) {
      const Tensor<1,dim> grad_psi_i_potential =
         fe_values_u_star.shape_grad(i, q);

      for (unsigned int j = 0; j < dofs_per_cell; ++j) {
        const Tensor<1,dim> grad_psi_j_potential =
                fe_values_u_star.shape_grad(j, q);
    
    
        cell_matrix(i, j) +=  grad_psi_i_potential * grad_psi_j_potential* fe_values_u_star.JxW(q);
        
      }

       cell_vector(i) += - grad_psi_i_potential * q_values[q] * fe_values_u_star.JxW(q);
    }
  }

    for (unsigned int j = 0; j < dofs_per_cell; ++j)
    {
        for (unsigned int q = 0; q < n_q_points; ++q){
        
        cell_matrix(0, j) += fe_values_u_star.shape_value(j, q) * fe_values_u_star.JxW(q);
        }
    }
    {
    for (unsigned int q = 0; q < n_q_points; ++q)
       cell_vector(0) += u_values[q] * fe_values_u_star.JxW(q);
    }


    cell_matrix.gauss_jordan();
    cell_matrix.vmult(cell_solution_u_star, cell_vector);
    cell_u_star->distribute_local_to_global(cell_solution_u_star, solution_u_star);

    
















      }

    }

/*
 const TrueSolution<dim> true_solution;
 DoFHandler<dim> dof_handler_Lag(triangulation);
  FESystem<dim> fe_Lag(FESystem<dim>(FE_DGQ<dim>(degree), dim),
                       FE_DGQ<dim>(degree));
  dof_handler_Lag.distribute_dofs(fe_Lag);
  TrilinosWrappers::MPI::Vector solution_const;
  solution_const.reinit(dof_handler_Lag.locally_owned_dofs(), MPI_COMM_WORLD);

  VectorTools::interpolate(dof_handler_Lag, true_solution, solution_const);

FEValues<dim> fe_values_Lag(fe_Lag, quadrature_formula, update_flags);


  ///TrilinosWrappers::MPI::Vector solution_const;
  //solution_const.reinit(dof_handler_local.locally_owned_dofs(), MPI_COMM_WORLD);
  //VectorTools::interpolate(dof_handler_local, true_solution, solution_const);

solution_const.print(std::cout);

cell_u_star = dof_handler_u_star.begin_active();
    double sum_a=0, sum_a1=0;
  for (; cell_u_star != endc_u_star; ++cell_u_star) {
   // std::cout<<cell_u_star<<std::endl;
std::vector<double> u_values(quadrature_formula.size());
std::vector<double> u_values_const(quadrature_formula.size());
std::vector<double> u_star_values(quadrature_formula.size());

#if USE_MPI_ASSEMBLE
      if (cell_u_star ->is_locally_owned())
#endif
      {
        const typename DoFHandler<dim>::active_cell_iterator cell_local =
        cell_u_star->as_dof_handler_iterator(dof_handler_local);
                const typename DoFHandler<dim>::active_cell_iterator cell_Lag =
        cell_u_star->as_dof_handler_iterator(dof_handler_Lag);
 
        fe_values_local.reinit(cell_local);
        fe_values_u_star.reinit(cell_u_star);
        fe_values_Lag.reinit(cell_Lag);
        
        fe_values_local[Potential].get_function_values(solution_local,
                                                        u_values);
        fe_values_Lag[Potential].get_function_values(solution_const,
                                                        u_values_const);                                                        
        fe_values_u_star.get_function_values(solution_u_star,
                                                        u_star_values);

       // for(auto i : u_values)                                           
       // std::cout<<i<<" ";
       // std::cout<<std::endl;
       // for(auto i : u_star_values)                                           
       // std::cout<<i<<" ";
       // std::cout<<std::endl;  
       // std::cout<<"-------------------"<<std::endl;  
    
            double sum = 0, sum1 = 0;
            for(unsigned int i = 0; i < u_values.size();i++)
            {
              // std::cout<<u_values_const[i]<< " " <<u_values[i]<<std::endl;
                sum += (u_values_const[i] - u_star_values[i]) *  (u_values_const[i] - u_star_values[i]);
                sum1 += (u_values_const[i] - u_values[i]) *  (u_values_const[i] - u_values[i]);
            }

            std::cout<<"u_star " <<sqrt(sum)<<std::endl;
            std::cout<<"u " <<sqrt(sum1)<<std::endl<<"--------"<<std::endl;
            sum_a+=sum * sum;
            sum_a1 += sum1 * sum1;
    }

    }

    std::cout<<"u_star "<<sqrt(sum_a)<<std::endl;
    std::cout<<"u "<<sqrt(sum_a1)<<std::endl;

*/


    //solution_u_star.print(std::cout);

    const QTrapezoid<1> q_trapez;
    const QIterated<dim> quadrature(q_trapez, degree + 3);

    Vector<double> cellwise_errors_U;
    cellwise_errors_U.grow_or_shrink(triangulation.n_active_cells());
     

    const ComponentSelectFunction<dim> potential_mask(dim,
                                                      dim + 1);
    const DistanceWeight<dim> distance_weight(alpha, radius, h_min); //, radius
    const ProductFunction<dim> connected_function_potential(potential_mask,
                                                            distance_weight);


    const TrueSolutionPotential<dim> true_solution_potential;
    

    VectorTools::integrate_difference(
        dof_handler_u_star, solution_u_star, true_solution_potential, cellwise_errors_U, quadrature,
        VectorTools::L2_norm, &distance_weight); //
   std::cout<<"error"<<std::endl;
       std::cout<<"max cellwise_errors_U "<<cellwise_errors_U.linfty_norm()<<std::endl;
    std::cout<<"l1_norm cellwise_errors_U "<<cellwise_errors_U.l1_norm()<<std::endl;
    std::cout<<"L2_norm  cellwise_errors_U "<<cellwise_errors_U.l2_norm ()<<std::endl;
   //cellwise_errors_U.print(std::cout);
    return  VectorTools::compute_global_error(
        triangulation, cellwise_errors_U, VectorTools::L2_norm);    

}


template <int dim>
void TrueSolutionPotential<dim>::vector_value(const Point<dim> &p,
                                     Vector<double> &values) const {
                                     
             //   std::cout<<"values.size() "<<values.size()<<std::endl;
            // Assert(values.size() == 1, ExcDimensionMismatch(values.size(), 1));
                Vector<double> values_complete(dim +1);
                true_solution.vector_value(p,values_complete);
                values_complete.print(std::cout);
                values[dim] = values_complete[dim];                 
}


template <int dim>
double TrueSolutionPotential<dim>::value(const Point<dim> & p,
      const unsigned int component) const {
                                     
               // std::cout<<"aallllaaa "<<std::endl;
            // Assert(values.size() == 1, ExcDimensionMismatch(values.size(), 1));
                Vector<double> values_complete(dim +1);
                true_solution.vector_value(p,values_complete);
               // values_complete.print(std::cout);
              return  values_complete[dim];                 
}