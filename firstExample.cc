/* ---------------------------------------------------------------------
 *
 * Copyright (C) 1999 - 2023 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 *
 * Author: Wolfgang Bangerth, University of Heidelberg, 1999
 */


// @sect3{Include files}

// The first few (many?) include files have already been used in the previous
// example, so we will not explain their meaning here again.
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/grid/grid_out.h>
#include <fstream>
#include <iostream>

#include <vector>
// This is new, however: in the previous example we got some unwanted output
// from the linear solvers. If we want to suppress it, we have to include this
// file and add a single line somewhere to the program (see the main()
// function below for that):
#include <deal.II/base/logstream.h>

// The final step, as in previous programs, is to import all the deal.II class
// and function names into the global namespace:
using namespace dealii;

// @sect3{The <code>firstExample</code> class template}

// This is again the same <code>firstExample</code> class as in the previous
// example. The only difference is that we have now declared it as a class
// with a template parameter, and the template parameter is of course the
// spatial dim_omegaension in which we would like to solve the Laplace equation. Of
// course, several of the member variables depend on this dim_omegaension as well,
// in particular the Triangulation class, which has to represent
// quadrilaterals or hexahedra, respectively. Apart from this, everything is
// as before.
template <int dim_omega, int dim_sigma>
class firstExample
{
public:
  firstExample();
  void run();

private:
  void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;

  Triangulation<dim_omega> triangulation;
  FE_Q<dim_omega>          fe;
  DoFHandler<dim_omega>    dof_handler;

  Triangulation<dim_sigma> triangulation_sigma;
  FE_Q<dim_sigma>          fe_sigma;
  DoFHandler<dim_sigma>    dof_handler_sigma;
  //Vector<unsigned int>     dof_sigma;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double> solution;
  Vector<double> system_rhs;
};


// @sect3{Right hand side and boundary values}

// In the following, we declare two more classes denoting the right hand side
// and the non-homogeneous Dirichlet boundary values. Both are functions of a
// dim_omega-dim_omegaensional space variable, so we declare them as templates as well.
//
// Each of these classes is derived from a common, abstract base class
// Function, which declares the common interface which all functions have to
// follow. In particular, concrete classes have to overload the
// <code>value</code> function, which takes a point in dim_omega-dim_omegaensional space
// as parameters and returns the value at that point as a
// <code>double</code> variable.
//
// The <code>value</code> function takes a second argument, which we have here
// named <code>component</code>: This is only meant for vector-valued
// functions, where you may want to access a certain component of the vector
// at the point <code>p</code>. However, our functions are scalar, so we need
// not worry about this parameter and we will not use it in the implementation
// of the functions. Inside the library's header files, the Function base
// class's declaration of the <code>value</code> function has a default value
// of zero for the component, so we will access the <code>value</code>
// function of the right hand side with only one parameter, namely the point
// where we want to evaluate the function. A value for the component can then
// simply be omitted for scalar functions.
//
// Function objects are used in lots of places in the library (for example, in
// step-3 we used a Functions::ZeroFunction instance as an argument to
// VectorTools::interpolate_boundary_values) and this is the first tutorial
// where we define a new class that inherits from Function. Since we only ever
// call Function::value(), we could get away with just a plain function (and
// this is what is done in step-5), but since this is a tutorial we inherit from
// Function for the sake of example.
template <int dim_omega>
class RightHandSide : public Function<dim_omega>
{
public:
  virtual double value(const Point<dim_omega>  &p,
                       const unsigned int component = 0) const override;
};



template <int dim_omega>
class BoundaryValues : public Function<dim_omega>
{
public:
  virtual double value(const Point<dim_omega>  &p,
                       const unsigned int component = 0) const override;
};

// If you are not familiar with what the keywords `virtual` and `override` in
// the function declarations above mean, you will probably want to take a look
// at your favorite C++ book or an online tutorial such as
// http://www.cplusplus.com/doc/tutorial/polymorphism/ . In essence, what is
// happening here is that Function<dim_omega> is an "abstract" base class that
// declares a certain "interface" -- a set of functions one can call on
// objects of this kind. But it does not actually *implement* these functions:
// it just says "this is how Function objects look like", but what kind of
// function it actually is, is left to derived classes that implement
// the `value()` function.
//
// Deriving one class from another is often called an "is-a" relationship
// function. Here, the `RightHandSide` class "is a" Function class
// because it implements the interface described by the Function base class.
// (The actual implementation of the `value()` function is in the code block
// below.) The `virtual` keyword then means "Yes, the
// function here is one that can be overridden by derived classes",
// and the `override` keyword means "Yes, this is in fact a function we know
// has been declared as part of the base class". The `override` keyword is not
// strictly necessary, but is an insurance against typos: If we get the name
// of the function or the type of one argument wrong, the compiler will warn
// us by stating "You say that this function overrides one in a base class,
// but I don't actually know any such function with this name and these
// arguments."
//
// But back to the concrete case here:
// For this tutorial, we choose as right hand side the function
// $4(x^4+y^4)$ in 2d, or $4(x^4+y^4+z^4)$ in 3d. We could write this
// distinction using an if-statement on the space dim_omegaension, but here is a
// simple way that also allows us to use the same function in 1d (or in 4D, if
// you should desire to do so), by using a short loop.  Fortunately, the
// compiler knows the size of the loop at compile time (remember that at the
// time when you define the template, the compiler doesn't know the value of
// <code>dim_omega</code>, but when it later encounters a statement or declaration
// <code>RightHandSide@<2@></code>, it will take the template, replace all
// occurrences of dim_omega by 2 and compile the resulting function).  In other
// words, at the time of compiling this function, the number of times the body
// will be executed is known, and the compiler can minimize the overhead
// needed for the loop; the result will be as fast as if we had used the
// formulas above right away.
//
// The last thing to note is that a <code>Point@<dim_omega@></code> denotes a point
// in dim_omega-dim_omegaensional space, and its individual components (i.e. $x$, $y$,
// ... coordinates) can be accessed using the () operator (in fact, the []
// operator will work just as well) with indices starting at zero as usual in
// C and C++.
template <int dim_omega>
double RightHandSide<dim_omega>::value(const Point<dim_omega> &p,
                                 const unsigned int /*component*/) const
{
  double return_value = 0.0;

 //for (unsigned int i = 0; i < dim_omega; ++i)
    //return_value += 4.0 * std::pow(p(i), 4.0);
    
 /* if((int(p[1]*10) > -9 && int(p[1]*10) < 9) && (int(p[0] *10) == 0) )
  {
    return 1;
  } 
  else*/
  return return_value;
}
double g = 1;

// As boundary values, we choose $x^2+y^2$ in 2d, and $x^2+y^2+z^2$ in 3d. This
// happens to be equal to the square of the vector from the origin to the
// point at which we would like to evaluate the function, irrespective of the
// dim_omegaension. So that is what we return:
template <int dim_omega>
double BoundaryValues<dim_omega>::value(const Point<dim_omega> &p,
                                  const unsigned int /*component*/) const
{
  return p.square();
  //return 0.0;
}



// @sect3{Implementation of the <code>firstExample</code> class}

// Next for the implementation of the class template that makes use of the
// functions above. As before, we will write everything as templates that have
// a formal parameter <code>dim_omega</code> that we assume unknown at the time we
// define the template functions. Only later, the compiler will find a
// declaration of <code>firstExample@<2@></code> (in the <code>main</code> function,
// actually) and compile the entire class with <code>dim_omega</code> replaced by 2,
// a process referred to as `instantiation of a template'. When doing so, it
// will also replace instances of <code>RightHandSide@<dim_omega@></code> by
// <code>RightHandSide@<2@></code> and instantiate the latter class from the
// class template.
//
// In fact, the compiler will also find a declaration <code>firstExample@<3@></code>
// in <code>main()</code>. This will cause it to again go back to the general
// <code>firstExample@<dim_omega@></code> template, replace all occurrences of
// <code>dim_omega</code>, this time by 3, and compile the class a second time. Note
// that the two instantiations <code>firstExample@<2@></code> and
// <code>firstExample@<3@></code> are completely independent classes; their only
// common feature is that they are both instantiated from the same general
// template, but they are not convertible into each other, for example, and
// share no code (both instantiations are compiled completely independently).


// @sect4{firstExample::firstExample}

// After this introduction, here is the constructor of the <code>firstExample</code>
// class. It specifies the desired polynomial degree of the finite elements
// and associates the DoFHandler to the triangulation just as in the previous
// example program, step-3:
template <int dim_omega, int dim_sigma>
firstExample<dim_omega, dim_sigma>::firstExample()
  : fe(1)
  , dof_handler(triangulation)
  , fe_sigma(1)
  , dof_handler_sigma(triangulation_sigma)
{}


// @sect4{firstExample::make_grid}

// Grid creation is something inherently dim_omegaension dependent. However, as long
// as the domains are sufficiently similar in 2d or 3d, the library can
// abstract for you. In our case, we would like to again solve on the square
// $[-1,1]\times [-1,1]$ in 2d, or on the cube $[-1,1] \times [-1,1] \times
// [-1,1]$ in 3d; both can be termed GridGenerator::hyper_cube(), so we may
// use the same function in whatever dim_omegaension we are. Of course, the
// functions that create a hypercube in two and three dim_omegaensions are very much
// different, but that is something you need not care about. Let the library
// handle the difficult things.
template <int dim_omega, int dim_sigma>
void firstExample<dim_omega, dim_sigma>::make_grid()
{
  GridGenerator::hyper_cube(triangulation, -1, 1);
  triangulation.refine_global(4);

  GridGenerator::hyper_cube(triangulation_sigma, -1, 1);
  triangulation_sigma.refine_global(4);


  std::ofstream out("grid-1.svg");
  GridOut       grid_out;
  grid_out.write_svg(triangulation, out);
  std::cout << "Grid written to grid-1.svg" << std::endl;

  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;

  std::cout << "   Number of active cells Sigma: " << triangulation_sigma.n_active_cells()
            << std::endl
            << "   Total number of cells Sigma: " << triangulation_sigma.n_cells()
            << std::endl;
}

// @sect4{firstExample::setup_system}

// This function looks exactly like in the previous example, although it
// performs actions that in their details are quite different if
// <code>dim_omega</code> happens to be 3. The only significant difference from a
// user's perspective is the number of cells resulting, which is much higher
// in three than in two space dim_omegaensions!
template <int dim_omega, int dim_sigma>
void firstExample<dim_omega, dim_sigma>::setup_system()
{
  dof_handler.distribute_dofs(fe);

  std::cout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  dof_handler_sigma.distribute_dofs(fe_sigma);

  std::cout << "   Number of degrees of freedom Sigma: " << dof_handler_sigma.n_dofs()
            << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}


// @sect4{firstExample::assemble_system}

// Unlike in the previous example, we would now like to use a non-constant
// right hand side function and non-zero boundary values. Both are tasks that
// are readily achieved with only a few new lines of code in the assemblage of
// the matrix and right hand side.
//
// More interesting, though, is the way we assemble matrix and right hand side
// vector dim_omegaension independently: there is simply no difference to the
// two-dim_omegaensional case. Since the important objects used in this function
// (quadrature formula, FEValues) depend on the dim_omegaension by way of a template
// parameter as well, they can take care of setting up properly everything for
// the dim_omegaension for which this function is compiled. By declaring all classes
// which might depend on the dim_omegaension using a template parameter, the library
// can make nearly all work for you and you don't have to care about most
// things.
template <int dim_omega, int dim_sigma>
void firstExample<dim_omega, dim_sigma>::assemble_system()
{
  std::cout<<"fe.degree "<<fe.degree<<std::endl;
  QGauss<dim_omega> quadrature_formula(fe.degree + 1);
  QGauss<dim_sigma> quadrature_formula_sigma(fe_sigma.degree +1);

  // We wanted to have a non-constant right hand side, so we use an object of
  // the class declared above to generate the necessary data. Since this right
  // hand side object is only used locally in the present function, we declare
  // it here as a local variable:
  RightHandSide<dim_omega> right_hand_side;

  // Compared to the previous example, in order to evaluate the non-constant
  // right hand side function we now also need the quadrature points on the
  // cell we are presently on (previously, we only required values and
  // gradients of the shape function from the FEValues object, as well as the
  // quadrature weights, FEValues::JxW() ). We can tell the FEValues object to
  // do for us by also giving it the #update_quadrature_points flag:
  FEValues<dim_omega> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);
  FEValues<dim_sigma> fe_values_sigma(fe_sigma,
                          quadrature_formula_sigma,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  std::vector<types::global_dof_index> line_source_indices;
  std::vector<double> points_on_source_line;
  
/* for (const auto &cell : dof_handler_sigma.active_cell_iterators())
  {
      fe_values_sigma.reinit(cell);
      for (const unsigned int q_index : fe_values_sigma.quadrature_point_indices())
      {
        
        //std::cout<<"nr "<<fe_values.quadrature_point_indices().size()<<std::endl;
        
        for (const unsigned int i : fe_values_sigma.dof_indices())
          {
             for (const unsigned int j : fe_values_sigma.dof_indices())
            {
            std::cout<< i<<" "<<j<<" "<< ((fe_values_sigma.shape_grad(i, q_index) * // grad phi_i(x_q)
                    fe_values_sigma.shape_grad(j, q_index)))* // grad phi_j(x_q)  
                    fe_values_sigma.JxW(q_index) <<std::endl;
            }
          }
      }

  }*/
  int index = 0;
  
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

    index++;
      
      cell->get_dof_indices(local_dof_indices);
      /*for(auto kk : local_dof_indices)
      std::cout<<kk<<std::endl;
      std::cout<<"---"<<std::endl;*/
      
       /*for (const auto v : cell->vertex_indices())
        {
          const Point<2> center(0, 0);
              const double distance_from_center =
                center.distance(cell->vertex(v));
                std::cout<<"points " <<cell->vertex(v)<<" "<<distance_from_center<<std::endl;
       }*/
     
     // std::cout<<"-------"<<std::endl;
      for (const unsigned int q_index : fe_values.quadrature_point_indices())
      {

       const auto &x_q = fe_values.quadrature_point(q_index);
       const auto &v_q = cell->vertex(q_index);
       
       
  /*if((int(v_q[1]*10) > -9 && int(v_q[1]*10) < 9) && (int(v_q[0]*100) == 0))
  {
    cell_matrix(0,0) +=
           ((fe_values_sigma.shape_grad(0, 0) * // grad phi_i(x_q)
            fe_values_sigma.shape_grad(0, 0)) )* // grad phi_j(x_q)  //+ fe_values.shape_value(i, q_index) 
            fe_values_sigma.JxW(q_index) ;           // dx

                     cell_rhs(0) +=  1;
  }*/
        //std::cout<<"nr "<<fe_values.quadrature_point_indices().size()<<std::endl;
        if((int(v_q[1]*10) > -7 && int(v_q[1]*10) < 7) && (int(v_q[0] *100) == 0) )
        {
          //std::cout<<q_index<<" x_q "<<x_q[0]<<" "<<x_q[1]<<" v_q "<<v_q[0]<<" "<<v_q[1]<<"glob "<<local_dof_indices[q_index]<<std::endl;   
          if(std::find(line_source_indices.begin(),line_source_indices.end(), local_dof_indices[q_index]) == line_source_indices.end() )
          {
            line_source_indices.push_back(local_dof_indices[q_index]);
            points_on_source_line.push_back(v_q[1]);
           // std::cout<<"push "<< v_q[1]<<" "<<local_dof_indices[q_index]<<std::endl;
            
          }
            /*for(auto p :points_on_source_line)
            std::cout<<p<<std::endl;*/
        }

        //Wichtig
        /*if((int(x_q[1]*10) > -7 && int(x_q[1]*10) < 7) && (int(x_q[0] *10) == 0) )
        {          
            //std::cout<<q_index<<" x_q "<<x_q[0]<<" "<<x_q[1]<<" v_q "<<v_q[0]<<" "<<v_q[1]<<"glob "<<local_dof_indices[q_index]<<std::endl;     
            continue;           
        } */

        for (const unsigned int i : fe_values.dof_indices())
          {
            for (const unsigned int j : fe_values.dof_indices())
            {
                //if((int(v_q[1]*10) > -9 && int(v_q[1]*10) < 9) && (int(v_q[0]*100) == 0))
               // if((int(v_q[1]*10) > -9 && int(v_q[1]*10) < 9) && (int(v_q[0] *10) == 0) )
               /* if((int(x_q[1]*10) > -9 && int(x_q[1]*10) < 9) && (int(x_q[0] *10) == 0) )
                //if(index  > 100 && index < 102)
                {
                   cell_matrix(i, j) = 0;
                     cell_rhs(i) =  0;    
                   std::cout<<q_index<<" x_q "<<x_q[0]<<" "<<x_q[1]<<" v_q "<<v_q[0]<<" "<<v_q[1]<<" cell "<<cell_matrix(i, j)<<" "<<cell_rhs(i)<<std::endl;
                   // std::cout<<q_index<<std::endl;
                   // std::cout<<q_index<<" "<<v_q[0]<<" "<<v_q[1]<<" "<<cell_matrix(i, j)<<" " <<cell_rhs(i)<<std::endl;
                  // std::cout<<fe_values.shape_grad(i, q_index)<<std::endl;
                          
                } 
                else*/
                {
                    cell_matrix(i, j) +=
                    ((fe_values.shape_grad(i, q_index) * // grad phi_i(x_q)
                    fe_values.shape_grad(j, q_index)) + g * 1 )* // grad phi_j(x_q)  //+ fe_values.shape_value(i, q_index) 
                    fe_values.JxW(q_index) ;           // dx

                     cell_rhs(i) +=  (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            right_hand_side.value(x_q) *        // f(x_q)
                            fe_values.JxW(q_index));            // dx
                }

            

            }
          }
      
      

      }
      
      // As a final remark to these loops: when we assemble the local
      // contributions into <code>cell_matrix(i,j)</code>, we have to multiply
      // the gradients of shape functions $i$ and $j$ at point number
      // q_index and
      // multiply it with the scalar weights JxW. This is what actually
      // happens: <code>fe_values.shape_grad(i,q_index)</code> returns a
      // <code>dim</code> dimensional vector, represented by a
      // <code>Tensor@<1,dim@></code> object, and the operator* that
      // multiplies it with the result of
      // <code>fe_values.shape_grad(j,q_index)</code> makes sure that the
      // <code>dim</code> components of the two vectors are properly
      // contracted, and the result is a scalar floating point number that
      // then is multiplied with the weights. Internally, this operator* makes
      // sure that this happens correctly for all <code>dim</code> components
      // of the vectors, whether <code>dim</code> be 2, 3, or any other space
      // dimension; from a user's perspective, this is not something worth
      // bothering with, however, making things a lot simpler if one wants to
      // write code dimension independently.

      // With the local systems assembled, the transfer into the global matrix
      // and right hand side is done exactly as before, but here we have again
      // merged some loops for efficiency:

      for (const unsigned int i : fe_values.dof_indices())
        {
          for (const unsigned int j : fe_values.dof_indices())
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));

          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }
    }

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           BoundaryValues<dim_omega>(),
                                           boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
//Remove line
for(unsigned int i = 0; i<line_source_indices.size();i++)
{
 // std::cout<<line_source_indices[i]<<std::endl;
  for(unsigned int j = 0; j< dof_handler.n_dofs();j++)
  {
    system_matrix.set(line_source_indices[i],j,0);
    system_matrix.set(j,line_source_indices[i],0);
  }
  
}
/*for(unsigned int i = 0; i<line_source_indices.size();i++)
{
  system_matrix.set(line_source_indices[i],line_source_indices[i],1);
    std::cout<<"set "<<line_source_indices[i]<<" "<<system_matrix(line_source_indices[i],line_source_indices[i])<<std::endl;
  system_rhs(line_source_indices[i]) = 10;
}*/



for (const auto &cell : dof_handler_sigma.active_cell_iterators())
{
      fe_values_sigma.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;
      bool isLine = true; 
      unsigned int count = 0;
      unsigned int counter1;
      for (const unsigned int q_index : fe_values_sigma.quadrature_point_indices())
      {
       const auto &v_q = cell->vertex(q_index);
      std::vector<double> copy_points; 
      for(double p: points_on_source_line)
      {
          copy_points.push_back(std::abs(p-v_q[0]));

      }
      auto it = std::min_element(std::begin(copy_points), std::end(copy_points));
      unsigned int counter = std::distance(std::begin(copy_points), it);
       if(copy_points[counter]< 0.01)
        {
          isLine = isLine && true;
          count++;
          counter1 = counter;
        }
        else
          isLine = isLine && false;
      }
      if(!isLine)
      continue;

std::vector<unsigned int> local_dof_indices;
      for (const unsigned int q_index : fe_values_sigma.quadrature_point_indices())
      //for(unsigned int q_index = 0; q_index < 1; q_index++)
      {
      const auto &x_q = fe_values_sigma.quadrature_point(q_index);
       const auto &v_q = cell->vertex(q_index);
       //std::cout<<"po "<<v_q[0]<<std::endl;
      std::vector<double> copy_points; 
      for(double p: points_on_source_line)
      {
          copy_points.push_back(std::abs(p-v_q[0]));
          //std::cout<<p<<" "<<v_q[0]<<" "<<p-v_q[0]<<std::endl;
      }
      auto it = std::min_element(std::begin(copy_points), std::end(copy_points));
      //std::cout << "index of smallest element: " << std::distance(std::begin(copy_points), it)<<std::endl;
      unsigned int counter = std::distance(std::begin(copy_points), it);
       //if(copy_points[counter]< 0.01)
        {
          types::global_dof_index index = line_source_indices[counter];
          
          //std::cout<<index<<" "<<copy_points[counter]<<std::endl;
    local_dof_indices.push_back(index );
          
          for (const unsigned int i : fe_values_sigma.dof_indices())
            {
              for (const unsigned int j : fe_values_sigma.dof_indices())
              {
                //std::cout<<q_index<<" cellmatrix "<<cell_matrix(i, j)<<std::endl;
                    cell_matrix(i, j) +=
                    ((fe_values_sigma.shape_grad(i, q_index) * // grad phi_i(x_q)
                    fe_values_sigma.shape_grad(j, q_index)) )* // grad phi_j(x_q)  //+ fe_values.shape_value(i, q_index) 
                    fe_values_sigma.JxW(q_index) ;           // dx

                     cell_rhs(i) += 0.0; //(fe_values.shape_value(i, q_index) * // phi_i(x_q)
                            //right_hand_side.value(x_q) *        // f(x_q)
                            //fe_values.JxW(q_index));            // dx
              }
            }
          }
      }
    
    for (const unsigned int i : fe_values_sigma.dof_indices())
    {
      for (const unsigned int j : fe_values_sigma.dof_indices())
      {
/*if(((local_dof_indices[i] == local_dof_indices[j]) && local_dof_indices[i] == line_source_indices[0]) ||
        ((local_dof_indices[i] == local_dof_indices[j]) && local_dof_indices[i] == line_source_indices[line_source_indices.size()-1]) )
        {
            std::cout<<"Bound "<<local_dof_indices[i] <<std::endl;
            system_rhs(local_dof_indices[i]) = 1.0;//cell_rhs(i);
             system_matrix.set(local_dof_indices[i],
                          local_dof_indices[j],1);
                          std::cout<<"cell_matrix(i, j) "<<cell_matrix(i, j)<<std::endl;
        }
        else*/
        
        system_matrix.add(local_dof_indices[i], local_dof_indices[j],cell_matrix(i, j));
        system_rhs(local_dof_indices[i]) += cell_rhs(i);



           
      }
        


             
    }

  }

 /*for(unsigned int j = 0; j< dof_handler.n_dofs();j++)
  {
    system_matrix.set(line_source_indices[0],j,1);
    system_matrix.set(j,line_source_indices[0],1);

    system_matrix.set(line_source_indices[line_source_indices.size()-1],j,1);
    system_matrix.set(j,line_source_indices[line_source_indices.size()-1],1);
    
  }*/
  system_matrix.set(line_source_indices[0],line_source_indices[0],1);
 // std::cout<<system_matrix(line_source_indices[0],line_source_indices[0]);
  system_rhs(line_source_indices[0]) = 0;
  system_matrix.set(line_source_indices[line_source_indices.size()-1],line_source_indices[line_source_indices.size()-1],1);
  system_rhs(line_source_indices[line_source_indices.size()-1]) = 0;

  system_rhs(line_source_indices[line_source_indices.size()/2]) = 10;
  for(unsigned int i = 0; i < line_source_indices.size(); i++)
  std::cout<< line_source_indices[i]<<" "<<  line_source_indices[i]<<" "<<system_matrix( line_source_indices[i], line_source_indices[i])<<std::endl;
  //system_matrix.print(std::cout);



 /*for(unsigned int ind = 0; ind < line_source_indices.size()-1; ind++)
  {
    for (const unsigned int i : fe_values_sigma.dof_indices())
    {
        for (const unsigned int j : fe_values_sigma.dof_indices())
      {
        types::global_dof_index index = line_source_indices[ind];
            std::cout<<index<< i<<" "<<j<<" "<< ((fe_values_sigma.shape_grad(i, q_index) * // grad phi_i(x_q)
              fe_values_sigma.shape_grad(j, q_index)))* // grad phi_j(x_q)  
              fe_values_sigma.JxW(q_index) <<std::endl;
      }
    }
        
    
  }*/
      

  // As the final step in this function, we wanted to have non-homogeneous
  // boundary values in this example, unlike the one before. This is a simple
  // task, we only have to replace the Functions::ZeroFunction used there by an
  // object of the class which describes the boundary values we would like to
  // use (i.e. the <code>BoundaryValues</code> class declared above):
  //
  // The function VectorTools::interpolate_boundary_values() will only work
  // on faces that have been marked with boundary indicator 0 (because that's
  // what we say the function should work on with the second argument below).
  // If there are faces with boundary id other than 0, then the function
  // interpolate_boundary_values will do nothing on these faces. For
  // the Laplace equation doing nothing is equivalent to assuming that
  // on those parts of the boundary a zero Neumann boundary condition holds.

}


// @sect4{firstExample::solve}

// Solving the linear system of equations is something that looks almost
// identical in most programs. In particular, it is dim_omegaension independent, so
// this function is copied verbatim from the previous example.
template <int dim_omega, int dim_sigma>
void firstExample<dim_omega, dim_sigma>::solve()
{
  SolverControl            solver_control(1000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  // We have made one addition, though: since we suppress output from the
  // linear solvers, we have to print the number of iterations by hand.
  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence." << std::endl;
}


// @sect4{firstExample::output_results}

// This function also does what the respective one did in step-3. No changes
// here for dim_omegaension independence either.
//
// Since the program will run both 2d and 3d versions of the Laplace solver,
// we use the dim_omegaension in the filename to generate distinct filenames for
// each run (in a better program, one would check whether <code>dim_omega</code> can
// have other values than 2 or 3, but we neglect this here for the sake of
// brevity).
template <int dim_omega, int dim_sigma>
void firstExample<dim_omega, dim_sigma>::output_results() const
{
  DataOut<dim_omega> data_out;

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");

  data_out.build_patches();

  //std::ofstream output(dim_omega == 2 ? "solution-2d.vtk" : "solution-3d.vtk");
  std::ofstream output(dim_omega == 1 ? "solution-1d.vtk" : "solution-2d.vtk");
  data_out.write_vtk(output);
}



// @sect4{firstExample::run}

// This is the function which has the top-level control over everything. Apart
// from one line of additional output, it is the same as for the previous
// example.
template <int dim_omega, int dim_sigma>
void firstExample<dim_omega, dim_sigma>::run()
{
  std::cout << "Solving problem in " << dim_omega << " space dim_omegaensions."
            << std::endl;

  make_grid();
  setup_system();
  assemble_system();
  solve();
  output_results();
}


// @sect3{The <code>main</code> function}

// And this is the main function. It also looks mostly like in step-3, but if
// you look at the code below, note how we first create a variable of type
// <code>firstExample@<2@></code> (forcing the compiler to compile the class template
// with <code>dim_omega</code> replaced by <code>2</code>) and run a 2d simulation,
// and then we do the whole thing over in 3d.
//
// In practice, this is probably not what you would do very frequently (you
// probably either want to solve a 2d problem, or one in 3d, but not both at
// the same time). However, it demonstrates the mechanism by which we can
// simply change which dim_omegaension we want in a single place, and thereby force
// the compiler to recompile the dim_omegaension independent class templates for the
// dim_omegaension we request. The emphasis here lies on the fact that we only need
// to change a single place. This makes it rather trivial to debug the program
// in 2d where computations are fast, and then switch a single place to a 3 to
// run the much more computing intensive program in 3d for `real'
// computations.
//
// Each of the two blocks is enclosed in braces to make sure that the
// <code>laplace_problem_2d</code> variable goes out of scope (and releases
// the memory it holds) before we move on to allocate memory for the 3d
// case. Without the additional braces, the <code>laplace_problem_2d</code>
// variable would only be destroyed at the end of the function, i.e. after
// running the 3d problem, and would needlessly hog memory while the 3d run
// could actually use it.
int main()
{
//   {
   // firstExample<1> laplace_problem_1d;
   // laplace_problem_1d.run();
//   }

  {
    firstExample<2, 1> laplace_problem_2d;
    laplace_problem_2d.run();
  }

  // {
  //   firstExample<3> laplace_problem_3d;
  //   laplace_problem_3d.run();
  // }

  return 0;
}
