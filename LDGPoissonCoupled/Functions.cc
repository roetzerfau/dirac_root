// @sect3{Functions.cc}
// In this file we keep right hand side function, Dirichlet boundary
// conditions and solution to our Poisson equation problem.  Since
// these classes and functions have been discussed extensively in
// the deal.ii tutorials we won't discuss them any further.
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <cmath>
#include <numbers> 
//std::numbers::PI

using namespace dealii;
const double w = numbers::PI * 3 / 2;
const double y_l = 0.0;
const double z_l = 0.0;

template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide() : Function<dim>(1)
  {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0 ) const override;
};
template <int dim>
class RightHandSide_omega : public Function<dim>
{
public:
  RightHandSide_omega(): Function<dim>(1)
  {}
  virtual double value(const Point<dim>  &p,
                           const unsigned int component = 0) const override;
};

template <int dim>
class DirichletBoundaryValues : public Function<dim>
{
public:
  DirichletBoundaryValues() : Function<dim>(1)
  {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0 ) const override;
};
template <int dim>
class DirichletBoundaryValues_omega : public Function<dim>
{
  public:
  DirichletBoundaryValues_omega() : Function<dim>(1)
  {}

  virtual double value(const Point<dim>  &p,
                           const unsigned int component = 0) const override;
};

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


template<int dim>
class TrueSolution : public Function<dim>
{
public:
  TrueSolution() : Function<dim>(dim +3)
  {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &valuess) const override;
};
template<int dim>
class TrueSolution_omega : public Function<dim>
{
public:
  TrueSolution_omega() : Function<dim>(dim +1)
  {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &valuess) const override;
};

template <int dim>
double
RightHandSide<dim>::
value(const Point<dim> &p,
      const unsigned int ) const
{
 return 0;
 if(dim == 2)
 return(2 * std::pow(w, 2)) * std::cos(w * p[0] ) *
             std::cos(w * p[1]);
  if(dim == 3)
  return (3 * std::pow(w, 2)) * std::cos(w * p[0] ) *
             std::cos(w * p[1]) * std::cos(w * p[2]);
}
template <int dim>
double RightHandSide_omega<dim>::value(const Point<dim> & p,
                                     const unsigned int /*component*/) const
{
  return  2;
   if(dim ==2)
    return std::pow(w, 2) * std::cos(w * p[0]) * std::cos(w * y_l);
  if(dim ==3)
    return std::pow(w, 2) * std::cos(w * p[0]) * std::cos(w * y_l) * std::cos(w * z_l);
}

template <int dim>
double
DirichletBoundaryValues<dim>::
value(const Point<dim> &p,
      const unsigned int ) const
{
  double x = p[0];
  double y = p[1];
  if(dim ==2)
  return  std::cos(w * x) * std::cos(w * y);
  if(dim == 3)
  {
     double z = p[2];
     return  std::cos(w * x) * std::cos(w * y) * std::cos(w * z);
  }
}
    
template <int dim>
double DirichletBoundaryValues_omega<dim>::value(const Point<dim> &p,
                                       const unsigned int /*component*/) const
{
  if(dim == 2)
  return  std::cos(w * p[0]) * std::cos(w * y_l);
  if(dim == 3)
  return  std::cos(w * p[0]) * std::cos(w * y_l) * std::cos(w * z_l);
}


template <int dim>
void KInverse<dim>::value_list(const std::vector<Point<dim>> &points,
                                std::vector<Tensor<2, dim>>   &values) const
{
  (void)points;
  AssertDimension(points.size(), values.size());

  for (auto &value : values)
    value = unit_symmetric_tensor<dim>();
}


template <int dim>
void
TrueSolution<dim>::
vector_value(const Point<dim> &p,
             Vector<double> &values) const
{
  Assert(values.size() == dim+3,
         ExcDimensionMismatch(values.size(), dim+3) );
  if(dim == 2)
  {
  double x = p[0];
  double y = p[1];

  values(0) = w *std::sin(w * x) * std::cos(w * y);
  values(1) = w *std::cos(w *x) * std::sin(w * y);
  values(2) = w *std::sin(w * x) * std::cos(w * y_l);
  values(3) = std::cos(w * x) * std::cos(w * y);
  values(4) = std::cos(w * x) * std::cos(w * y_l);
  }
  if(dim == 3)
  {
    double x = p[0];
    double y = p[1];
    double z = p[2];

  values(0) = w *std::sin(w * x) * std::cos(w * y) * std::cos(w * z);
  values(1) = w *std::cos(w * x) * std::sin(w * y) * std::cos(w * z);
  values(2) = w *std::cos(w * x) * std::cos(w * y) * std::sin(w * z);
  values(3) = w *std::sin(w * x) * std::cos(w * y_l) * std::cos(w * z_l);
  values(4) = std::cos(w * x) * std::cos(w * y) * std::cos(w * z);
  values(5) = std::cos(w * x) * std::cos(w * y_l) * std::cos(w * z_l);
  }
}

template <int dim>
void
TrueSolution_omega<dim>::
vector_value(const Point<dim> &p,
             Vector<double> &values) const
{
 // std::cout<<"values.size() "<<values.size()<<std::endl;
  Assert(values.size() == dim + 1,
         ExcDimensionMismatch(values.size(), dim + 1) );
 
    double x = p[0];

  if(dim == 2)
  { 
    values(0) = w * std::sin(w * x) * std::cos(w * y_l);
    values(1) = std::cos(w * x) * std::cos(w * y_l);
  }
  if(dim == 3)
  { 
    values(0) = w * std::sin(w * x) * std::cos(w * y_l)* std::cos(w * z_l);
    values(1) = std::cos(w * x) * std::cos(w * y_l) * std::cos(w * z_l);
  }
 
}