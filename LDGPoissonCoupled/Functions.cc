// @sect3{Functions.cc}
// In this file we keep right hand side function, Dirichlet boundary
// conditions and solution to our Poisson equation problem.  Since
// these classes and functions have been discussed extensively in
// the deal.ii tutorials we won't discuss them any further.
#include <cmath>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <numbers>
// std::numbers::PI

using namespace dealii;
const double w = numbers::PI * 3 / 2;
const double y_l = 0.0;
const double z_l = 0.0;
const double radius = 0.01;
const bool lumpedAvarage = true;

constexpr unsigned int constructed_solution{3};   // 1:sin cos, 2:papper log, 3: dangelo thesis log
const double g = constructed_solution == 3 ? (2 * numbers::PI) / (2 * numbers::PI + std::log(radius)): 1;

template <int dim> double distance(Point<dim> point1, Point<dim> point2) {
  double d = 0;
  for (unsigned int i = 0; i < dim; i++) {
    d += std::pow(point1[i] - point2[i], 2);
  }
  return std::sqrt(d);
}

template <int dim> class RightHandSide : public Function<dim> {
public:
  RightHandSide() : Function<dim>(1) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};
template <int dim> class RightHandSide_omega : public Function<dim> {
public:
  RightHandSide_omega() : Function<dim>(1) {}
  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

template <int dim> class DirichletBoundaryValues : public Function<dim> {
public:
  DirichletBoundaryValues() : Function<dim>(1) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};
template <int dim> class NeumannBoundaryValues : public Function<dim> {
public:
  NeumannBoundaryValues() : Function<dim>(1) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};
template <int dim> class DirichletBoundaryValues_omega : public Function<dim> {
public:
  DirichletBoundaryValues_omega() : Function<dim>(1) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

template <int dim> class KInverse : public TensorFunction<2, dim> {
public:
  KInverse() : TensorFunction<2, dim>() {}

  virtual void value_list(const std::vector<Point<dim>> &points,
                          std::vector<Tensor<2, dim>> &values) const override;
};

template <int dim> class TrueSolution : public Function<dim> {
public:
  TrueSolution() : Function<dim>(dim + 3) {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override;
};
template <int dim> class TrueSolution_omega : public Function<dim> {
public:
  TrueSolution_omega() : Function<dim>(dim + 1) {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override;
};
template <int dim> class ProductFunction : public Function<dim> {
public:
  ProductFunction (const Function<dim> &f1,
                    const Function<dim> &f2) : Function<dim>(dim + 3), function1(f1), function2(f2) {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override;
   
  private:
    const Function<dim> &function1;
    const Function<dim> &function2;                          
};
template <int dim> class DistanceWeight : public Function<dim> {
public:
  DistanceWeight(double _alpha, double R = 0) : Function<dim>(dim + 3), alpha(_alpha), radius(R) {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override;
  private:
    double alpha, radius;
};
template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int) const {
  switch (constructed_solution) {
  case 1: {
    if (dim == 2)
      return (2 * std::pow(w, 2)) * std::cos(w * p[0]) * std::cos(w * p[1]);
    if (dim == 3)
      return (3 * std::pow(w, 2)) * std::cos(w * p[0]) * std::cos(w * p[1]) *
             std::cos(w * p[2]);
    break;
  }
  case 2: {
    return 0;
    break;
  }
  case 3: {
    return 0;
    break;
  }
  default:
    return 0;
    break;
  }
}
template <int dim>
double RightHandSide_omega<dim>::value(const Point<dim> &p,
                                       const unsigned int /*component*/) const {
   //std::cout<<"rhs omega dim "<<dim<<" "<<p[0]<<std::endl;
  switch (constructed_solution) {
  case 1: {
   /*if (dim == 2)
      return std::pow(w, 2) * std::cos(w * p[0]) * std::cos(w * y_l);
    if (dim == 3)*/
      return std::pow(w, 2) * std::cos(w * p[0]) * std::cos(w * y_l) *
             std::cos(w * z_l);
    break;
  }
  case 2: {
    //if (dim == 3) 
    {
      return std::pow(numbers::PI, 2) * std::sin(numbers::PI * p[0]);
    }
    break;
  }
  case 3: {
    return -(1 + p[0]);
    //return - std::sin(2 * numbers::PI * p[0]);//std::pow(2 * numbers::PI, 2) * std::sin(2 * numbers::PI * p[0]) ;
    break;
  }
  default:
    break;
  }
}

template <int dim>
double DirichletBoundaryValues<dim>::value(const Point<dim> &p,
                                           const unsigned int) const {
  double x = p[0];
  double y = p[1];

  switch (constructed_solution) {
  case 1: {
    if (dim == 2)
      return std::cos(w * x) * std::cos(w * y);
    if (dim == 3) {
      double z = p[2];
      return std::cos(w * x) * std::cos(w * y) * std::cos(w * z);
    }
    break;
  }
  case 2: {

    if (dim == 3) {
      /* Vector<double> values;
       Point<1> p_omega = Point<1>(x);
       TrueSolution_omega(p_omega, values);
       return   values[1] * g/(1 +g) * (1 - radius * std::log(r/radius));
       */
      // std::cout<<"bound"<<std::endl;
      Vector<double> values(6);
      TrueSolution<dim> solution;
      solution.vector_value(p, values);
      // std::cout<<"BD_Omega "<<values[4]<<std::endl;
      return values[4];
    }
    break;
  case 3: {
    return 0;
    break;
  }
  }
  default:
    break;
  }
}

template <int dim>
double NeumannBoundaryValues<dim>::value(const Point<dim> &p,
                                         const unsigned int) const {
  double x, y, z;
  x = p[0];
  y = p[1];
  Point<dim> closest_point_line;
 if (dim == 2)
    closest_point_line = Point<dim>(x, y_l);
  if (dim == 3)
  {
    z = p[2];
    closest_point_line = Point<dim>(x, y_l, z_l);
  }
  double r = distance(p, closest_point_line);

  switch (constructed_solution) {
  case 3: {
    if (p[0] == 1)
    {
    //  std::cout<<"neum1 "<<p[0]<<std::endl;
      return -1 / (2 * numbers::PI) * std::log(r);///-
    }
    if (p[0] == 0)
    {
      //std::cout<<"neum0 "<<p[0]<<std::endl;
      return 1 / (2 * numbers::PI) * std::log(r);
    }
    break;
  }
  default:
    break;
  }
}

template <int dim>
double DirichletBoundaryValues_omega<dim>::value(
    const Point<dim> &p, const unsigned int /*component*/) const {
  switch (constructed_solution) {
  case 1: {
    /*if (dim == 2)
      return std::cos(w * p[0]) * std::cos(w * y_l);
    if (dim == 3)*/
      return std::cos(w * p[0]) * std::cos(w * y_l) * std::cos(w * z_l);
    break;
  }
  case 2: {
    // if(dim == 3)
    {
      Vector<double> values(dim + 1);
      TrueSolution_omega<dim> solution;
      solution.vector_value(p, values);
      // std::cout<<"BD_omega "<<values[1]<<std::endl;
      return values[1];
    }
    break;
  }
  case 3: {
    //return 0;
    if (p[0] == 0)
      return 1;
    if (p[0] == 1)
      return 2;

    break;
  }
  default:
    break;
  }
}

template <int dim>
void KInverse<dim>::value_list(const std::vector<Point<dim>> &points,
                               std::vector<Tensor<2, dim>> &values) const {
  (void)points;
  AssertDimension(points.size(), values.size());
  // std::cout<<"points.size() "<<points.size()<<std::endl;
  // for (auto &value : values)

  for (auto &value : values) {
    value = unit_symmetric_tensor<dim>();
    //if(dim == 1)
    //std::cout <<"kinverse_omega " <<value<<std::endl;
    if (constructed_solution == 3 && dim == 1) {
      for (unsigned int i = 0; i < dim; i++) {
        Point<dim> p = points[i];
      // value[i][i]  = p[0] + 1;//
      value[i][i]  = 1/(1 + p[0] + 0.5 * std::pow(p[0], 2));
      //value[i][i] = -std::pow(numbers::PI *2,2);
           
      }
    }
  }
}

template <int dim>
void TrueSolution<dim>::vector_value(const Point<dim> &p,
                                     Vector<double> &values) const {
  Assert(values.size() == dim + 3,
         ExcDimensionMismatch(values.size(), dim + 3));
  double x, y, z;
  x = p[0];
  y = p[1];
  values = 0;
  Point<dim> closest_point_line;
 if (dim == 2)
    closest_point_line = Point<dim>(x, y_l);
  if (dim == 3) 
  {
    z = p[2];
    closest_point_line = Point<dim>(x, y_l, z_l);
  }
  double r = distance(p, closest_point_line);

  switch (constructed_solution) {

  case 1: {
    if (dim == 2) {
      values(0) = w * std::sin(w * x) * std::cos(w * y); // Q
      values(1) = w * std::cos(w * x) * std::sin(w * y);
      values(2) = w * std::sin(w * x) * std::cos(w * y_l); // q
      values(3) = std::cos(w * x) * std::cos(w * y);       // U
      values(4) = std::cos(w * x) * std::cos(w * y_l);     // u
    }
    if (dim == 3) 
    {
      values(0) = w * std::sin(w * x) * std::cos(w * y) * std::cos(w * z); // Q
      values(1) = w * std::cos(w * x) * std::sin(w * y) * std::cos(w * z);
      values(2) = w * std::cos(w * x) * std::cos(w * y) * std::sin(w * z);
      values(3) =
          w * std::sin(w * x) * std::cos(w * y_l) * std::cos(w * z_l);     // q
      values(4) = std::cos(w * x) * std::cos(w * y) * std::cos(w * z);     // U
      values(5) = std::cos(w * x) * std::cos(w * y_l) * std::cos(w * z_l); // u
    }
    break;
  }
  case 2: {
    /*if(dim == 2)
    {

    values(0) = 0; //Q
    values(1) = 0;
    values(2) = r > radius ? values(4) * g/(1 +g) * (1 - radius *
    std::log(r/radius)) : values(4) * g/(1 +g) ; // U values(3) = 0; //q
    values(4) = sin(numbers::PI);//u
    }*/
   // if (dim == 3) 
    {
      values(3) = -numbers::PI * std::cos(numbers::PI * x); // q
      values(5) = std::sin(numbers::PI * x);                // u

      values(0) = r > radius
                      ? g / (g + 1) * (1 - radius * std::log(r / radius)) *
                            numbers::PI * std::cos(numbers::PI * x)
                      : g / (g + 1) * values(3); // Q
      values(1) =
          r > radius
              ? -g / (g + 1) * (radius * (y - y_l) / std::pow(r, 2)) * values(5)
              : 0;
      ;
      values(2) =
          r > radius
              ? -g / (g + 1) * (radius * (z - z_l) / std::pow(r, 2)) * values(5)
              : 0;

      values(4) = r > radius ? values(5) * g / (1 + g) *
                                   (1 - radius * std::log(r / radius))
                             : values(5) * g / (1 + g); // U
    }
    break;
  }
  case 3: {
    if (r != 0) {
      values(0) = 1/(2*numbers::PI) * std::log(r); //Q 
       values(1) = (1+x)/(2*numbers::PI) * (y/std::pow(r,2)); // Q
      values(2) = (1+x)/(2*numbers::PI) * (z/std::pow(r,2)); //Q
    //values(1) = 0;
    //values(2) = (1+x)/ (2*numbers::PI *r );
   //std::cout<<r << " " <<x <<" " <<values(2)<<std::endl;
      values(4) = -(1 + x) / (2 * numbers::PI) * std::log(r); // U  
      //values(4) = std::sin(numbers::PI * 2 *x) / (2 * numbers::PI) * std::log(r); // U      
    } else {
      values(4) = (1 + x); // U
      //values(4) = std::sin(numbers::PI * 2 *x); // U
    }
    //values(5) = std::sin(numbers::PI * 2 *x);  // u
    values(5) = 1 + x ;  // u
    values(3) = -(1 + x + 0.5 * std::pow(x,2)); //q
    break;
  }

  default:
    break;
  }
}

template <int dim>
void TrueSolution_omega<dim>::vector_value(const Point<dim> &p,
                                           Vector<double> &values) const {
  // std::cout<<"values.size() "<<values.size()<<std::endl;
  Assert(values.size() == dim + 1,
         ExcDimensionMismatch(values.size(), dim + 1));

  double x = p[0];
  values = 0;
   //std::cout<<constructed_solution<<" "<<dim<<" ";
  switch (constructed_solution) {
  case 1: {
   /*if (dim == 2) {
      values(0) = w * std::sin(w * x) * std::cos(w * y_l);
      values(1) = std::cos(w * x) * std::cos(w * y_l);
    }
    if (dim == 3) */
    {
     
      values(0) = w * std::sin(w * x) * std::cos(w * y_l) * std::cos(w * z_l);
      values(1) = std::cos(w * x) * std::cos(w * y_l) * std::cos(w * z_l);
    }
    break;
  }
  case 2: {
    // if(dim == 3)
    {
      values(0) = -numbers::PI * std::cos(numbers::PI * x); // q
      values(1) = std::sin(numbers::PI * x) + 2;            // u
    }
    break;
  }
  case 3: {
    //values(1) = std::sin(numbers::PI * 2 *x);//u
    values(1) = 1 + x;//u
    values(0) = -(1 + x + 0.5 * std::pow(x,2)); //q 
    // std::cout<<"w "<<values(0)<<" "<<values(1)<<std::endl;
    break;
  }
  default:
    break;
  }
 // std::cout<<values<<std::endl;
}
template <int dim>
void ProductFunction<dim>::vector_value(const Point<dim> &p,
                                     Vector<double> &values) const 
{
    Assert(values.size() == dim + 3,
         ExcDimensionMismatch(values.size(), dim + 3));

    const unsigned int n_components = function1.n_components;
   
    AssertDimension(function2.n_components, n_components);
    values.reinit(n_components);

    Vector<double> value1(n_components), value2(n_components);
    function1.vector_value(p, value1);
    function2.vector_value(p, value2);
    
    for (unsigned int i = 0; i < n_components; ++i)
    {
        values[i] = value1[i] * value2[i];
    }
    //std::cout<<p <<" "<<values<<std::endl;
    /*if(value2[0] == 0)
    std::cout<<p<<" | " << value1 <<  " | "<< value2<< " | "<< values<<std::endl;
    */

}
template <int dim>
void DistanceWeight<dim>::vector_value(const Point<dim> &p,
                                     Vector<double> &values) const {
  Assert(values.size() == dim + 3,
         ExcDimensionMismatch(values.size(), dim + 3));
  unsigned int n_components = dim + 3;
  double x, y, z;
  x = p[0];
  y = p[1];
  values = 0;
  Point<dim> closest_point_line;
 if (dim == 2)
    closest_point_line = Point<dim>(x, y_l);
  if (dim == 3) 
  {
    z = p[2];
    closest_point_line = Point<dim>(x, y_l, z_l);
  }
  double r = distance(p, closest_point_line);

  for(unsigned int i = 0; i < n_components; i++)
  {
    if(r < radius)
       values(i) = 0;
    else  
      values(i) = 1;
    //values(i) = std::pow(r,2*alpha);
  }
 /* if(values[0] == 0)
  std::cout<<"distValues " <<values<<std::endl;
*/
   

}


template <int dim>
Point<dim> cross_product(const Point<dim> &a, const Point<dim> &b) {
  // static_assert(dim == 3, "Cross product is only defined for 3-dimensional
  // space.");
  return Point<dim>(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0]);
}

template <int dim>
std::vector<Point<dim>>
equidistant_points_on_circle(const Point<dim> &center, double radius,
                             const Point<dim> &normal, int num_points = 10) {
  std::vector<Point<dim>> points;
  if (num_points == 1) {
    points.push_back(center);
    return points;
  }
  double angle_step =
      2 * M_PI / num_points; // Angle between each point in radians
  if (dim == 2) {
    for (int i = 0; i < num_points; ++i) {
      double angle = i * angle_step;
      double x = center[0] + radius * std::cos(angle);
      double y = center[1] + radius * std::sin(angle);
      points.push_back(Point<dim>(x, y));
    }
  }
  if (dim == 3) {
    // static_assert(dim == 3, "This function is designed for 3-dimensional
    // space only.");
    //  Normalize the normal vector
    Point<dim> norm = normal / normal.norm();

    // Create a vector that is not parallel to the normal vector
    Point<dim> arbitrary_vector = (std::abs(norm[0]) > std::abs(norm[1]))
                                      ? Point<dim>(0, 1, 0)
                                      : Point<dim>(1, 0, 0);

    // Compute two orthogonal vectors in the plane of the circle
    Point<dim> u = cross_product(arbitrary_vector, norm);
    u /= u.norm(); // Normalize u
    Point<dim> v = cross_product(norm, u);

    for (int i = 0; i < num_points; ++i) {
      double angle = i * angle_step;
      double x = center[0] +
                 radius * (u[0] * std::cos(angle) + v[0] * std::sin(angle));
      double y = center[1] +
                 radius * (u[1] * std::cos(angle) + v[1] * std::sin(angle));
      double z = center[2] +
                 radius * (u[2] * std::cos(angle) + v[2] * std::sin(angle));
      points.push_back(Point<dim>(x, y, z));
    }
  }
  return points;
}