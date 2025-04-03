// @sect3{Functions.cc}
// In this file we keep right hand side function, Dirichlet boundary
// conditions and solution to our Poisson equation problem.  Since
// these classes and functions have been discussed extensively in
// the deal.ii tutorials we won't discuss them any further.
#include <cmath>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/mapping.h>
#include <numbers>
// std::numbers::PI

#define COUPLED 1
#define TEST 1
#define SOLVE_BLOCKWISE 1
#define GRADEDMESH 1
#define MEMORY_CONSUMPTION 0

#define USE_MPI_ASSEMBLE 1
#define FASTER 1 //nur verfügbar bei der aktuellsten dealii version
#define CYLINDER 0
#define A11SCHUR 0

#define ANISO 1
#define PAPER_SOLUTION 1 //paper dangelo, O: thesis
#define VESSEL 0
#define SOLUTION1_LINEAR 1
using namespace dealii;
const double w = numbers::PI * 3 / 2;

enum GeometryConfiguration
{
  TwoD_ZeroD = 0, //constructed solution 3 (omega wird unabhängig davon auch noch ausgerechnet)
  TwoD_OneD = 1,//constructed solution 1(Coupled)
  ThreeD_OneD = 2 ////constructed solution 1, 2, 3

};
const bool is_omega_on_face = true;
constexpr double y_l = is_omega_on_face ? 0.0 : 0.0001;
constexpr double z_l =  is_omega_on_face ? 0.0 : 0.0001;
constexpr unsigned int geo_conf{2};
constexpr unsigned int dimension_Omega = geo_conf == ThreeD_OneD ? 3 : 2;
constexpr unsigned int constructed_solution{1};   // 1:sin cos (Kopplung hebt sich auf), 2: omega constant funktion, ohne fluss, 3: dangelo thesis log, linear funktion on omega



const unsigned int refinement[4] = {1,2,3,4};//,7,8,9,10
const unsigned int p_degree[1] = {1};

const unsigned int n_r = 1;
const unsigned int n_LA = 1;
const double radii[n_r] = {0.01};
const double D = 1;
const double penalty_sigma = 10;

#if PAPER_SOLUTION && COUPLED
const double sol_factor = D * radii[0]/(1- D * radii[0]*  std::log(radii[0]));
#else
const double sol_factor =  1/(2*numbers::PI);
#endif

const bool lumpedAverages[n_LA] = {false};//TODO bei punkt wuelle noch berücksichtnge

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
template <int dim> class NeumannBoundaryValues_omega : public Function<dim> {
public:
  NeumannBoundaryValues_omega() : Function<dim>(1) {}

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
  TrueSolution() : Function<dim>(dim + 1) {}

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
                    const Function<dim> &f2) : Function<dim>(dim + 1), function1(f1), function2(f2) {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override;
   
  private:
    const Function<dim> &function1;
    const Function<dim> &function2;                          
};
template <int dim> class DistanceWeight : public Function<dim> {
public:
  DistanceWeight(double _alpha, double R = 1, double h = 1) : Function<dim>(dim + 1), alpha(_alpha), radius(R), cell_size(h) {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double> &values) const override;
  private:
    double alpha, radius, cell_size;
};

template <int dim>
Point<dim> nearest_point_on_singularity(const Point<dim> &p)
{
  double x = p[0];
  
  Point<dim> nearest_singularity_point;
  if (GeometryConfiguration::TwoD_OneD == geo_conf )
    nearest_singularity_point = Point<dim>(x, y_l); //nearest point on line
  if (GeometryConfiguration::ThreeD_OneD == geo_conf) 
  {
   nearest_singularity_point = Point<dim>(x, y_l, z_l);
  }
  
  
  if(GeometryConfiguration::TwoD_ZeroD == geo_conf )
  {
     nearest_singularity_point = Point<dim>( y_l,z_l);//center
  }
  return  nearest_singularity_point;
}
template <int dim>
double distance_to_singularity(const Point<dim> &p)
{
  
  double r;

  r = distance(p, nearest_point_on_singularity(p));
  return r;
}


template <int dim>
double RightHandSide<dim>::value(const Point<dim> &p,
                                 const unsigned int) const {
  switch (constructed_solution) {
  case 1: {
   #if SOLUTION1_LINEAR
     return 0;
   #else
   return -2;
    #endif
    if (dim == 2)
      return (2 * std::pow(w, 2)) * std::cos(w * p[0]) * std::cos(w * p[1]);
    if (dim == 3)
      return (3 * std::pow(w, 2)) * std::cos(w * p[0]) * std::cos(w * p[1]) *
             std::cos(w * p[2]);
    break;
  }
  case 2:
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
  // std::cout<<"rhs omega dim "<<dim<<" "<<p[0]<<std::endl;
  switch (constructed_solution) {
  case 1: {
  /* if (dim == 2)
      return std::pow(w, 2) * std::cos(w * p[0]) * std::cos(w * y_l);
    if (dim == 3)*/
    #if SOLUTION1_LINEAR
      return 0;
    #else
      return -2;
    #endif
      return std::pow(w, 2) * std::cos(w * p[0]) * std::cos(w * y_l) *
             std::cos(w * z_l);
    break;
  }
  case 2:
  {
#if PAPER_SOLUTION
if(COUPLED == 1)
return 2 * numbers::PI *sol_factor+1;
#else 
if(COUPLED == 1)
return 2;
else 
return 1;
#endif

    break;
  }
  case 3: {
    if(geo_conf == GeometryConfiguration::TwoD_ZeroD)
      return -(1 + p[0]);
    else
    {
    if(COUPLED == 1)
      return (1 + p[0]) * 2 * numbers::PI* sol_factor - (1 + p[0]);
    else
      return -(1 + p[0]);
    }

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
  double x, y,z;
  x = p[0];
  y = p[1];
  if(dim == 3)
  z= p[2];
  switch (constructed_solution) {
  case 1: {
    Vector<double> values(dim + 1);
    TrueSolution<dim> solution;
    solution.vector_value(p, values);
    return values[dim];
   
    #if SOLUTION1_LINEAR
    return x;
    #else
    return std::pow(x,2);
    #endif

    if (dim == 2)
      return std::cos(w * x) * std::cos(w * y);
    if (dim == 3) {
      return std::cos(w * x) * std::cos(w * y) * std::cos(w * z);
    }
    break;
  }
  case 2:
  case 3: {
     Vector<double> values(dim + 1);
      TrueSolution<dim> solution;
      solution.vector_value(p, values);
      return values[dim];
      //return 0;
    break;
  }
  default:
    break;
  }
}

template <int dim>
double NeumannBoundaryValues<dim>::value(const Point<dim> &p,
                                         const unsigned int) const {
  double x;//, y, z;
  x = p[0];
  //y = p[1];
  double r = distance_to_singularity<dim>(p);

  switch (constructed_solution) {
  case 2:
      return 0;
  case 3: {
    if (p[0] > 1)
    {
    //  std::cout<<"neum1 "<<p[0]<<std::endl;
      return  sol_factor  * std::log(r);///-
    }
    if (p[0] < 1)
    {
      //std::cout<<"neum0 "<<p[0]<<std::endl;
      return - sol_factor  * std::log(r);
    }
    break;
  }
  default:
  {
     std::cout<<"default"<<std::endl;
      break;
  }
   
  }
  return 0;
}
template <int dim>
double NeumannBoundaryValues_omega<dim>::value(
    const Point<dim> &p, const unsigned int /*component*/) const {
  switch (constructed_solution) {
  case 1:
  case 2:
  case 3: {
    Vector<double> values(dim + 1);
      TrueSolution_omega<dim> solution;
      solution.vector_value(p, values);
      return values[0];
    break;
  }
  default:
  {
    std::cout<<"default"<<std::endl;
    break;
  }
    
  }
 return 0;
}
template <int dim>
double DirichletBoundaryValues_omega<dim>::value(
    const Point<dim> &p, const unsigned int /*component*/) const {
  switch (constructed_solution) {
  case 1: {
    Vector<double> values(dim + 1);
    TrueSolution_omega<dim> solution;
    solution.vector_value(p, values);
    return values[dim];

    /*if (dim == 2)
      return std::cos(w * p[0]) * std::cos(w * y_l);
    if (dim == 3)*/
    #if SOLUTION1_LINEAR
       return p[0];
  #else
       return std::pow(p[0],2);
  #endif
      return std::cos(w * p[0]) * std::cos(w * y_l) * std::cos(w * z_l);
    break;
  }
  case 2:
  case 3: {
    //return 0;
    /*if (p[0] == 0)
      return 1;
    if (p[0] == 1)
      return 2;*/
    Vector<double> values(dim + 1);
      TrueSolution_omega<dim> solution;
      solution.vector_value(p, values);
      return values[dim];

  //return 1 + p[0];
    break;
  }
  default:
  {
    std::cout<<"default"<<std::endl;
    break;
  }
    
  }
 return 0;
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
    if ((constructed_solution == 3||constructed_solution == 2) && dim == 1) {
      for (unsigned int i = 0; i < dim; i++) {
        Point<dim> p = points[i];
      // value[i][i]  = p[0] + 1;//
      value[i][i]  = 1/(1 + p[0] + 0.5 * std::pow(p[0], 2));
     //value[i][i]  = 1/(2 * numbers::PI*radii[0]/(1- radii[0]*  std::log(radii[0]))* p[0]) ;
      //value[i][i] = -std::pow(numbers::PI *2,2);
           
      }
    }
  }
}

template <int dim>
void TrueSolution<dim>::vector_value(const Point<dim> &p,
                                     Vector<double> &values) const {
  Assert(values.size() == dim + 1,
         ExcDimensionMismatch(values.size(), dim + 1));
  double x, y, z;
  x = p[0];
  y = p[1];
  if(dim ==3)
    z = p[2];
  values = 0;
  double r =  distance_to_singularity<dim>(p);

  if(r > (1 + 0.01))
  std::cout<<"FALSCH!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!   "<<r<<std::endl;
  switch (constructed_solution) {

  case 1: {
    if (dim == 2) {
      values(0) = w * std::sin(w * x) * std::cos(w * y); // Q
      values(1) = w * std::cos(w * x) * std::sin(w * y);
      values(2) = std::cos(w * x) * std::cos(w * y);       // U

      #if SOLUTION1_LINEAR
      values(0) = -1; // Q
      values(1) = 0;
      values(2) = x + 1;     // U
      #else
      values(0) = - 2 * x; // Q
      values(1) = 0;
      values(2) = std::pow(x,2);     // U
      #endif

    }
    if (dim == 3) 
    {
      
      values(0) = w * std::sin(w * x) * std::cos(w * y) * std::cos(w * z); // Q
      values(1) = w * std::cos(w * x) * std::sin(w * y) * std::cos(w * z);
      values(2) = w * std::cos(w * x) * std::cos(w * y) * std::sin(w * z);
      values(3) = std::cos(w * x) * std::cos(w * y) * std::cos(w * z);     // U

      #if SOLUTION1_LINEAR
      values(0) = -1; // Q
      values(1) = 0;
      values(2) = 0;
      values(3) = x + 1;     // U
      #else
      values(0) = - 2 * x; // Q
      values(1) = 0;
      values(2) = 0;
      values(3) = std::pow(x,2);     // U
      #endif


    }
    break;
  }
  case 2:
  {
    if(dim==3)
    {
    if (r != 0) {
      values(0) =0; //Q 
       values(1) = sol_factor * (y/std::pow(r,2)); // Q
      values(2) =  sol_factor * (z/std::pow(r,2)); //Q
      values(3) = - sol_factor * std::log(r); // U  
    } else {
      values(3) = 1  ; // U
    }
     }
     break;
  }
  case 3: {
    if(dim==3)//
    {
    if (r != 0) {
      values(0) = sol_factor * std::log(r); //Q 
       values(1) = (1.0+x) *sol_factor* (y/std::pow(r,2)); // Q
      values(2) = (1.0+x)*sol_factor * (z/std::pow(r,2)); //Q
      values(3) = -(1.0+x)*sol_factor* std::log(r); // U  
    } else {
      values(3) = 1 + x ; // U
    }

     }
     if(dim == 2)//
     {
      if(GeometryConfiguration::TwoD_ZeroD == geo_conf)
      {
#if COUPLED
    if(r!= 0)
    {

    values(0) = radii[0]/(1- radii[0]*  std::log(radii[0])) * (x/std::pow(r,2)); //Q 
    values(1) =radii[0]/(1- radii[0]*  std::log(radii[0])) * (y/std::pow(r,2)); // Q   
    values(2) = -radii[0]/(1- radii[0]*  std::log(radii[0]))  *  std::log(r); // U   
    }
    else
      values(2) = 1 ;
#else
    if(r!= 0)
    {
    values(0) = 1/(2*numbers::PI) * (x/std::pow(r,2)); //Q 
    values(1) =  1/(2*numbers::PI) *(y/std::pow(r,2)); // Q   
    values(2) = - 1/(2*numbers::PI) *  std::log(r); // U   
    }
    else
      values(2) = 1 ;
#endif
  }
  }
    break;
  }
  default:
    break;
  }
}

template <int dim>
void TrueSolution_omega<dim>::vector_value(const Point<dim> &p,
                                           Vector<double> &values) const {
   //std::cout<<"values.size() "<<values.size()<<std::endl;
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
    //{
     
      values(0) = w * std::sin(w * x) * std::cos(w * y_l) * std::cos(w * z_l);
      values(1) = std::cos(w * x) * std::cos(w * y_l) * std::cos(w * z_l);

      #if SOLUTION1_LINEAR
        values(0) = -1;
        values(1) = x+1;
        
      #else
        values(0) = -2 * x;
        values(1) = std::pow(x,2);
      #endif


     //std::cout<<x<<" "<< values(0) <<" "<< values(1) <<std::endl;
    //}
    break;
  }
  case 2:
  {
    values(0) = 0; //q 
    values(1) = 1;//u
    break;
  }
  case 3: {
     //values(0) = 0; //q 
    //values(1) =  1 + x;//u
    values(0) = -(1 + x + 0.5 * std::pow(x,2)); //q 
    values(1) = 1 + x;//u
    // std::cout<<"w "<<values(0)<<" "<<values(1)<<std::endl;
    break;
  }
  default:
    break;
  }
 //std::cout<<p<<" | "<<values<<std::endl;
}
template <int dim>
void ProductFunction<dim>::vector_value(const Point<dim> &p,
                                     Vector<double> &values) const 
{
    Assert(values.size() == dim + 1,
         ExcDimensionMismatch(values.size(), dim + 1));

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
  Assert(values.size() == dim +1,
         ExcDimensionMismatch(values.size(), dim + 1));
  unsigned int n_components = values.size();
  double r;

 // y = p[1];
  values = 1;

  
  r = distance_to_singularity<dim>(p);


  for(unsigned int i = 0; i < n_components; i++)
  {
  
    if(r <= cell_size * 1.1)//
    {
       values(i) = 0;
    }
    else
      values(i) = std::pow(r,2*alpha);
    //else  
if(constructed_solution == 1)
   values(i) = 1;
// values(i) = 1;
  //values(i) = values(i) * std::pow(r,2*alpha);
  }
 
 //values = 1;
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

  template <int dim, int spacedim>
  std::vector<std::pair<typename Triangulation<dim, spacedim>::active_cell_iterator,
                        Point<dim>>>
  find_all_active_cells_around_point(
      const Mapping<dim, spacedim>  &mapping,
      const Triangulation<dim, spacedim> &mesh,
      const Point<spacedim>         &p,
      const double                   tolerance,
      const std::pair<typename Triangulation<dim, spacedim>::active_cell_iterator,
                      Point<dim>>   &first_cell,
      const std::vector<
        std::set<typename Triangulation<dim, spacedim>::active_cell_iterator>>
        *vertex_to_cells)
  {
    std::vector<
      std::pair<typename Triangulation<dim, spacedim>::active_cell_iterator,
                Point<dim>>>
      cells_and_points;
 
    // insert the fist cell and point into the vector
    cells_and_points.push_back(first_cell);
 
    const Point<dim> unit_point = cells_and_points.front().second;
    const auto       my_cell    = cells_and_points.front().first;
 
    std::vector<typename Triangulation<dim, spacedim>::active_cell_iterator>
      cells_to_add;
 
    if (my_cell->reference_cell().is_hyper_cube())
      {
        // check if the given point is on the surface of the unit cell. If yes,
        // need to find all neighbors
 
        Tensor<1, dim> distance_to_center;
        unsigned int   n_dirs_at_threshold     = 0;
        unsigned int   last_point_at_threshold = numbers::invalid_unsigned_int;
        for (unsigned int d = 0; d < dim; ++d)
          {
            distance_to_center[d] = std::abs(unit_point[d] - 0.5);
            if (distance_to_center[d] > 0.5 - tolerance)
              {
                ++n_dirs_at_threshold;
                last_point_at_threshold = d;
              }
          }
 
        // point is within face -> only need neighbor
        if (n_dirs_at_threshold == 1)
          {
            unsigned int neighbor_index =
              2 * last_point_at_threshold +
              (unit_point[last_point_at_threshold] > 0.5 ? 1 : 0);
            if (!my_cell->at_boundary(neighbor_index))
              {
                const auto neighbor_cell = my_cell->neighbor(neighbor_index);
 
                if (neighbor_cell->is_active())
                  cells_to_add.push_back(neighbor_cell);
                else
                  for (const auto &child_cell :
                       neighbor_cell->child_iterators())
                    {
                      if (child_cell->is_active())
                        cells_to_add.push_back(child_cell);
                    }
              }
          }
        // corner point -> use all neighbors
        else if (n_dirs_at_threshold == dim)
          {
            unsigned int local_vertex_index = 0;
            for (unsigned int d = 0; d < dim; ++d)
              local_vertex_index += (unit_point[d] > 0.5 ? 1 : 0) << d;
 
            const auto fu = [&](const auto &tentative_cells) {
              for (const auto &cell : tentative_cells)
                if (cell != my_cell)
                  cells_to_add.push_back(cell);
            };
 
            const auto vertex_index = my_cell->vertex_index(local_vertex_index);
 
            if (vertex_to_cells != nullptr)
              fu((*vertex_to_cells)[vertex_index]);
            else
              fu(GridTools::find_cells_adjacent_to_vertex(mesh, vertex_index));
          }
        // point on line in 3d: We cannot simply take the intersection between
        // the two vertices of cells because of hanging nodes. So instead we
        // list the vertices around both points and then select the
        // appropriate cells according to the result of read_to_unit_cell
        // below.
        else if (n_dirs_at_threshold == 2)
          {
            std::pair<unsigned int, unsigned int> vertex_indices[3];
            unsigned int                          count_vertex_indices = 0;
            unsigned int free_direction = numbers::invalid_unsigned_int;
            for (unsigned int d = 0; d < dim; ++d)
              {
                if (distance_to_center[d] > 0.5 - tolerance)
                  {
                    vertex_indices[count_vertex_indices].first = d;
                    vertex_indices[count_vertex_indices].second =
                      unit_point[d] > 0.5 ? 1 : 0;
                    ++count_vertex_indices;
                  }
                else
                  free_direction = d;
              }
 
            AssertDimension(count_vertex_indices, 2);
            Assert(free_direction != numbers::invalid_unsigned_int,
                   ExcInternalError());
 
            const unsigned int first_vertex =
              (vertex_indices[0].second << vertex_indices[0].first) +
              (vertex_indices[1].second << vertex_indices[1].first);
            for (unsigned int d = 0; d < 2; ++d)
              {
                const auto fu = [&](const auto &tentative_cells) {
                  for (const auto &cell : tentative_cells)
                    {
                      bool cell_not_yet_present = true;
                      for (const auto &other_cell : cells_to_add)
                        if (cell == other_cell)
                          {
                            cell_not_yet_present = false;
                            break;
                          }
                      if (cell_not_yet_present)
                        cells_to_add.push_back(cell);
                    }
                };
 
                const auto vertex_index =
                  my_cell->vertex_index(first_vertex + (d << free_direction));
 
                if (vertex_to_cells != nullptr)
                  fu((*vertex_to_cells)[vertex_index]);
                else
                  fu(GridTools::find_cells_adjacent_to_vertex(mesh, vertex_index));
              }
          }
      }
    else
      {
        // Note: The non-hypercube path takes a very naive approach and
        // checks all possible neighbors. This can be made faster by 1)
        // checking if the point is in the inner cell and 2) identifying
        // the right lines/vertices so that the number of potential
        // neighbors is reduced.
 
        for (const auto v : my_cell->vertex_indices())
          {
            const auto fu = [&](const auto &tentative_cells) {
              for (const auto &cell : tentative_cells)
                {
                  bool cell_not_yet_present = true;
                  for (const auto &other_cell : cells_to_add)
                    if (cell == other_cell)
                      {
                        cell_not_yet_present = false;
                        break;
                      }
                  if (cell_not_yet_present)
                    cells_to_add.push_back(cell);
                }
            };
 
            const auto vertex_index = my_cell->vertex_index(v);
 
            if (vertex_to_cells != nullptr)
              fu((*vertex_to_cells)[vertex_index]);
            else
              fu(GridTools::find_cells_adjacent_to_vertex(mesh, vertex_index));
          }
      }
 
    for (const auto &cell : cells_to_add)
      {
        if (cell != my_cell)
          try
            {
              const Point<dim> p_unit =
                mapping.transform_real_to_unit_cell(cell, p);
              if (cell->reference_cell().contains_point(p_unit, tolerance))
              //if (cell->point_inside(p))
                cells_and_points.emplace_back(cell, p_unit);
            }
          catch (typename Mapping<dim>::ExcTransformationFailed &)
            {}
      }
 
    std::sort(
      cells_and_points.begin(),
      cells_and_points.end(),
      [](const std::pair<typename Triangulation<dim, spacedim>::active_cell_iterator,
                         Point<dim>> &a,
         const std::pair<typename Triangulation<dim, spacedim>::active_cell_iterator,
                         Point<dim>> &b) { return a.first < b.first; });
 
    return cells_and_points;
  }



