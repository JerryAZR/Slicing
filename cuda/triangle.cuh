//source: http://www.sgh1.net/posts/read-stl-file.md
#ifndef TRIANGLE
#define TRIANGLE

#include <string>
#include <vector>

using std::string;
using std::vector;

#define TRIANGLE_SIZE 2

typedef
struct v3
{
    public:
    __host__ v3(char* bin);
    __host__ v3(double x, double y, double z);
    __host__ __device__ v3() {}
    // ~v3();

    double x, y, z;
} v3;

typedef
struct triangle
{
    public:
    __host__ triangle(v3 p1, v3 p2, v3 p3);
    __host__ __device__ triangle() {}
    // ~triangle();
    v3 p1, p2, p3;
} triangle;

// utils
__host__ void read_stl(string fname, vector<triangle>&v);
__host__ void preprocess_stl(string fname, vector<triangle>&small_tri, vector<triangle>&large_tri, vector<double>&zmins);

#endif
