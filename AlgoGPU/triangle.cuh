//source: http://www.sgh1.net/posts/read-stl-file.md
#ifndef TRIANGLE
#define TRIANGLE

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

#include <string>
#include <vector>

using std::string;
using std::vector;

typedef
struct v3
{
    public:
    __host__ __device__ v3(char* bin);
    __host__ __device__ v3(double x = 0.0, double y = 0.0, double z = 0.0);
    // ~v3();

    double x, y, z;
} v3;

typedef
struct triangle
{
    public:
    __host__ __device__ triangle(v3 p1 = v3(), v3 p2 = v3(), v3 p3 = v3());
    // ~triangle();
    v3 p1, p2, p3;
} triangle;

// utils
__host__ void read_stl(string fname, vector<triangle>&v);

#endif