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

class v3
{
    public:
    __host__ __device__ v3(char* bin);
    __host__ __device__ v3(double x = 0.0, double y = 0.0, double z = 0.0);
    // ~v3();
    __host__ void display();
    __host__ __device__ v3 operator+(v3 ptr);
    __host__ __device__ v3 operator-(v3 ptr);
    __host__ __device__ v3 operator*(double a);

    double x, y, z;
};

class triangle
{
    public:
    __host__ __device__ triangle(v3 p1 = v3(), v3 p2 = v3(), v3 p3 = v3());
    // ~triangle();
    __host__ void display();
    v3 p1, p2, p3;
};

// utils
__host__ void read_stl(string fname, vector<triangle>&v);

#endif
