//source: http://www.sgh1.net/posts/read-stl-file.md
#ifndef TRIANGLE
#define TRIANGLE

#include<string>
#include<vector>

using std::string;
using std::vector;

class v3
{
public:
    v3(char* bin);
    v3(double x = 0.0, double y = 0.0, double z = 0.0);
    // ~v3();
    void display();
    v3 operator+(v3 ptr);
    v3 operator-(v3 ptr);
    v3 operator*(double a);

    double x, y, z;
};

class triangle
{
public:
    triangle(v3 p1 = v3(), v3 p2 = v3(), v3 p3 = v3());
    // ~triangle();
    void display();
    v3 p1, p2, p3;
};

// utils
void read_stl(string fname, vector <triangle>& v);

#endif
