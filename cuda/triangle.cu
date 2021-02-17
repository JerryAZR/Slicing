#include "triangle.cuh"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <thrust/functional.h>

using namespace std;

// point class
// This constructor will help us create an instance of v3 from binary data found in the STL file.
v3::v3(char* facet)
{
    float xx = *((float*) facet );
    float yy = *((float*) facet + 1 );
    float zz = *((float*) facet + 2 );

    x = double(xx);
    y = double(yy);
    z = double(zz);
}

v3::v3(double x, double y, double z) : x(x), y(y), z(z) {}

//triangle class
triangle::triangle(v3 p1, v3 p2, v3 p3) : p1(p1), p2(p2), p3(p3) {}

// util
__host__
void read_stl(string fname, vector <triangle>&v) {
    ifstream myFile (
    fname.c_str(), ios::in | ios::binary);

    char header_info[80] = "";
    char nTri[4];
    unsigned nTriLong;
    unsigned count;

    //read 80 byte header
    if (myFile) {
        myFile.read (header_info, 80);
        cout <<"header: " << header_info << endl;
    }
    else{
        cout << "error" << endl;
    }

    //read 4-byte ulong
    if (myFile) {
        myFile.read (nTri, 4);
        nTriLong = *((unsigned*)nTri) ;
        cout <<"Number of triangles in file: " << nTriLong << endl;
    }
    else{
        cout << "error" << endl;
    }

    count = 0;

    //now read in all the triangles
    for(int i = 0; i < nTriLong; i++){
        char facet[50];
        if (myFile) {
            //read one 50-byte triangle
            myFile.read (facet, 50);
            //populate each point of the triangle
            //using v3::v3(char* bin);
            //Ignore triangles that are parallel to some pixel ray
            // v3 norm(facet);
            // if (norm.z == 0) continue;
            //facet + 12 skips the triangle's unit normal
            v3 p1(facet+12);
            v3 p2(facet+24);
            v3 p3(facet+36);
            //add a new triangle to the array
            v.push_back( triangle(p1,p2,p3) );
            count++;
        }
    }
    cout <<"Number of triangles added: " << count << endl;

    return;
}

__host__
void load_point_array(string fname, vector<vector<double>>&v, vector<triangle>&tris) {
    ifstream myFile (
    fname.c_str(), ios::in | ios::binary);

    char header_info[80] = "";
    char nTri[4];
    unsigned nTriLong;
    unsigned count;

    //read 80 byte header
    if (myFile) {
        myFile.read (header_info, 80);
        cout <<"header: " << header_info << endl;
    }
    else{
        cout << "error" << endl;
    }

    //read 4-byte ulong
    if (myFile) {
        myFile.read (nTri, 4);
        nTriLong = *((unsigned*)nTri) ;
        cout <<"Number of triangles in file: " << nTriLong << endl;
    }
    else{
        cout << "error" << endl;
    }

    count = 0;

    //now read in all the triangles
    for(int i = 0; i < nTriLong; i++){
        char facet[50];
        if (myFile) {
            //read one 50-byte triangle
            myFile.read (facet, 50);
            //populate each point of the triangle
            //using v3::v3(char* bin);
            //Ignore triangles that are parallel to some pixel ray
            v3 norm(facet);
            // if (norm.z == 0) continue;
            //facet + 12 skips the triangle's unit normal
            v3 p1(facet+12);
            v3 p2(facet+24);
            v3 p3(facet+36);
            //add a new triangle to the array
            v[0].push_back(p1.x);
            v[1].push_back(p1.y);
            v[2].push_back(p1.z);
            v[3].push_back(p2.x);
            v[4].push_back(p2.y);
            v[5].push_back(p2.z);
            v[6].push_back(p3.x);
            v[7].push_back(p3.y);
            v[8].push_back(p3.z);
            tris.push_back(triangle(p1,p2,p3));
            count++;
        }
    }
    cout <<"Number of triangles added: " << count << endl;

    return;
}

__host__ void preprocess_stl(string fname, vector<triangle>&small_tri, vector<triangle>&large_tri, vector<double>&zmins) {
    ifstream myFile (
        fname.c_str(), ios::in | ios::binary);

    char header_info[80] = "";
    char nTri[4];
    unsigned nTriLong;
    unsigned small_count, large_count;

    //read 80 byte header
    if (myFile) {
        myFile.read (header_info, 80);
        cout <<"header: " << header_info << endl;
    }
    else{
        cout << "error" << endl;
    }

    //read 4-byte ulong
    if (myFile) {
        myFile.read (nTri, 4);
        nTriLong = *((unsigned*)nTri) ;
        cout <<"Number of triangles in file: " << nTriLong << endl;
    }
    else{
        cout << "error" << endl;
    }

    small_count = 0;
    large_count = 0;

    //now read in all the triangles
    for(int i = 0; i < nTriLong; i++){
        char facet[50];
        if (myFile) {
            //read one 50-byte triangle
            myFile.read (facet, 50);
            //populate each point of the triangle
            //using v3::v3(char* bin);
            //Ignore triangles that are parallel to some pixel ray
            v3 norm(facet);
            if (norm.z == 0) continue;
            //facet + 12 skips the triangle's unit normal
            v3 p1(facet+12);
            v3 p2(facet+24);
            v3 p3(facet+36);
            //add a new triangle to the array
            double zmin = min({p1.z, p2.z, p3.z});
            double zmax = max({p1.z, p2.z, p3.z});
            if (zmax - zmin < TRIANGLE_SIZE) {
                small_tri.push_back(triangle(p1,p2,p3));
                zmins.push_back(zmin);
                small_count++;
            } else {
                large_tri.push_back( triangle(p1,p2,p3) );
                large_count++;
            }
        }
    }
    cout <<"Number of small triangles added: " << small_count << endl;
    cout <<"Number of large triangles added: " << large_count << endl;

    return;
}

__host__
double triangle::zmin() {
    return min({p1.z, p2.z, p3.z});
}
