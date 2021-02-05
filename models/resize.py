#!/usr/bin/env python3

import numpy as np
import stl
import sys
from stl import mesh

if len(sys.argv) < 2:
    model_name = input("STL file path:")
else:
    model_name = sys.argv[1]

fail = True
while fail:
    try:
        triangle_mesh = mesh.Mesh.from_file(model_name)
        fail = False
    except FileNotFoundError:
        print("STL file not found. Please try again")
        model_name = input("STL file path:")
        fail = True
                
print("{} triangles loaded.".format(len(triangle_mesh.v0)))


if len(sys.argv) < 3:
    scale_str = input("Scaling factor:")
else:
    scale_str = sys.argv[2]
    
fail = True
while fail:
    try:
        scale = float(scale_str)
        fail = False
    except ValueError:
        print("Please enter a number")
        scale_str = input("Scaling factor:")
        fail = True
    
print("Scaling factor set to {}x".format(scale))

triangle_mesh.v0 = triangle_mesh.v0 * scale
triangle_mesh.v1 = triangle_mesh.v1 * scale
triangle_mesh.v2 = triangle_mesh.v2 * scale

new_file_name = model_name[:-4] + "_{}x.stl".format(scale)

triangle_mesh.save(new_file_name, mode = stl.Mode.BINARY)
print("Output file saved to " + new_file_name)
