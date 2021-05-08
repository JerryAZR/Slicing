#!/usr/bin/env python3

import numpy as np
import stl
import sys
from stl import mesh
from tqdm import tqdm
from multiprocessing import cpu_count

## analyze_mesh
# results = [x_dim, y_dim, z_dim, tri_area, tri_ratio, bbox_area, bbox_ratio]
def analyze_mesh(points, results, use_pgbar=True):
    num_tris = len(points)

    min_x, min_y, min_z = points[0][0:3]
    max_x, max_y, max_z = points[0][0:3]

    tri_area_array = np.empty(num_tris)
    bbox_area_array = np.empty(num_tris)

    iter_obj = tqdm(range(num_tris)) if use_pgbar else range(num_tris)

    for i in iter_obj:
        # Get all points
        x0, y0, z0, x1, y1, z1, x2, y2, z2 = points[i]
        
        # Get 2D bounding box
        local_min_x = min(x0, x1, x2)
        local_max_x = max(x0, x1, x2)
        local_min_y = min(y0, y1, y2)
        local_max_y = max(y0, y1, y2)
        bbox_area = (local_max_x-local_min_x) * (local_max_y-local_min_y)
        bbox_area_array[i] = bbox_area

        # Update global min/max values
        min_x = min(local_min_x, min_x)
        max_x = max(local_max_x, max_x)
        min_y = min(local_min_y, min_y)
        max_y = max(local_max_y, max_y)
        min_z = min(z0, z1, z2, min_z)
        max_z = max(z0, z1, z2, max_z)

        # Compute triangle area
        vect1 = np.array([x1, y1]) - np.array([x0, y0])
        vect2 = np.array([x2, y2]) - np.array([x0, y0])
        tri_area = np.linalg.norm(np.cross(vect1, vect2)) / 2
        tri_area_array[i] = tri_area

    mean_tri_area = np.mean(tri_area_array)
    mean_bbox_area = np.mean(bbox_area_array)

    results[0:3] = [max_x-min_x, max_y-min_y, max_z-min_z]
    results[3:5] = [tri_area, bbox_area]

if __name__ == "__main__":
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

    num_tris = len(triangle_mesh.v0)
    print("{}: {} triangles loaded.".format(model_name, num_tris))

    results = np.empty(5)
    analyze_mesh(triangle_mesh.points[0::1], results)
    x_dim, y_dim, z_dim = results[0:3]
    mean_tri_area, mean_bbox_area = results[3:5]
    tri_ratio = mean_tri_area / (x_dim*y_dim)
    bbox_ratio = mean_bbox_area / (x_dim*y_dim)

    print("Model dimensions: {} x {} x {}".format(x_dim, y_dim, z_dim))

    print("Average triangle size: {}".format(mean_tri_area))
    print("Average triangle/layer size ratio: {:E}".format(tri_ratio))

    print("Average bounding box size: {}".format(mean_bbox_area))
    print("Average bbox/layer size ratio: {:E}".format(bbox_ratio))

    with open("model_analysis.csv", "a") as f:
        f.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(model_name, num_tris,
        x_dim, y_dim, z_dim, mean_tri_area, tri_ratio, mean_bbox_area, bbox_ratio))
