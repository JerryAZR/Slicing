#!/usr/bin/env python3

import numpy as np
from stl import mesh
from tqdm import tqdm
from multiprocessing import Process
from subprocess import run

def analyze_single(fname):
    run(["python3", "analyze.py", fname])

all_stl = ["Android.stl", "ring.stl", "puppy.stl", "bunny.stl", "squirrel.stl",
            "tweel.stl", "heart.stl", "possum.stl", "kidney.stl", "engine.stl"]

if __name__ == "__main__":
    processes = []
    for f in all_stl:
        p = Process(target=analyze_single, args=[f])
        p.start()
        processes.append(p)

    for i in range(len(processes)):
        processes[i].join()