#!/usr/bin/env python3
'''Slicer Run Script

Usage:
    ./run.py test <exe> [--stl STL]
    ./run.py prof <exe> [-a | --all] [--txt | --csv] [--stl STL]
    ./run.py list
    ./run.py -h | --help

Options:
    --stl STL   The stl file to be sliced [default: models/bunny.stl]
    -a --all    Proflie all events and metrics
    --txt       Save output to a text file
    --csv       Save output to a csv file
    -h --help   Show this screen.

Arguments:
    exe         The implemenation to test or profile. Can be one of (fps|pps|new)
'''

from docopt import docopt
from subprocess import run
import re
import os

if __name__ == '__main__':
    args = docopt(__doc__)
    if args["<exe>"]:
        exe = "./out/" + args["<exe>"]
        stl = args["--stl"]
        model = re.search("/(.*).stl", stl).group(1)
        outFileName = "performance/" + args["<exe>"] + "-" + model

    if args["test"]:
        # Use test binary instead of original
        exe += "-test"
        run([exe, stl])
    
    elif args["list"]:
        for file in os.listdir("models/"):
            if file.endswith(".stl"):
                print(os.path.join("models/", file))
    
    elif args["prof"]:
        cmd = ["nvprof"]
        if args["--all"]:
            cmd += ["--events", "all", "--metrics", "all"]
            outFileName += "-all"
        if args["--csv"]:
            cmd += ["--csv"]
            outFileName += ".csv"
        else:
            outFileName += ".txt"
        cmd += [exe, stl]

        if args["--csv"] or args["--txt"]:
            with open(outFileName, "w") as outFile:
                run(cmd, stderr=outFile)
        else:
            run(cmd)

