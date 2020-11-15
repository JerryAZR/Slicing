#!/usr/bin/env python3
'''Slicer Run Script

Usage:
    run.py test [--exe EXE] [--stl STL]
    run.py prof [-a | --all] [--txt | --csv] [--exe EXE] [--stl STL]
    run.py -h | --help

Options:
    --exe EXE   The implemenation to test or profile. 
                Can be one of (fps|pps|new) [default: fps]
    --stl STL   The stl file to be sliced [default: models/bunny.stl]
    -a --all    Proflie all events and metrics
    --txt       Save output to a text file
    --csv       Save output to a csv file
    -h --help   Show this screen.
'''

from docopt import docopt
from subprocess import run
import re

if __name__ == '__main__':
    args = docopt(__doc__)
    exe = "./out/" + args["--exe"]
    stl = args["--stl"]
    model = re.search("/(.*).stl", stl).group(1)
    outFileName = "performance/" + args["--exe"] + "-" + model

    if args["test"]:
        # Use test binary instead of original
        exe += "-test"
        run([exe, stl])
    
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

