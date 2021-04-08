from subprocess import run
from os.path import join
from os import listdir
from tqdm import tqdm

def trace_all(dir, names, total):
    for i in tqdm(range(total)):
        infile = join(dir, names.format(i)+".bmp")
        outfile = join(dir, names.format(i)+".svg")
        cmd = ["potrace", "-a", "0", "-b", "svg", infile, "-o", outfile]
        run(cmd)

if __name__ == "__main__":
    files = listdir("./")
    total = len([f for f in files if ".bmp" in f])
    trace_all("./", "layer_{}", total)
    
