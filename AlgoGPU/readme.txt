//To compile:
nvcc *.cu -o print.exe
//To run the program
./print.exe ./bunny.stl
//To get performance
sudo nvprof --events all --metrics all ./print.exe ./bunny.stl