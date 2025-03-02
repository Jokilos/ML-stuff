all:
	g++ -fopenmp hello_world.cpp -o hello_world
	g++ -fopenmp mat_mult.cpp -o mat_mult 

run:
	OMP_NUM_THREADS=4 ./hello_world

mult:
	OMP_NUM_THREADS=8 ./mat_mult
