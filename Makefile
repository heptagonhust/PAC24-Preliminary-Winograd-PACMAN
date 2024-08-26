all:
	# gcc -std=c11 -fopenmp -O2 -I /shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/include -L /shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/lib/kblas/omp -lkblas -g driver.c ttt.c -o winograd
	# gcc -std=c11 -D__DEBUG -O0 -g driver.c winograd.c -o winograd
	# clang -mcpu=cortex-a72 --version
	clang -march=armv8-a+sve -Rpass-missed=loop-vectorize -I /shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/include -L /shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/lib/kblas/omp -lkblas -std=c11 -fopenmp -O3  driver.c ttt.c -o winograd 

