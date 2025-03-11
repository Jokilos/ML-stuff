CFLAGS = -no-pie -march=haswell -fno-asynchronous-unwind-tables -fcf-protection=none -fno-stack-protector

.PHONY: p1 p3 

p1re:
	gcc p1.c -o p1
	readelf -a p1
	
p1o:
	gcc p1.c -o p1
	objdump -xtrds p1

pse:
	gcc p1.c -o p1s -static
	readelf -a p1s

pso:
	gcc p1.c -o p1s -static
	objdump -xtrds p1s