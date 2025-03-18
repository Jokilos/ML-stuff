CFLAGS = -no-pie -march=haswell -fno-asynchronous-unwind-tables -fcf-protection=none -fno-stack-protector

.PHONY: p1 p3 

p1re:
	gcc p1.c -o data/p1
	readelf -a data/p1
	
p1o:
	gcc p1.c -o data/p1
	objdump -xtrds data/p1

pse:
	gcc p1.c -o data/p1s -static
	readelf -a data/p1s

pso:
	gcc p1.c -o data/p1s -static
	objdump -xtrds data/p1s