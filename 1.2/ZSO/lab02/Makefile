CFLAGS = -no-pie -march=haswell -fno-asynchronous-unwind-tables -fcf-protection=none -fno-stack-protector

.PHONY: parts

parts:
	gcc -c part_a.c -o part_a.o $(CFLAGS)
	gcc -c part_b.c -o part_b.o $(CFLAGS)

	gcc part_a.o part_b.o -o parts $(CFLAGS)  -Wl,-emit-relocs

	# ld part_a.o part_b.o -o parts
	# gcc part_a.o part_b.o -o parts $(CFLAGS)  -Wl,-emit-relocs -v

libtroll: 
	gcc -c libtroll.c -o libtroll.s $(CFLAGS)