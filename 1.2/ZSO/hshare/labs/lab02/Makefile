CFLAGS = -no-pie -march=haswell -fno-asynchronous-unwind-tables \
-fcf-protection=none -fno-stack-protector -masm=intel

# CFLAGS += -fpic 
# CFLAGS += -fno-pic 
# CFLAGS += -fno-plt 
# CFLAGS += -mcmodel=large

.PHONY: parts

parts:
	gcc -c part_a.c -o data/part_a.o $(CFLAGS)
	gcc -c part_b.c -o data/part_b.o $(CFLAGS)

	gcc data/part_a.o data/part_b.o -o data/parts $(CFLAGS)  -Wl,-emit-relocs -v

	# ld data/part_a.o data/part_b.o -o data/parts

paoo:
	objdump -dr data/part_a.o -M intel

pboo:
	objdump -dr data/part_b.o -M intel

po:
	objdump -dr -M intel data/parts

libtroll: 
	gcc -c libtroll.c -o libtroll.s $(CFLAGS)
