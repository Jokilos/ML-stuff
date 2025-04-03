run:
	gcc example.user.c -o example.user -lbpf
	./example.user
genskel:
	bpftool gen skeleton  example.bpf.o name example > example.skel.h
all:
	clang --target=bpf -g -Og -c  example.bpf.c -o example.bpf.o
