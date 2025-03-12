CFLAGS = -static -g
TFLAGS = -no-pie -mcmodel=large -fno-pie

.PHONY: call tod fork

call:
	gcc call.c $(CFLAGS) -o call

tod:
	gcc tod.c $(CFLAGS) -o tod 

fork:
	gcc fork.c $(CFLAGS) -o fork 

look:
	@ $(file) | vim "+set ft=asm" -

task:
	gcc snp.c -c $(TFLAGS) snp.o
