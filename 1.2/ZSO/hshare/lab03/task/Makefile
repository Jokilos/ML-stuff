CFLAGS = -static -g
TFLAGS = -no-pie -mcmodel=large -fno-pie -g

.PHONY: call tod fork task

call:
	gcc call.c $(CFLAGS) -o call

tod:
	gcc tod.c $(CFLAGS) -o tod 

fork:
	gcc fork.c $(CFLAGS) -o fork 

look:
	@ $(file) | vim "+set ft=asm" -

task:
	gcc snp.c -c $(TFLAGS)
	gcc task.c $(TFLAGS) -o task
	gcc exp.c $(TFLAGS) -o exp
