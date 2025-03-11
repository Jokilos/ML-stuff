CFLAGS = -static -g

.PHONY: call tod fork

call:
	gcc call.c $(CFLAGS) -o call

tod:
	gcc tod.c $(CFLAGS) -o tod 

fork:
	gcc fork.c $(CFLAGS) -o fork 

look:
	@ $(file) | vim "+set ft=asm" -
