	.file	"p3.c"
	.intel_syntax noprefix
	.text
	.globl	f
	.type	f, @function
f:
	push	rbp
	mov	rbp, rsp
	mov	DWORD PTR -4[rbp], edi
	mov	eax, DWORD PTR -4[rbp]
	add	eax, eax
	pop	rbp
	ret
	.size	f, .-f
	.globl	g
	.type	g, @function
g:
	push	rbp
	mov	rbp, rsp
	mov	DWORD PTR -4[rbp], edi
	mov	DWORD PTR -8[rbp], esi
	mov	DWORD PTR -12[rbp], edx
	mov	eax, DWORD PTR -4[rbp]
	cmp	eax, DWORD PTR -8[rbp]
	jle	.L4
	mov	eax, 97
	jmp	.L5
.L4:
	mov	eax, DWORD PTR -8[rbp]
	cmp	eax, DWORD PTR -12[rbp]
	jle	.L6
	mov	eax, 98
	jmp	.L5
.L6:
	mov	eax, 99
.L5:
	pop	rbp
	ret
	.size	g, .-g
	.section	.rodata
.LC0:
	.string	"g(42, 42, 42) is %c\n" 	
	# comment heres the string used in main, i changed d to c
	.text
	.globl	main
	.type	main, @function
main:
	push	rbp
	mov	rbp, rsp
	mov	edi, 42
	mov	esi, edi 
	mov	edx, edi 
	# comment instead of f we call g
	call	g 	
	mov	esi, eax
	lea	rax, .LC0[rip]
	mov	rdi, rax
	mov	eax, 0
	call	printf@PLT
	mov	eax, 0
	pop	rbp
	ret
	.size	main, .-main
	.ident	"GCC: (GNU) 14.2.1 20250207"
	.section	.note.GNU-stack,"",@progbits
