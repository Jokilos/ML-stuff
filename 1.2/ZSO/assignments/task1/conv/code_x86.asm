push rbp
mov rbp, rsp
sub rsp, 0x30
mov dword ptr [esp + 0x1c], edi
mov dword ptr [esp + 0x18], esi
mov esi, dword ptr [esp + 0x1c]
mov edi, dword ptr [esp + 0x18]
add edi, esi
mov dword ptr [esp + 0x2c], edi
mov edi, dword ptr [esp + 0x1c]
call 0x7fffffff
mov rdi, rax
lea rdi, [rip + 0x7fffffff]
and rdi, ~0xfff
add rdi, 0
call 0x7fffffff
mov rdi, rax
mov edi, dword ptr [esp + 0x18]
call 0x7fffffff
mov rdi, rax
lea rdi, [rip + 0x7fffffff]
and rdi, ~0xfff
add rdi, 0
call 0x7fffffff
mov rdi, rax
mov edi, dword ptr [esp + 0x2c]
call 0x7fffffff
mov rdi, rax
lea rdi, [rip + 0x7fffffff]
and rdi, ~0xfff
add rdi, 0
call 0x7fffffff
mov rdi, rax
mov edi, dword ptr [esp + 0x2c]
mov rax, rdi
leave
ret
