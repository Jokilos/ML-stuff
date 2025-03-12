input = """
   0:   55                      push   rbp
   1:   48 89 e5                mov    rbp,rsp
   4:   48 83 ec 10             sub    rsp,0x10
   8:   89 7d fc                mov    DWORD PTR [rbp-0x4],edi
   b:   ba 02 00 00 00          mov    edx,0x2
  10:   48 b8 00 00 00 00 00    movabs rax,0x0
  17:   00 00 00 
                        12: R_X86_64_64 .rodata
  1a:   48 89 c6                mov    rsi,rax
  1d:   bf 01 00 00 00          mov    edi,0x1
  22:   48 b8 00 00 00 00 00    movabs rax,0x0
  29:   00 00 00 
                        24: R_X86_64_64 write
  2c:   ff d0                   call   rax
  2e:   90                      nop
  2f:   c9                      leave
  30:   c3                      ret

"""

input = """
    1179:       48 89 fa                mov    rdx,rdi
    117c:       48 c7 c7 01 00 00 00    mov    rdi,0x1
"""

def comment(word):
    return "//" + word + "  "

output = ""
for line in input.splitlines():
    for i, word in enumerate(line.split()):
        if i == 0:
            output += "/*" + word + "*/  "

        elif word[0] == 'R':
            output += comment(word) 

        elif len(word) == 2:
            try:
                hex = '0x' + word
                int(hex, base = 16)
                output += hex + ", "
            
            except:
                output += comment(word) 

        else:
            output += comment(word) 

    output += "\n"

print(output)