import capstone
import keystone
from comparator import Comparator
from parse_insn import ParseInsn
from rela import Rela

class Translator:
    prolog_x86 = """
    push rbp
    mov rbp, rsp
    sub rsp, #prologue_shift
    """.replace('    ', '').strip() + '\n'

    epilog_x86 = """
    mov rax, rdi
    leave
    ret
    """.replace('    ', '').strip() + '\n'

    # insn_fields = [
        # 'address', 'bytes', 'errno', 'group', 'group_name', 'groups', 'id',
        # 'insn_name', 'mnemonic', 'op_count', 'op_find', 'op_str', 'reg_name',
        # 'reg_read', 'reg_write', 'regs_access', 'regs_read', 'regs_write', 'size'
    # ]

    def count_functions(code_section):
        code = Translator.disassemble_code(code_section, show_offsets = False)
        return Comparator.check_function(code)

    def disassemble_code(
            code_section, 
            show_offsets = True, 
            x86 = False,
            show_bytes = False,
            verbose = False,
        ):
        # AArch64 architecture
        if x86:
            md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        else:
            md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)

        instructions = md.disasm(code_section, 0)

        code = ""
        for insn in instructions:
            off = f"0x{insn.address:x}:\t" if show_offsets else ""
            if show_bytes:
                code_line = f"{off} {bytes(insn.bytes)} \t\t\t {insn.mnemonic}\t{insn.op_str}"
            else:
                code_line = f"{off} {insn.mnemonic}\t{insn.op_str}"

            code += code_line + "\n"

            if verbose:
                print(code_line)
        
        return code

    def jump_dict_gen(lines):
        jump_dict = {}
        label_num = 0
        for insn in lines:
            if insn.mnemonic[0] == 'b':
                code_line_x86 = ParseInsn.parse(insn, None)
                disp = int(code_line_x86[3:].strip(), base = 16)
                jump_dict[disp] = f'.jmp_label_{label_num}:\n'
                label_num += 1

        return jump_dict

    def translate_lines(lines : list[capstone.CsInsn], p_shift, jump_dict, rela_section):
        label_num = 0
        code_x86 = Translator.prolog_x86.replace('#prologue_shift', p_shift)
        _, code_x86_size = Translator.assemble_code(code_x86)

        for insn in lines:
            if rela_section and insn.address in rela_section.keys():
                rela_section[insn.address].overwrite_rela(offset = code_x86_size)

            code_line_x86 = ParseInsn.parse(insn, rela_section)

            if insn.address in jump_dict.keys():
                code_x86 += jump_dict[insn.address]
                # print(jump_dict[insn.address])

            if code_line_x86[0] == 'j':
                inst = code_line_x86[:3].strip()
                code_line_x86_asm = f'{inst} .jmp_label_{label_num}\n'
                label_num += 1
                code_line_x86 = f'{inst} 0xffffffff\n'
            else:
                code_line_x86_asm = code_line_x86

            _, line_size = Translator.assemble_code(code_line_x86)

            # print(f'{hex(insn.address)}: {code_line_x86_asm}')
            code_x86 += code_line_x86_asm
            code_x86_size += line_size

        arm_code_size = len(lines) * 4 + 8 
        if arm_code_size in jump_dict.keys():
            code_x86 += jump_dict[arm_code_size]

        return code_x86, code_x86_size

    def pad_jumps(code_x86):
        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        instructions = md.disasm(code_x86, 0)

        def pad(b, num):
            return b + b'\x90' * (num - len(b))
                
        bytes_x86  = b'' 
        for insn in instructions:
            mnem = insn.mnemonic
            if mnem[:3] == 'jmp':
                bytes_x86 += pad(bytes(insn.bytes), 5)
            elif mnem[0] == 'j':
                bytes_x86 += pad(bytes(insn.bytes), 6)
            else:
                bytes_x86 += bytes(insn.bytes)
                
        return bytes_x86, len(bytes_x86) 

    def translate_code(
            code_section,
            p_shift,
            rela_section : dict[int, Rela] = None,
            verbose = False,
        ):
        md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
        md.detail = True
        instructions = md.disasm(code_section, 0)

        inst_list = []
        for insn in instructions:
            inst_list += [insn]

        jump_dict = Translator.jump_dict_gen(inst_list[2:-2])

        code_x86, code_x86_size = Translator.translate_lines(
            inst_list[2:-2],
            p_shift,
            jump_dict,
            rela_section
        )

        _, epi_size = Translator.assemble_code(Translator.epilog_x86)

        code_x86 += Translator.epilog_x86 
        code_x86_size += epi_size

        if verbose:
            print(f'{code_x86_size=}')
            print(jump_dict)

        print(code_x86)

        code_x86_bytes, _ = Translator.assemble_code(code_x86)

        code_x86_bytes, final_size = Translator.pad_jumps(code_x86_bytes)

        # Translator.disassemble_code(code_x86_bytes, x86=True, show_bytes=True, verbose=True)

        assert final_size == code_x86_size, (final_size, code_x86_size)

        return code_x86_bytes, code_x86_size

    # if verbose:
    #     off = f"0x{insn.address:x}:\t" 
    #     code_line = f"{off}{insn.mnemonic}\t{insn.op_str}"
    #     print(code_line)
    #     print(code_line_x86)

    def assemble_code(code, verbose = False):
        # separate assembly instructions by ; or \n
        code = code.strip()

        try:
            ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
            encoding, count = ks.asm(code)

            if verbose:
                print(  "%s = %s (no.statements: %u) (no.bytes %u)"
                        %(code, encoding, count, len(encoding)))

            return bytes(encoding), len(encoding) 

        except keystone.KsError as e:
            print(f"ERROR: {e} \nCODE: {code}")

    def debug_assemble(code):
        code = code.strip()

        try:
            ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)

            for l in code.splitlines():
                encoding, count = ks.asm(l)
                print(f'{l} = {encoding}')

        except keystone.KsError as e:
            print("ERROR: %s" %e)

    code_correct='''.jmp_label_0:
push rbp
mov rbp, rsp
sub rsp, 0x40
mov qword ptr [rsp + 0x38], 0
mov qword ptr [rsp + 0x30], 0
mov qword ptr [rsp + 0x28], 0
jmp .jmp_label_0
.jmp_label_3:
mov rdi, qword ptr [rsp + 0x38]
mov qword ptr [rsp + 0x20], rdi
mov rdi, qword ptr [rsp + 0x20]
mov qword ptr [rsp + 0x30], rdi
mov rsi, qword ptr [rsp + 0x30]
mov rdi, qword ptr [rsp + 0x20]
add rdi, rsi
mov qword ptr [rsp + 0x28], rdi
mov rdi, qword ptr [rsp + 0x28]
mov qword ptr [rsp + 0x20], rdi
mov rdi, qword ptr [rsp + 0x20]
mov qword ptr [rsp + 0x30], rdi
mov rsi, qword ptr [rsp + 0x30]
mov rdi, qword ptr [rsp + 0x20]
add rdi, rsi
mov qword ptr [rsp + 0x28], rdi
mov rdi, qword ptr [rsp + 0x28]
mov qword ptr [rsp + 0x20], rdi
mov rdi, qword ptr [rsp + 0x20]
mov qword ptr [rsp + 0x30], rdi
mov rsi, qword ptr [rsp + 0x30]
mov rdi, qword ptr [rsp + 0x20]
add rdi, rsi
mov qword ptr [rsp + 0x28], rdi
mov rdi, qword ptr [rsp + 0x28]
mov qword ptr [rsp + 0x20], rdi
lea rdi, [rip + 0x7fffffff]
and rdi, ~0xfff
mov r10, 0x7fffffff
and r10, 0xfff
add rdi, r10
mov qword ptr [rsp + 0x18], rdi
mov rdi, qword ptr [rsp + 0x20]
mov rsi, qword ptr [rsp + 0x18]
add rdi, rsi
mov qword ptr [rsp + 0x10], rdi
mov rdi, qword ptr [rsp + 0x10]
mov rdi, qword ptr [rdi]
mov rsi, qword ptr [rsp + 0x38]
cmp rsi, rdi
je .jmp_label_1
mov edi, -1
jmp .jmp_label_2
.jmp_label_1:
mov rdi, qword ptr [rsp + 0x38]
add rdi, 1
mov qword ptr [rsp + 0x30], rdi
mov rdi, qword ptr [rsp + 0x30]
mov qword ptr [rsp + 0x38], rdi
lea rdi, [rip + 0x7fffffff]
and rdi, ~0xfff
mov r10, 0x7fffffff
and r10, 0xfff
add rdi, r10
mov rdi, qword ptr [rdi]
mov rsi, qword ptr [rsp + 0x38]
cmp rsi, rdi
jl .jmp_label_3
mov edi, 0
.jmp_label_2:
mov rax, rdi
leave
ret
'''