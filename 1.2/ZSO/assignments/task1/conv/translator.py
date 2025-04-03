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
                code_line = f"{off} {insn.bytes} {insn.mnemonic}\t{insn.op_str}"
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

            if code_line_x86[0] == 'j':
                inst = code_line_x86[:3].strip()
                code_line_x86_asm = f'{inst} .jmp_label_{label_num}\n'
                label_num += 1
                code_line_x86 = f'{inst} 0xffffffff\n'
            else:
                code_line_x86_asm = code_line_x86

            _, line_size = Translator.assemble_code(code_line_x86)

            # print(f'{insn.address}: {code_line_x86_asm}')
            code_x86 += code_line_x86_asm
            code_x86_size += line_size

        arm_code_size = len(lines) * 4 + 8 
        if arm_code_size in jump_dict.keys():
            code_x86 += jump_dict[arm_code_size]

        return code_x86, code_x86_size

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

        print(f'{code_x86_size=}')
        print(jump_dict)

        code_x86_bytes, final_size = Translator.assemble_code(code_x86)

        Translator.disassemble_code(code_x86_bytes, x86=True, show_bytes=True, verbose=True)

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

