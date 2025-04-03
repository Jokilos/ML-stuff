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

    def disassemble_code(code_section, show_offsets = True, verbose = False):
        # AArch64 architecture
        md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
        instructions = md.disasm(code_section, 0)

        code = ""
        for insn in instructions:
            off = f"0x{insn.address:x}:\t" if show_offsets else ""
            code_line = f"{off}{insn.mnemonic}\t{insn.op_str}"

            code += code_line + "\n"

            if verbose:
                print(code_line)
        
        return code
    
    def translate_code(
            code_section,
            p_shift,
            rela_section : dict[int, Rela] = None,
            verbose = False,
        ):

        code_x86 = Translator.prolog_x86.replace('#prologue_shift', p_shift)
        code_x86_bytes, code_x86_size = Translator.assemble_code(code_x86)
        code_arm_size = 8
        
        md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
        md.detail = True
        instructions = md.disasm(code_section, 0)

        inst_list = []
        for insn in instructions:
            inst_list += [insn]

        for insn in inst_list[2:-2]:
            # print(code_x86_size)

            if rela_section and code_arm_size in rela_section.keys():
                rela_section[code_arm_size].overwrite_rela(offset = code_x86_size)

            code_line_x86 = ParseInsn.parse(insn, rela_section)
            line_bytes, line_size = Translator.assemble_code(code_line_x86)

            code_x86 += code_line_x86 
            code_x86_bytes += line_bytes
            code_x86_size += line_size
            code_arm_size += 4

        return (
            code_x86 + Translator.epilog_x86,
            code_x86_bytes + Translator.assemble_code(Translator.epilog_x86)[0],
        )

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
            print("ERROR: %s" %e)

    def debug_assemble(code):
        code = code.strip()

        try:
            ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)

            for l in code.splitlines():
                encoding, count = ks.asm(l)
                print(f'{l} = {encoding}')

        except keystone.KsError as e:
            print("ERROR: %s" %e)

