import capstone
import keystone
from comparator import Comparator
from parse_insn import ParseInsn

class Translator:
    prolog_x86 = """
    push rbp
    mov rbp, rsp
    sub rsp, #prologue_shift
    """

    epilog_x86 = """
    mov rax, rdi
    leave
    ret
    """

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
    
    def translate_code(code_section, p_shift, rela_section = None):
        code = Translator.disassemble_code(code_section, True).splitlines()
        code_x86 = Translator.prolog_x86.replace('#prologue_shift', p_shift)

        md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
        md.detail = True
        instructions = md.disasm(code_section, 0)

        fields = [
        'address', 'bytes', 'errno', 'group', 'group_name', 'groups', 'id',
        'insn_name', 'mnemonic', 'op_count', 'op_find', 'op_str', 'reg_name',
        'reg_read', 'reg_write', 'regs_access', 'regs_read', 'regs_write', 'size'
        ]

        inst_list = []
        for insn in instructions:
            inst_list += [insn]

        for insn in inst_list[2:-2]:
            off = f"0x{insn.address:x}:\t" 

            code_line = f"{off}{insn.mnemonic}\t{insn.op_str}"
            code_line_x86 = ParseInsn.parse(insn, rela_section)

            code += code_line + "\n"
            code_x86 += code_line_x86 

            print(code_line)
            print(code_line_x86)

            # print(ParseInsn.parse(insn, rela_section))

        return code_x86 + Translator.epilog_x86

    def assemble_code(code):
        # separate assembly instructions by ; or \n
        CODE = b"INC ecx; DEC edx"
        
        try:
            ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
            encoding, count = ks.asm(CODE)
            print("%s = %s (number of statements: %u)" %(CODE, encoding, count))

        except keystone.KsError as e:
            print("ERROR: %s" %e)