import capstone
import keystone
import copy
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

    def count_functions(code_section):
        code = Translator.disassemble_code(code_section, show_offsets = False)
        return Comparator.check_function(code)

    def translate_lines(
            lines : list[capstone.CsInsn],
            lines_86 : list[capstone.CsInsn],
            rela_section,
            jump_to,
        ):

        idx86 = 3
        code_x86_size = lines_86[idx86].address
        code_x86 = '.jmp_label:\n' + Translator.prolog_x86
        code_size = 8
        offset_dict = {0 : 0, code_size : code_x86_size}
        fixlines = []
        def linecount(x) : return len(x.splitlines())
        def sign(x) : return hex(x) if x < 0 else f'+{hex(x)}'

        for insn in lines[2:-2]:
            if rela_section and insn.address in rela_section.keys():
                rela_section[insn.address].overwrite_rela(offset = code_x86_size)

            code_line_x86 = ParseInsn.parse(insn, rela_section)

            if code_line_x86[0] == 'j':
                mnem = code_line_x86[:3].strip()
                base_offset = lines_86[idx86].address
                code_line_x86 = f'{mnem} {hex(base_offset)}\n'
                fixlines += [idx86 + 1]

            lines = linecount(code_line_x86)
            idx86 += lines
            code_x86 += code_line_x86
            code_x86_size = lines_86[idx86].address
            code_size += 4

            offset_dict[code_size] = code_x86_size

        code_x86 += Translator.epilog_x86

        split = code_x86.splitlines()
        for i, line in enumerate(fixlines):
            mnem = split[line][:3].strip()
            off_from = int(split[line][3:].strip(), base = 16)
            off_to = jump_to[i]

            if mnem == 'jmp':
                # split[line] = f'{mnem} {hex(off_to - off_from)}]'
                split[line] = f'{mnem} {hex(off_to)}'
            else:
                split[line] = f'{mnem} {hex(off_to)}'
        
        return '\n'.join(split), lines_86[-1].address + lines_86[-1].size

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

    def assemble_whole(inst_list, rela_dict):
        code_x86 = Translator.prolog_x86

        jump_to = [] 
        for insn in inst_list:
            if insn.mnemonic[0] == 'b':
                code_line_x86 = ParseInsn.parse(insn, None)
                to_insn = int(code_line_x86[3:].strip(), base = 16)
                jump_to += [to_insn] 

                mnem = code_line_x86[:3].strip()

                if mnem == 'jmp':
                    code_x86 += f'{mnem} [rip + 0x7fffffff]\n'
                else:
                    code_x86 += f'{mnem} 0x7fffffff\n'
            else:
                code_x86 += ParseInsn.parse(insn, rela_dict) 

        code_x86 += Translator.epilog_x86

        bytecode, _ = Translator.assemble_code(code_x86)

        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True
        instructions86 = md.disasm(bytecode, 0)

        inst_list86 = [i for i in instructions86]

        return inst_list86, jump_to 

    def translate_code(
            code_section,
            p_shift,
            rela_section : dict[int, Rela] = None,
            verbose = False,
        ):
        Translator.prolog_x86 = Translator.prolog_x86.replace('#prologue_shift', p_shift)
        md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
        md.detail = True
        instructions = md.disasm(code_section, 0)

        inst_list = [i for i in instructions]

        inst_list86, jump_to = Translator.assemble_whole(
            inst_list[2:-2],
            copy.deepcopy(rela_section),
        )

        code86, code86len = Translator.translate_lines(
            inst_list,
            inst_list86,
            rela_section,
            jump_to,
        )

        # bytecode, _ = Translator.debug_assemble(code86) 
        bytecode, bcodelen = Translator.assemble_code(code86) 
        bytecode, bcodelen = Translator.pad_jumps(bytecode) 

        Translator.disassemble_code(
            bytecode,
            show_offsets=True,
            x86=True,
            show_bytes=True,
            verbose=True,
        )
        for i in inst_list86:
            print(hex(i.address), i.mnemonic)

        assert bcodelen == code86len, (bcodelen, code86len) 
        return bytecode, bcodelen

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
                print(l, end='')
                encoding, count = ks.asm(l)
                print(f' = {encoding}')

        except keystone.KsError as e:
            print("ERROR: %s" %e)

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

    # def translate_code_old(
    #         code_section,
    #         p_shift,
    #         rela_section : dict[int, Rela] = None,
    #         verbose = False,
    #     ):
    #     md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
    #     md.detail = True
    #     instructions = md.disasm(code_section, 0)

    #     inst_list = []
    #     for insn in instructions:
    #         inst_list += [insn]

    #     jump_dict = Translator.jump_dict_gen(inst_list[2:-2])

    #     code_x86, code_x86_size = Translator.translate_lines(
    #         inst_list[2:-2],
    #         p_shift,
    #         jump_dict,
    #         rela_section
    #     )

    #     _, epi_size = Translator.assemble_code(Translator.epilog_x86)

    #     code_x86 += Translator.epilog_x86 
    #     code_x86_size += epi_size

    #     if verbose:
    #         print(f'{code_x86_size=}')
    #         print(jump_dict)

    #     print(code_x86)

    #     code_x86_bytes, _ = Translator.assemble_code(code_x86)

    #     code_x86_bytes, final_size = Translator.pad_jumps(code_x86_bytes)

    #     # Translator.disassemble_code(code_x86_bytes, x86=True, show_bytes=True, verbose=True)

    #     assert final_size == code_x86_size, (final_size, code_x86_size)

    #     return code_x86_bytes, code_x86_size

