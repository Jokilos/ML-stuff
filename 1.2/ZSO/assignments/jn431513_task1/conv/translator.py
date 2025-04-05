import capstone
import keystone
import copy
import struct
from comparator import Comparator
from parse_insn import ParseInsn
from rela import Rela


class Translator:
    prolog_x86_template = """
    push rbp
    mov rbp, rsp
    sub rsp, #prologue_shift
    """.replace('    ', '').strip() + '\n'

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

    # Checks if a function is up to assignment assumptions and returns prologue_shift
    @staticmethod
    def check_function(code_section):
        code = Translator.disassemble_code(code_section, show_offsets = False)
        return Comparator.check_function(code)

    # Fix relocations and jumps offsets, produce final translated bytecode
    @staticmethod
    def translate_lines(
            lines : list[capstone.CsInsn],
            lines_86 : list[capstone.CsInsn],
            rela_section,
            jump_to,
        ):

        idx86 = 3
        code_x86_size = lines_86[idx86].address
        code_x86 = Translator.prolog_x86
        code_size = lines[2].address
        offset_dict = {0 : 0, code_size : code_x86_size}
        fixlines = []
        linebytes = [i.bytes for i in lines_86]
        def linecount(x) : return len(x.splitlines())

        for insn in lines[2:-2]:
            if rela_section and insn.address in rela_section.keys():
                rela_section[insn.address].overwrite_rela(offset = code_x86_size)

            code_line_x86 = ParseInsn.parse(insn, rela_section)

            if code_line_x86[0] == 'j':
                mnem = code_line_x86[:3].strip()
                base_offset = lines_86[idx86].address
                code_line_x86 = f'{mnem} {base_offset} {idx86}\n'
                fixlines += [idx86]

            lines = linecount(code_line_x86)
            idx86 += lines
            code_x86 += code_line_x86
            code_x86_size = lines_86[idx86].address
            code_size += 4

            offset_dict[code_size] = code_x86_size

        code_x86 += Translator.epilog_x86

        code_split = code_x86.splitlines()
        for i, line in enumerate(fixlines):
            mnem, off_from, idx86 = code_split[line].split()
            idx86 = int(idx86)
            insn = lines_86[int(idx86)]

            off_from = int(off_from) + len(insn.bytes)
            off_to = offset_dict[jump_to[i]]
            offset = off_to - off_from
            offset_packed = struct.pack('<i', offset)

            linebytes[idx86]= insn.bytes[:insn.imm_offset] + offset_packed

        bytecode = b''.join(linebytes)
        
        return bytecode, len(bytecode) 

    # Assemble translated code and dissasemble it back 
    @staticmethod
    def assemble_whole(inst_list, offset_x86, rela_dict):
        code_x86 = Translator.prolog_x86

        jump_to = [] 
        for insn in inst_list:
            if insn.mnemonic[0] == 'b' and insn.mnemonic[:2] != 'bl':
                code_line_x86 = ParseInsn.parse(insn, None)
                to_insn = int(code_line_x86[3:].strip(), base = 16)
                jump_to += [to_insn] 

                mnem = code_line_x86[:3].strip()
                code_x86 += f'{mnem} 0x7fffffff\n'
            else:
                code_x86 += ParseInsn.parse(insn, rela_dict) 

        code_x86 += Translator.epilog_x86

        bytecode, _ = Translator.assemble_code(code_x86)

        md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        md.detail = True
        instructions86 = md.disasm(bytecode, offset_x86)

        inst_list86 = [i for i in instructions86]

        return inst_list86, jump_to 

    # Translare AARCH64 code section into AMD64
    @staticmethod
    def translate_code(
            code_section,
            p_shift,
            offset_arm,
            offset_x86,
            rela_section : dict[int, Rela] = None,
            verbose = False,
        ):
        Translator.prolog_x86 = Translator.prolog_x86.replace('#prologue_shift', p_shift)
        md = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
        md.detail = True
        instructions = md.disasm(code_section, offset_arm)

        inst_list = [i for i in instructions]

        inst_list86, jump_to = Translator.assemble_whole(
            inst_list[2:-2],
            offset_x86,
            copy.deepcopy(rela_section),
        )

        bytecode, bcodelen = Translator.translate_lines(
            inst_list,
            inst_list86,
            rela_section,
            jump_to,
        )
        
        if verbose:
            Translator.disassemble_code(
                bytecode,
                show_offsets=True,
                x86=True,
                show_bytes=True,
                verbose=True,
            )

        Translator.prolog_x86 = Translator.prolog_x86_template
        return bytecode, bcodelen
    
    # Assembles provided code into x86 bytecode and its length
    @staticmethod
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

    # Returns dissasembled code in a form of a string
    @staticmethod
    def disassemble_code(
            code_section, 
            show_offsets = True, 
            x86 = False,
            show_bytes = False,
            verbose = False,
        ):
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


