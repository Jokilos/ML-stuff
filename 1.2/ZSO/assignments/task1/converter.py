#!/usr/bin/env python
# coding: utf-8

import capstone
import keystone
import shutil
import struct
import re

def make_idx_dict(names):
    d = {} 
    for i, n in enumerate(names):
        d[n] = i

    return d

class Const:
    HEADER_SIZE = 64 

class ElfHeader:
    unpacked_data = None

    fields = [
        "e_ident",
        "e_type",
        "e_machine",
        "e_version",
        "e_entry",
        "e_phoff",
        "e_shoff",
        "e_flags",
        "e_ehsize",
        "e_phentsize",
        "e_phnum",
        "e_shentsize",
        "e_shnum",
        "e_shstrndx",
    ]

    idx_dict = make_idx_dict(fields)

    format = (
        # e_ident (16 bytes), e_type (2 bytes), e_machine (2 bytes), e_version (4 bytes)
        '< 16s H H I' +
        # e_entry (8 bytes), e_phoff (8 bytes), e_shoff (8 bytes), e_flags (4 bytes)
        'Q Q Q I' +
        # e_ehsize (2 bytes), e_phentsize (2 bytes), e_phnum (2 bytes), e_shentsize (2 bytes)
        'H H H H' +
        # e_shnum (2 bytes), e_shstrndx (2 bytes)
        'H H'
    )

    def print():
        for i, f in enumerate(ElfHeader.fields):
            print(f'{f}: {ElfHeader.unpacked_data[i]}')
    
    def read_elf_header():
        if ElfFile.data[:4] != b'\x7fELF':
            raise ValueError("Not a valid ELF file.")
        
        ElfHeader.unpacked_data = list(struct.unpack(ElfHeader.format, ElfFile.data[:Const.HEADER_SIZE]))

    def overwrite_elf_header(file_path):
        amd_machine = 0x003e
        ElfHeader.unpacked_data[2] = amd_machine

        packed_data = struct.pack(ElfHeader.format, *ElfHeader.unpacked_data)

        with open(file_path, 'wb') as f:
            f.write(packed_data)
            f.write(ElfFile.data[Const.HEADER_SIZE:])

    def get(name):
        idx = ElfHeader.idx_dict[name]
        return ElfHeader.unpacked_data[idx]

class SectionHeader:
    fields = [
        "sh_name",      # Section name (index into section header string table)
        "sh_type",      # Section type
        "sh_flags",     # Section attributes
        "sh_addr",      # Virtual address in memory
        "sh_offset",    # Offset in file
        "sh_size",      # Size of section
        "sh_link",      # Link to other section
        "sh_info",      # Miscellaneous information
        "sh_addralign", # Address alignment boundary
        "sh_entsize"    # Size of entries, if section has table
    ]

    format = (
        # sh_name (4 bytes), sh_type (4 bytes), sh_flags (8 bytes), sh_addr (8 bytes)
        '< I I Q Q' +  
        # sh_offset (8 bytes), sh_size (8 bytes), sh_link (4 bytes), sh_info (4 bytes)
        'Q Q I I' +    
        # sh_addralign (8 bytes), sh_entsize (8 bytes)
        'Q Q' 
    )

    idx_dict = make_idx_dict(fields)

    shstroff = None
    
    def __init__(self, offset):
        self.unpacked_data = list(
            struct.unpack(
                SectionHeader.format, 
                ElfFile.data[offset : offset + ElfHeader.get('e_shentsize')],
            )
        )

        sh_off = self.get('sh_offset')
        self.section_data = ElfFile.data[sh_off : sh_off + self.get('sh_size')]

        self.name = None

    def print(self):
        for i, f in enumerate(SectionHeader.fields):
            print(f'{f}: {self.data[i]}')

    def get(self, name):
        idx = SectionHeader.idx_dict[name]
        return self.unpacked_data[idx]
    
    def set_name(self, verbose = True):
        self.name = ElfFile.find_string(self.get('sh_name'))

        if verbose:
            print(self.name)

class Rela:
    size = 0x18

    format = '<QQq'  # r_offset, r_info, r_addend

    fields = ['r_offset', 'r_info', 'r_addend']

    idx_dict = make_idx_dict(fields)

    #define ELF64_R_SYM(i)((i) >> 32)
    def R_SYM(i):
        return i >> 32
    
    #define ELF64_R_TYPE(i)((i) & 0xf f f f f f f f L)
    def R_TYPE(i):
        return i & 0xffffffff

    #define ELF64_R_INFO(s, t)(((s) << 32) + ((t) & 0xf f f f f f f f L))
    def R_INFO(s, t):
        return s << 32 + t & 0xffffffff

    def __init__(self, offset):
        self.unpacked_data = struct.unpack(Rela.format, ElfFile.data[offset : offset + Rela.size])
        self.offset = offset
        info = self.get('r_info')
        self.sym = Rela.R_SYM(info)
        self.type = Rela.R_TYPE(info)

    def print(self):
        for i, f in enumerate(Rela.fields):
            print(f'{f}: {self.unpacked_data[i]}')

        print(f'{self.sym=}')
        print(f'{self.type=}')

    def overwrite_rela(self, offset):
        pass

    def get(self, name):
        idx = Rela.idx_dict[name]
        return self.unpacked_data[idx]

    def collect_rela_entries(sh):
        base_offset = offset = sh.get('sh_offset')
        size = sh.get('sh_size')
        entsize = sh.get('sh_entsize')
        rela_entries = []

        while offset < base_offset + size:
            rela_entries += [Rela(offset)]
            offset += entsize

        return rela_entries

class Sym:
    size = 0x18

    format = (
        # st_name(4 bytes), st_info(1 byte), st_other(1 byte)
        '< I B B' +
        # st_shndx(2 bytes), st_value(8 bytes), st_size(8 bytes)
        'H Q Q'
    )

    fields = [
        'st_name',
        'st_info',
        'st_other',
        'st_shndx',
        'st_value',
        'st_size',
    ]

    idx_dict = make_idx_dict(fields)

    def __init__(self, offset):
        self.unpacked_data = struct.unpack(Sym.format, ElfFile.data[offset : offset + Sym.size])
        self.offset = offset

        self.name = ElfFile.find_string(self.get('sh_name'))

        print(self.name)

    def print(self):
        for i, f in enumerate(Sym.fields):
            print(f'{f}: {self.unpacked_data[i]}')

    def overwrite_sym(self, offset):
        pass

    def get(self, name):
        idx = Sym.idx_dict[name]
        return self.unpacked_data[idx]

    def collect_sym_entries(sh):
        base_offset = offset = sh.get('sh_offset')
        size = sh.get('sh_size')
        entsize = sh.get('sh_entsize')
        sym_entries = []

        while offset < base_offset + size:
            sym_entries += [Sym(offset)]
            offset += entsize

        return sym_entries

class Comparator:
    # for capstone output
    hex_pattern = '0x[0-9a-f]+' 

    prolog = """
        stp x29, x30, [sp, #-prologue_shift]!
        mov x29, sp
    """

    epilog = """
        ldp x29, x30, [sp], #prologue_shift
        ret
    """

    def tokenize(text):
        filtered = filter(lambda x : x != '', text.strip().split())
        return list(filtered)

    def tokenize_pe(text):
        tokens = Comparator.tokenize(text)
        tokens = [s.replace('prologue_shift', Comparator.hex_pattern) for s in tokens]

        return tokens
    
    def compare_part(code, prolog = True):
        tokenized = \
            Comparator.tokenize_pe(Comparator.prolog if prolog else Comparator.epilog)
        
        tokens = len(tokenized)

        code_tokenized = Comparator.tokenize(code)

        if prolog:
            for i in range(tokens):
                pattern = re.compile(tokenized[i] + 'xd', re.IGNORECASE)
                m1 = pattern.search(code_tokenized[i])
                m2 = pattern.fullmatch(code_tokenized[i])
                print(m1, m2)

        print(tokenized)
        print(code_tokenized)

class Translator:

    def count_functions(code_section):
        code = Translator.disassemble_code(code_section, show_offsets = False)
        Comparator.compare_part(code)

    def disassemble_code(code_section, show_offsets = True, rela_section = None, verbose = True):
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

    def assemble_code(code):
        # separate assembly instructions by ; or \n
        CODE = b"INC ecx; DEC edx"
        
        try:
            ks = keystone.Ks(keystone.KS_ARCH_X86, keystone.KS_MODE_64)
            encoding, count = ks.asm(CODE)
            print("%s = %s (number of statements: %u)" %(CODE, encoding, count))

        except keystone.KsError as e:
            print("ERROR: %s" %e)

class ElfFile:
    data = None
    section_headers = []
    shstroff = None
    symtab = None
    rela_dict = {}

    def setup(file_path):
        with open(file_path, 'rb') as f:
            ElfFile.data = f.read()

    def read_elf_header():
        ElfHeader.read_elf_header()

    def find_string(relative_offset):
        str_offset = relative_offset + ElfFile.shstroff 
        str_end = ElfFile.data.find(b'\x00', str_offset)
        str_len = str_end - str_offset

        str = struct.unpack(f'{str_len}s', ElfFile.data[str_offset : str_end])[0]

        return str
        
    def read_section_headers():
        for i in range(ElfHeader.get('e_shnum')):
            offset = ElfHeader.get('e_shoff') + i * ElfHeader.get('e_shentsize')

            ElfFile.section_headers += [SectionHeader(offset)]
        
        shstrns = ElfFile.section_headers[ElfHeader.get('e_shstrndx')]
        ElfFile.shstroff = shstrns.get('sh_offset')

        for sh in ElfFile.section_headers:
            sh.set_name()
            if (sh.name == b'.symtab'):
                ElfFile.symtab = sh

    def look_for_section(name):
        for sh in ElfFile.section_headers:
            if sh.name == name:
                return sh

    def find_code_sections():
        for sh in ElfFile.section_headers: 
            if sh.get('sh_type')== 1:  # SHT_PROGBITS 
                functions = Translator.count_functions(sh.section_data)
                assert False
                rela = ElfFile.look_for_section(b'.rela' + sh.name)

                if rela:
                    rela_entries = Rela.collect_rela_entries(rela)

                    for re in rela_entries:
                        re.print()

                print(rela.name if rela else '')

input = 'test-aarch64.o'  
output = 'out.o'
good_output = 'test-aarch64-x64.o'  

shutil.copy(input, output)

ElfFile.setup(input)
ElfFile.read_elf_header()
ElfFile.read_section_headers()

ElfFile.find_code_sections()

# disassemble_code(section)
# assemble_code(None)

