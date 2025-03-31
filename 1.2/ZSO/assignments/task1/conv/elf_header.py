import struct
from tools import make_idx_dict, Const
from elf_file import ElfFile

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