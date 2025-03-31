import struct
from tools import two_way_dict, make_idx_dict
from elf_file import ElfFile

class Rela:
    code_types = two_way_dict({
        'R_AARCH64_CALL26' : 283,
        'R_AARCH64_ADD_ABS_LO12_NC' : 277,	
        'R_AARCH64_ADR_PREL_PG_HI21' : 275,
        'R_AARCH64_ABS64' : 257,
        # not overlapping, so i keep it in the same dict
        'R_AMD64_64' : 1,
        'R_AMD64_PC32' : 2,
        'R_AMD64_32' : 10,
    })

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
        return (s << 32) + (t & 0xffffffff)

    def __init__(self, offset, verbose = False):
        self.unpacked_data = list(
            struct.unpack(Rela.format, ElfFile.data[offset : offset + Rela.size])
        )
        self.offset = offset
        info = self.get('r_info')
        self.sym = Rela.R_SYM(info)
        self.type = Rela.code_types[Rela.R_TYPE(info)]

        if verbose:
            self.print()

    def print(self):
        for i, f in enumerate(Rela.fields):
            print(f'{f}: {self.unpacked_data[i]}')

        print(f'{self.sym=}')
        print(f'{self.type=}')

    def overwrite_rela(
            self, 
            offset = None, 
            symbol = None, 
            type = None, 
            addend = None,
        ):
        if offset:
            self.unpacked_data[0] = offset
        if symbol:
            self.sym = symbol
        if type:
            self.type = type 
        if addend:
            self.unpacked_data[2] = addend

        self.unpacked_data[1] = Rela.R_INFO(self.sym, Rela.code_types[type])
        
    def get(self, name):
        idx = Rela.idx_dict[name]
        return self.unpacked_data[idx]

    def collect_rela_entries(sh):
        base_offset = offset = sh.get('sh_offset')
        size = sh.get('sh_size')
        entsize = sh.get('sh_entsize')
        rela_entries = {}

        while offset < base_offset + size:
            rela = Rela(offset)
            rela_entries[rela.get('r_offset')] = rela
            offset += entsize

        return rela_entries
