import struct
from tools import dict_safe_get, make_idx_dict, two_way_dict, overwrite_file
from elf_file import ElfFile

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

    stbsym = two_way_dict({
        'STB_LOCAL' : 0,
        'STB_GLOBAL' : 1,
        'STB_WEAK' : 2,
    })

    sttsym = two_way_dict({
        'STT_NOTYPE' : 0,
        'STT_OBJECT' : 1,
        'STT_FUNC' : 2,
        'STT_SECTION' : 3,
        'STT_FILE' : 4,
    })
        
    idx_dict = make_idx_dict(fields)

    def __init__(self, offset, verbose = False):
        self.unpacked_data = list(
            struct.unpack(Sym.format, ElfFile.data[offset : offset + Sym.size])
        )
        self.offset = offset

        self.name = ElfFile.find_string(self.get('st_name'), sh_string = False)
        self.binding, self.type = Sym.get_binding_and_type(self.get('st_info'))

        if(verbose):
            self.print()

    @staticmethod
    def get_binding_and_type(st_info):
        binding = (st_info >> 4) & 0xF  
        symbol_type = st_info & 0xF
        
        return dict_safe_get(Sym.stbsym, binding), dict_safe_get(Sym.sttsym, symbol_type)

    def print(self):
        for i, f in enumerate(Sym.fields):
            print(f'{f}: {self.unpacked_data[i]}')

        print(self.name, self.binding, self.type)

    def get(self, name):
        idx = Sym.idx_dict[name]
        return self.unpacked_data[idx]

    def set(self, name, value):
        idx = Sym.idx_dict[name]
        self.unpacked_data[idx] = value

    @staticmethod
    def collect_sym_entries(sh):
        base_offset = offset = sh.get('sh_offset')
        size = sh.get('sh_size')
        entsize = sh.get('sh_entsize')
        sym_entries = []

        while offset < base_offset + size:
            sym_entries += [Sym(offset)]
            offset += entsize

        return sym_entries

    @staticmethod
    def save(file, sh, rela_entries):
        data = b''

        for re in rela_entries:
            data += struct.pack(Sym.format, *re.unpacked_data)

        overwrite_file(file, sh.get('sh_offset'), data)