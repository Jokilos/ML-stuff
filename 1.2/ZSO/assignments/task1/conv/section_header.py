import re
import struct
from tools import two_way_dict, make_idx_dict, overwrite_file
from elf_file import ElfFile
from elf_header import ElfHeader

class SectionHeader:
    delete_pattern = \
        re.compile(r'(\.note\.gnu\.property|\.eh_frame)', 0)

    sh_types = two_way_dict({
        'SHT_NULL' : 0, 
        'SHT_PROGBITS' : 1, 
        'SHT_SYMTAB' : 2, 
        'SHT_STRTAB' : 3, 
        'SHT_RELA' : 4, 
        'SHT_HASH' : 5, 
        'SHT_DYNAMIC' : 6, 
        'SHT_NOTE' : 7, 
        'SHT_NOBITS' : 8, 
    })

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
    
    def __init__(self, offset, verbose = False):
        self.unpacked_data = list(
            struct.unpack(
                SectionHeader.format, 
                ElfFile.data[offset : offset + ElfHeader.get('e_shentsize')],
            )
        )

        sh_off = self.get('sh_offset')
        self.section_data = ElfFile.data[sh_off : sh_off + self.get('sh_size')]

        self.type = SectionHeader[self.get('sh_type')] \
            if self.get('sh_type') in SectionHeader.sh_types else "UNKNOWN"

        self.name = None
        self.is_shstrs = False
        self.is_expanded = False

        if verbose:
            self.print()

    def print(self):
        print(self.name)
        print(self.type)
        for i, f in enumerate(SectionHeader.fields):
            print(f'{f}: {self.unpacked_data[i]}')

    def get(self, name):
        idx = SectionHeader.idx_dict[name]
        return self.unpacked_data[idx]
    
    def set(self, name, value):
        idx = SectionHeader.idx_dict[name]
        self.unpacked_data[idx] = value

    def set_name(self, verbose = False):
        self.name = ElfFile.find_string(self.get('sh_name'), True)

        if verbose:
            print(self.name)

    def save_section(self, file, offset):
        overwrite_file(file, offset, self.section_data)
        self.set('sh_size', len(self.section_data))

        return offset + len(self.section_data)
    
    def save(self, file, offset):
        packed_data = struct.pack(SectionHeader.format, *self.unpacked_data)
        overwrite_file(file, offset, packed_data)

        return offset + len(packed_data)