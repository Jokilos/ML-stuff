import struct

class ElfFile:
    data = None

    section_headers = []
    symbols = []
    rela_dict = {}

    shstroff = None
    sh_dict = {}

    def setup(file_path):
        with open(file_path, 'rb') as f:
            ElfFile.data = f.read()

    def read_elf_header():
        from elf_header import ElfHeader
        ElfHeader.read_elf_header()

    def find_string(relative_offset, sh_string = False):
        str_offset = relative_offset

        if sh_string:
            str_offset += ElfFile.shstroff 
        else:
            str_offset += ElfFile.sh_dict['.strtab'].get('sh_offset')

        str_end = ElfFile.data.find(b'\x00', str_offset)
        str_len = str_end - str_offset

        str = struct.unpack(f'{str_len}s', ElfFile.data[str_offset : str_end])[0]

        return str
        
    def read_section_headers(verbose = False):
        from elf_header import ElfHeader
        from section_header import SectionHeader

        for i in range(ElfHeader.get('e_shnum')):
            offset = ElfHeader.get('e_shoff') + i * ElfHeader.get('e_shentsize')

            ElfFile.section_headers += [SectionHeader(offset)]
        
        shstrns = ElfFile.section_headers[ElfHeader.get('e_shstrndx')]
        ElfFile.shstroff = shstrns.get('sh_offset')

        null_sh = ElfFile.section_headers[0]

        for i in range(len(ElfFile.section_headers)):
            sh = ElfFile.section_headers[i]
            sh.set_name()

            decoded_name = sh.name.decode('utf-8')
            ElfFile.sh_dict[decoded_name] = sh

            if SectionHeader.delete_pattern.search(decoded_name):
                ElfFile.section_headers[i] = null_sh

            if verbose:
                ElfFile.section_headers[i].print()

    def read_symbols():
        from sym import Sym

        sym_sh = ElfFile.sh_dict['.symtab']
        ElfFile.symbols = Sym.collect_sym_entries(sym_sh)

    def read_rela():
        from rela import Rela

        for sh in ElfFile.section_headers:
            if (sh.type == 'SHT_RELA'):
                rela_entries = Rela.collect_rela_entries(sh)
                ElfFile.rela_dict[sh.name] = rela_entries
            
    def look_for_section(name):
        for sh in ElfFile.section_headers:
            if sh.name == name:
                return sh

    def find_code_sections():
        from translator import Translator

        for s in ElfFile.symbols:
            if s.type == 'STT_FUNC': 
                sh = ElfFile.section_headers[s.get('st_shndx')]

                offset = s.get('st_value')
                code = sh.section_data[offset : offset + s.get('st_size')]

                rela_name = b'.rela' + sh.name
                exists = rela_name in ElfFile.rela_dict.keys()
                rela = ElfFile.rela_dict[rela_name] if exists else None
                
                if p_shift := Translator.count_functions(code):
                    code_x86 = Translator.translate_code(code, p_shift, rela)