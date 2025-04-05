import struct
import sys

class ElfFile:
    data = None

    section_headers = []
    new_section_headers = []
    symbols = []
    sh_idx_translation = []

    rela_dict = {}
    symbol_code_dict = {}
    sh_dict = {}

    shstroff = None
    symtab_idx = None

    def setup(file_path):
        with open(file_path, 'rb') as f:
            ElfFile.data = f.read()

    def read_elf_header():
        from elf_header import ElfHeader
        ElfHeader.read_elf_header()

    def read_section_headers(verbose = False):
        from elf_header import ElfHeader
        from section_header import SectionHeader

        for i in range(ElfHeader.get('e_shnum')):
            offset = ElfHeader.get('e_shoff') + i * ElfHeader.get('e_shentsize')

            ElfFile.section_headers += [SectionHeader(offset)]
        
        shstrs = ElfFile.section_headers[ElfHeader.get('e_shstrndx')]
        shstrs.is_shstrs = True
        ElfFile.shstroff = shstrs.get('sh_offset')

        for i in range(len(ElfFile.section_headers)):
            sh = ElfFile.section_headers[i]
            sh.set_name()

            decoded_name = sh.name.decode()
            ElfFile.sh_dict[decoded_name] = sh

            if verbose:
                ElfFile.section_headers[i].print()

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
        
    def read_symbols():
        from sym import Sym

        for i, sh in enumerate(ElfFile.section_headers):
            if (sh.type == 'SHT_SYMTAB'):
                ElfFile.symtab_idx = i
                ElfFile.symbols = Sym.collect_sym_entries(sh)

    def read_rela():
        from rela import Rela
        from section_header import SectionHeader

        for sh in ElfFile.section_headers:
            ignore_section = SectionHeader.delete_pattern.search(sh.name.decode())
            if sh.type == 'SHT_RELA' and not ignore_section:
                rela_entries = Rela.collect_rela_entries(sh)
                ElfFile.rela_dict[sh.name] = rela_entries
            
    def look_for_section(name):
        for sh in ElfFile.section_headers:
            if sh.name == name:
                return sh

    def look_for_rela(sh):
        rela_name = b'.rela' + sh.name
        exists = rela_name in ElfFile.rela_dict.keys()
        return ElfFile.rela_dict[rela_name] if exists else None

    def find_code_sections():
        from translator import Translator

        for i, s in enumerate(ElfFile.symbols):
            if s.type == 'STT_FUNC': 
                sh = ElfFile.section_headers[s.get('st_shndx')]
                sh.is_expanded = True

                offset = s.get('st_value')
                bcode = sh.section_data[offset : offset + s.get('st_size')]

                rela = ElfFile.look_for_rela(sh)

                if p_shift := Translator.count_functions(bcode):
                    ElfFile.symbol_code_dict[i] = (bcode, p_shift, offset, rela)
                else:
                    print("The function is not up to assignment specification.")
                    sys.exit(1)

    def overwrite_code_sections():
        from section_header import SectionHeader

        for i in range(len(ElfFile.section_headers)):
            sh = ElfFile.section_headers[i]

            if sh.is_expanded:
                symbols = [(num, s) for num, s in enumerate(ElfFile.symbols) if s.get('st_shndx') == i]
                symbols = sorted(symbols, key = lambda s : s[1].get('st_value'))

                section_data = b''
                value = 0
                for sym_args in symbols:
                    section_data, value = ElfFile.overwrite_section(sh, section_data, value, *sym_args)

                sh.set('sh_size', len(section_data))
                sh.section_data = section_data

            elif not SectionHeader.delete_pattern.search(sh.name.decode()):
                ElfFile.fix_nonfun_rela(sh, 0, 0, len(sh.section_data))

    def overwrite_section(sh, section_data, value, symbol_num, symbol):
        from translator import Translator

        if value < symbol.get('st_value'):
            section_data += sh.section_data[value : symbol.get('st_value')]
            value = symbol.get('sh_value')

        assert value == symbol.get('st_value')

        if symbol.type == 'STT_FUNC': 
            bcode, p_shift, offset_arm, rela = ElfFile.symbol_code_dict[symbol_num]
            assembled, _ = Translator.translate_code(
                bcode,
                p_shift,
                offset_arm,
                len(section_data),
                rela,
            )

            section_data += assembled
            value = symbol.get('st_value') + symbol.get('st_size')
            symbol.set('st_size', len(assembled))
            symbol.set('st_value', len(section_data) - symbol.get('st_size'))

        else:
            ElfFile.fix_nonfun_rela(sh, len(section_data), value, symbol.get('st_size'))

            if symbol.type in ['STT_NOTYPE', 'STT_OBJECT']:
                section_data += sh.section_data[value : value + symbol.get('st_size')]
                value = symbol.get('st_value') + symbol.get('st_size')
                symbol.set('st_value', len(section_data) - symbol.get('st_size'))
            else:
                section_data += sh.section_data[value : value + symbol.get('st_size')]
                value = symbol.get('st_value') + symbol.get('st_size')

        return section_data, value

    def fix_nonfun_rela(sh, new_value, old_value, size):
        from rela import Rela

        rela = ElfFile.look_for_rela(sh)

        if rela:
            rela_offsets = rela.keys()
            keys_in_section = [
                off for off in rela_offsets
                if off >= old_value and off < old_value + size
            ]

            for k in keys_in_section:
                rela_entry : Rela = rela[k]

                assert rela_entry.type == 'R_AARCH64_ABS64', \
                    f"Unknown relocation type: {rela_entry.type}"
                
                rela_entry.overwrite_rela(
                    type = 'R_AMD64_64',
                    offset_shift = new_value - old_value,
                )

    # Unused, but may be useful for debugging
    def check_rela():
        for r in ElfFile.rela_dict.values():
            for r_ent in r.values():
                import re
                if re.compile('R_AARCH64', 0).search(r_ent.type):
                    assert False 

    def remove_sections():
        from elf_header import ElfHeader
        from section_header import SectionHeader 

        new_section_headers = []
        deleted_sections = 0
        for i, sh in enumerate(ElfFile.section_headers):
            if not SectionHeader.delete_pattern.search(sh.name.decode()):
                new_section_headers += [sh]
                ElfFile.sh_idx_translation.append(i - deleted_sections)
            else:
                deleted_sections += 1
                ElfFile.sh_idx_translation.append(ElfFile.symtab_idx)

        ElfHeader.set('e_shnum', len(new_section_headers))

        for i, sh in enumerate(new_section_headers):
            link = sh.get('sh_link')
            sh.set('sh_link', ElfFile.sh_idx_translation[link])

            if sh.is_shstrs:
                ElfHeader.set('e_shstrndx', i)

        ElfFile.new_section_headers = new_section_headers

    def save_expanded_sections(file):
        from elf_header import ElfHeader

        current_offset = ElfHeader.get('e_shoff')

        for sh in ElfFile.new_section_headers:
            if sh.is_expanded:
                align = sh.get('sh_offset') % 16
                current_offset += align

                sh.set('sh_offset', current_offset)
                current_offset = sh.save_section(file, current_offset)

                current_offset += 16 - (current_offset % 16)

        ElfHeader.set('e_shoff', current_offset)
        for sh in ElfFile.new_section_headers:
            current_offset = sh.save(file, current_offset)

    def save_rela_and_sym(file):
        from rela import Rela
        from sym import Sym

        for name, rela_entries in ElfFile.rela_dict.items():
            sh = ElfFile.sh_dict[name.decode()]
            Rela.save(file, sh, rela_entries.values())

        Sym.save(file, ElfFile.section_headers[ElfFile.symtab_idx], ElfFile.symbols)

    def save_header(file):
        from elf_header import ElfHeader

        ElfHeader.set('e_machine', ElfHeader.amd_machine)
        ElfHeader.save(file)

