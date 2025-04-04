#!/usr/bin/env python3

from elf_file import ElfFile
import shutil

# Error coloring
import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(color_scheme = 'Linux', call_pdb = False)

# code = """mov esi, edi
# .label:
# jmp <.label + 0x10>
# nop"""
# from translator import Translator
# print(Translator.assemble_code(code, True))
# assert False

input = sys.argv[1]
output = sys.argv[2]

shutil.copy(input, output)

ElfFile.setup(input)
ElfFile.read_elf_header()
ElfFile.read_section_headers()
ElfFile.read_symbols()
ElfFile.read_rela()

ElfFile.find_code_sections()
ElfFile.overwrite_code_sections()

ElfFile.remove_sections()
ElfFile.save_expanded_sections(output)
ElfFile.save_rela_and_sym(output)
ElfFile.save_header(output)
