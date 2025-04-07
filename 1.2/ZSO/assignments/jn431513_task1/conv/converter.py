#!/usr/bin/env python3

from elf_file import ElfFile
import shutil
import sys

# Optional error coloring
# from IPython.core import ultratb
# sys.excepthook = ultratb.FormattedTB(color_scheme = 'Linux', call_pdb = False)

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
ElfFile.check_rela()

ElfFile.remove_sections()
ElfFile.save_expanded_sections(output)
ElfFile.save_rela_and_sym(output)
ElfFile.save_header(output)
