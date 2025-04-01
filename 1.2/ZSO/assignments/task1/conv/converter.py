from elf_file import ElfFile
import shutil

input = '../test-aarch64.o'  
output = '../out.o'
good_output = '../test-aarch64-x64.o'  

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
