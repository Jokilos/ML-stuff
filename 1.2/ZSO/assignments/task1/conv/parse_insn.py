import capstone
import re
from tools import expand_rt_dict
from rela import Rela

class ParseInsn:
    cond_mapping = {
        'eq' : 'e',
        'ne' : 'ne',
        'hs' : 'ae',
        'lo' : 'b',
        'mi' : 's',
        'pl' : 'ns',
        'vs' : 'o',
        'vc' : 'no',
        'hi' : 'a',
        'ls' : 'be',
        'ge' : 'ge',
        'lt' : 'l',
        'gt' : 'g',
        'le' : 'le',
    }

    reg_p, register_translation = expand_rt_dict({
        'x0' : 'rdi',
        'x1' : 'rsi',
        'x2' : 'rdx',
        'x3' : 'rcx',
        'x4' : 'r8',
        'x5' : 'r9',
        'x9' : 'rax',
        'x10' : 'r10',
        'x29' : 'rbp',
        'x19' : 'rbx',
        'x20' : 'r12',
        'x21' : 'r13',
        'x22' : 'r14',
        'x23' : 'r15',
        'sp' : 'rsp',
    })

    str_p = r'([#a-z0-9]+)'
    hex_p = r'#([xa-f0-9]+)'

    @staticmethod
    def isreg64(reg):
        return reg[0] == 'x' or reg[0] == 's'
        
    @staticmethod
    def get_sizeq(reg):
        if ParseInsn.isreg64(reg):
            return 'qword ptr'
        else:
            return 'dword ptr'

    @staticmethod
    def is_reg(str):
        return re.compile(ParseInsn.reg_p).search(str)

    @staticmethod
    def parse(insn, rela = None):
        if insn.mnemonic[:2] == 'b.':
            return ParseInsn.bcond(insn, insn.mnemonic[2:], rela)
        else:
            handle_fun = getattr(ParseInsn, insn.mnemonic)
            return handle_fun(insn, rela)

    @staticmethod
    def ldr(insn : capstone.CsInsn, rela : dict[int, Rela] = None):
        ptn = f'{ParseInsn.reg_p}, ' 
        ptn += rf'\[{ParseInsn.reg_p}, {ParseInsn.hex_p}\]'
        ptn = re.compile(ptn)

        reg1, reg2, op2d = ptn.search(insn.op_str).groups()

        op1 = ParseInsn.register_translation[reg1]

        if not ParseInsn.isreg64(reg1) and reg2 == 'sp':
            op2b = 'esp'
        else:
            op2b = ParseInsn.register_translation[reg2]

        sq = ParseInsn.get_sizeq(reg1)

        return f'mov {op1}, {sq} [{op2b} + {op2d}]\n'

    @staticmethod
    def str(insn : capstone.CsInsn, rela : dict[int, Rela] = None):
        ptn = f'{ParseInsn.reg_p}, ' 
        ptn += rf'\[{ParseInsn.reg_p}, {ParseInsn.hex_p}\]'
        ptn = re.compile(ptn)

        reg1, reg2, op2d = ptn.search(insn.op_str).groups()

        op1 = ParseInsn.register_translation[reg1]

        if not ParseInsn.isreg64(reg1) and reg2 == 'sp':
            op2b = 'esp'
        else:
            op2b = ParseInsn.register_translation[reg2]

        sq = ParseInsn.get_sizeq(reg1)

        return f'mov {sq} [{op2b} + {op2d}], {op1}\n'

    @staticmethod
    def adrp(insn : capstone.CsInsn, rela : dict[int, Rela] = None):
        ptn = f'{ParseInsn.reg_p}, ' 
        ptn += rf'{ParseInsn.hex_p}'
        ptn = re.compile(ptn)

        reg1, _ = ptn.search(insn.op_str).groups()

        rela[insn.address].overwrite_rela(
            type = 'R_AMD64_32',
            offset_shift = 3,
        )

        op1 = ParseInsn.register_translation[reg1]

        # the displacement forces the assembler to use a 32-bit immediate; it is relocated
        ret = f'lea {op1}, [rip + 0x7fffffff]\n' 

        # set 12 lowest bits to 0
        ret += f'and {op1}, ~0xfff\n' 

        return ret

    @staticmethod
    def mov_or_cmp(insn : capstone.CsInsn, op_name, rela = None):
        ptn = f'{ParseInsn.reg_p}, ' 
        ptn = re.compile(ptn + rf'{ParseInsn.str_p}')

        op1, op2 = ptn.search(insn.op_str).groups()
        op1 = ParseInsn.register_translation[op1]

        if ParseInsn.is_reg(op2):
            op2 = ParseInsn.register_translation[op2]
        else:
            op2 = op2[1:]

        return f'{op_name} {op1}, {op2}\n'

    @staticmethod
    def mov(insn, rela = None):
        return ParseInsn.mov_or_cmp(insn, 'mov', rela)

    @staticmethod
    def cmp(insn, rela = None):
        return ParseInsn.mov_or_cmp(insn, 'cmp', rela)

    @staticmethod
    def add(insn : capstone.CsInsn, rela : dict[int, Rela] = None):
        ptn = f'{ParseInsn.reg_p}, ' 
        ptn += f'{ParseInsn.reg_p}, ' 
        ptn = re.compile(ptn + rf'{ParseInsn.str_p}')

        op1_old, op2, op3 = ptn.search(insn.op_str).groups()
        op1 = ParseInsn.register_translation[op1_old]
        op2 = ParseInsn.register_translation[op2]

        has_rela = insn.address in rela.keys() 
        if ParseInsn.is_reg(op3):
            op3 = ParseInsn.register_translation[op3]
        else:
            if has_rela:
                rela[insn.address].overwrite_rela(
                    type = 'R_AMD64_32',
                    offset_shift = -4,    
                )
            op3 = op3[1:]

        if op1 == op2:
            return f'add {op1}, {op3}\n'
        elif op1 == op3:
            return f'add {op1}, {op2}\n'
        elif has_rela:
            tmp = 'r11' if ParseInsn.isreg64(op1_old) else 'r11d'
            ret = f'mov {tmp}, 0x7fffffff\n' # the immediate is relocated
            ret += f'and {tmp}, 0xfff\n'
            return ret + f'add {op1}, {tmp}\n'
        else:
            ret = f'mov {op1}, {op2}\n'       
            return ret + f'add {op1}, {op3}\n'       

    @staticmethod
    def bl(insn : capstone.CsInsn, rela : dict[int, Rela] = None):
        rela[insn.address].overwrite_rela(
            type = 'R_AMD64_PC32',
            offset_shift = 1,
        )

        # the offset is relocated
        ret = 'call 0x7fffffff\n' 
        # put the return value in the register to which x0 maps'
        ret += 'mov rdi, rax\n'
        return ret
    
    @staticmethod
    def b(insn : capstone.CsInsn, rela : dict[int, Rela] = None):
        ptn = re.compile(f'{ParseInsn.hex_p}')

        imm = ptn.search(insn.op_str).groups()

        return f'jmp {imm}\n'
    
    @staticmethod
    def bcond(insn : capstone.CsInsn, cond, rela : dict[int, Rela] = None):
        ptn = re.compile(f'{ParseInsn.hex_p}')

        imm = ptn.search(insn.op_str).groups()
        cond = ParseInsn.cond_mapping[cond]

        return f'j{cond} {imm}\n'