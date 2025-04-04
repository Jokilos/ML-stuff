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

    str_p = r'([-#a-z0-9]+)'
    hex_p = r'#([-xa-f0-9]+)'

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
        return re.compile(ParseInsn.reg_p).fullmatch(str)

    @staticmethod
    def parse(insn, rela = None):
        if insn.mnemonic[:2] == 'b.':
            return ParseInsn.bcond(insn, insn.mnemonic[2:], rela)
        else:
            handle_fun = getattr(ParseInsn, insn.mnemonic)
            return handle_fun(insn, rela)

    @staticmethod
    def str_or_ldr(insn : capstone.CsInsn, is_ldr, rela : dict[int, Rela] = None):
        ptn = rf'{ParseInsn.reg_p}, \[{ParseInsn.reg_p}' 

        # check if there is displacement
        if re.compile('#').search(insn.op_str):
            ptn += rf', {ParseInsn.hex_p}\]'
            ptn = re.compile(ptn)
            reg1, reg2, op2d = ptn.search(insn.op_str).groups()
            op2d = ' + ' + op2d
        else:
            ptn += r'\]'
            ptn = re.compile(ptn)
            reg1, reg2 = ptn.search(insn.op_str).groups()
            op2d = ''

        op1 = ParseInsn.register_translation[reg1]
        op2b = ParseInsn.register_translation[reg2]

        sq = ParseInsn.get_sizeq(reg1)

        if is_ldr:
            return f'mov {op1}, {sq} [{op2b}{op2d}]\n'
        else:
            return f'mov {sq} [{op2b}{op2d}], {op1}\n'

    @staticmethod
    def ldr(insn : capstone.CsInsn, rela : dict[int, Rela] = None):
        return ParseInsn.str_or_ldr(insn, is_ldr = True, rela = rela)

    @staticmethod
    def str(insn : capstone.CsInsn, rela : dict[int, Rela] = None):
        return ParseInsn.str_or_ldr(insn, is_ldr = False, rela = rela)

    @staticmethod
    def adrp(insn : capstone.CsInsn, rela : dict[int, Rela] = None):
        ptn = f'{ParseInsn.reg_p}, ' 
        ptn += rf'{ParseInsn.hex_p}'
        ptn = re.compile(ptn)

        reg1, _ = ptn.search(insn.op_str).groups()

        rela[insn.address].overwrite_rela(
            type = 'R_AMD64_PC32',
            offset_shift = 3,
            addend_shift = -4,
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
    def add(insn : capstone.CsInsn, rela : dict[int, Rela] = None, is_sub = False):
        ptn = f'{ParseInsn.reg_p}, ' 
        ptn += f'{ParseInsn.reg_p}, ' 
        ptn = re.compile(ptn + rf'{ParseInsn.str_p}')

        op1_old, op2_old, op3 = ptn.search(insn.op_str).groups()
        op1 = ParseInsn.register_translation[op1_old]
        op2 = ParseInsn.register_translation[op2_old]

        has_rela = rela and insn.address in rela.keys() 
        has_imm = not ParseInsn.is_reg(op3)

        if not has_imm:
            op3 = ParseInsn.register_translation[op3]
        else:
            if has_rela:
                rela[insn.address].overwrite_rela(
                    type = 'R_AMD64_32',
                    offset_shift = 3,    
                )
            op3 = op3[1:]
        
        add_op = 'add' if not is_sub else 'sub'
        def add_opy_to_opx(opy, opx):
            if has_imm and has_rela:
                tmp = 'r10' if ParseInsn.isreg64(op1_old) else 'r11d'
                ret = f'mov {tmp}, 0x7fffffff\n' # the immediate is relocated
                ret += f'and {tmp}, 0xfff\n'
                ret += f'{add_op} {opx}, {tmp}\n'
            else:
                ret = f'{add_op} {opx}, {opy}\n'

            return ret

        if op1 == op2:
            return add_opy_to_opx(op3, op1) 
        elif op1 == op3:
            return add_opy_to_opx(op2, op1) 
        else:
            ret = f'mov {op1}, {op2}\n' 
            return ret + add_opy_to_opx(op3, op1)

    @staticmethod
    def sub(insn, rela):
        return ParseInsn.add(insn, rela, True)

    @staticmethod
    def bl(insn : capstone.CsInsn, rela : dict[int, Rela] = None):
        rela[insn.address].overwrite_rela(
            type = 'R_AMD64_PC32',
            offset_shift = 1,
            addend_shift = -4,
        )

        # the offset is relocated
        ret = 'call 0x7fffffff\n' 
        # put the return value in the register to which x0 maps'
        ret += 'mov rdi, rax\n'
        return ret
    
    @staticmethod
    def b(insn : capstone.CsInsn, rela : dict[int, Rela] = None):
        ptn = re.compile(f'{ParseInsn.hex_p}')

        imm = ptn.search(insn.op_str).groups()[0]

        return f'jmp {imm}\n'
    
    @staticmethod
    def bcond(insn : capstone.CsInsn, cond, rela : dict[int, Rela] = None):
        ptn = re.compile(f'{ParseInsn.hex_p}')

        imm = ptn.search(insn.op_str).groups()[0]
        cond = ParseInsn.cond_mapping[cond]

        return f'j{cond} {imm}\n'