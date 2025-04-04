import re

class Comparator:
    # for capstone output
    hex_pattern = '(0x[0-9a-f]+)' 

    prolog = """
        stp x29, x30, [sp, #-prologue_shift]!
        mov x29, sp
    """

    epilog = """
        ldp x29, x30, [sp], #prologue_shift
        ret
    """

    escape_chars = '[]'

    def unify(text, pattern = False):
        if pattern:
            for char in Comparator.escape_chars:
                text = text.replace(f'{char}', fr'\{char}')

        text = text.replace('prologue_shift', Comparator.hex_pattern)

        tokens = text.strip().split()

        unified = ''
        for t in tokens:
            unified += t + ' '

        return unified 
    
    def compare_part(code, is_prolog = True, verbose = False):
        unified_code = Comparator.unify(code)

        pattern = getattr(Comparator, 'prolog' if is_prolog else 'epilog')
        unified_pattern = Comparator.unify(pattern, True)

        if verbose: 
            print(unified_code + '\n')
            print(unified_pattern + '\n')

        compiled_pattern = re.compile(unified_pattern, re.IGNORECASE)
        return compiled_pattern.search(unified_code)

    # def count_lines(chars, code, from_end = False):
    #     lengths = []
    #     for line in code.splitlines():
    #         length = 0
    #         for word in line.split():
    #             length += len(word)
    #         lengths += [length]

    #     lengths = reversed(lengths) if from_end else lengths

    #     lines = 0
    #     for l in lengths:
    #         if chars == 0:
    #             return lines

    #         chars -= l
    #         lines += 1

    #     assert False, 'Function detection failed'

    def check_function(code):
        match_p = Comparator.compare_part(code, True)
        match_e = Comparator.compare_part(code, False)
        length = len(Comparator.unify(code))

        if match_p and match_e:
            span = (match_p.span()[0], match_e.span()[1])

            assert match_p.group(1) == match_p.group(1), "Prologue shift doesn't match"

            if span == (0, length):
                return match_p.group(1) 

            assert False, (span, length, 'Assignment conditions not fullfilled.')
            
        return None