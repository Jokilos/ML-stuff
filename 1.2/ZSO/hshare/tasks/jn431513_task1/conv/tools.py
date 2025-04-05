import re

class Const:
    HEADER_SIZE = 64 

def make_idx_dict(names):
    d = {} 
    for i, n in enumerate(names):
        d[n] = i

    return d

def two_way_dict(dictionary):
    for k in list(dictionary.keys()):
        dictionary[dictionary[k]] = k

    return dictionary

def dict_safe_get(dictionary, key):
    try:
        return dictionary[key]
    except KeyError:
        if isinstance(key, str):
            return 0
        else:
            return 'UNKNOWN'
        
def expand_rt_dict(dictionary):
    for k in list(dictionary.keys()):
        if k[0] == 'x':
            rx_pattern = re.compile(r'r(\d)+', 0)
            val = dictionary[k]

            if rx_pattern.search(val):
                val += 'd'
            else:
                val = 'e' + val[1:]

            dictionary['w' + k[1:]] = val

    dictionary['xzr'] = 0
    dictionary['wzr'] = 0

    reg_p = '('
    for k in dictionary.keys():
        reg_p += k + '|'
    reg_p = reg_p[:-1] + ')'

    # for k,v in dictionary.items():
        # print(k,v)

    return reg_p, dictionary

def overwrite_file(file, offset, data):
    with open(file, 'r+b') as f:
        content = f.read()

        content = content[0 : offset] + \
            data + content[offset + len(data) :]

        f.seek(0)
        f.write(content)
