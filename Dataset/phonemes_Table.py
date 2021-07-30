phonemes_table = []

ONS = ['k0', 'kk', 'nn', 't0', 'tt', 'rr', 'mm', 'p0', 'pp', 's0', 'ss', 'oh', 'c0', 'cc', 'ch', 'kh', 'th', 'ph', 'h0']
NUC = ['aa', 'qq', 'ya', 'yq', 'vv', 'ee', 'yv', 'ye', 'oo', 'wa', 'wq', 'wo', 'yo', 'uu', 'wv', 'we', 'wi', 'yu', 'xx', 'xi', 'ii']
COD = ['kf', 'kk', 'ks', 'nf', 'nc', 'nh', 'tf', 'll', 'lk', 'lm', 'lb', 'ls', 'lt', 'lp', 'lh', 'mf', 'pf', 'ps', 's0', 'ss', 'oh', 'c0', 'ch', 'kh', 'th', 'ph', 'h0']
EXC = ['h0', 'ng', 'lm', 'lt', 'nf', 'lk', 'nh', 'pf', 'nc', 'll', 'mf', 'ls', 'ps', 'lp', 'lh', 'tf', 'kf', 'ks', 'lb']

phonemes_table_temp = [' '] + ONS + NUC + COD + EXC
for phone in phonemes_table_temp:
    if phone not in phonemes_table:
        phonemes_table.append(phone)

import numpy as np
phonemes_table = np.asarray(phonemes_table) # 59 lens

