k = 'lgbm_lr=0.03_lv=15_ne=100'
print(repr(k), k.split('='))
import re
print(re.findall(r'[\d.]+', k.split('=')[1]))
k2 = 'lgbm_lr=0.03_lv=15_ne=100'.split('=')
print(k2)
