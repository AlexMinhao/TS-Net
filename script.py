
from subprocess import call
import sys


lr = [0.001]

epoch = [100]
hid  = [1]
bt = [8]
# head = [8,  16 , 32]
# pred_len = [128, 168]
stacks = [1]
type = [6,9]

for t in type:
    for h in hid:
        for l in lr:
            for b in bt:
                    cmd = """python -u main.py --lr {} --batch_size {} --hidden-size {} --lradj {} > log/FirstConvResReOrderTrue_PeMS08_12to12_bt{}_epoch50_lr{}type{}_hid{}.log 2>&1""".format(l,b,h,t,b,l,t,h)
                    print(cmd)
                    call(cmd, shell=True)
print('Finish!')


# lr = [0.0003]
# bt = [32]
# Hid = [1]
# H = [24]
# input = [168, 96, 32, 24]
# type = [1, 6]
# for l in lr:
#     for t in type:
#         for b in bt:
#             for h in H:
#                 for i in input:
#                     cmd = """python -u run_financial_power.py --lr {} --batch_size {} --window_size {} --horizon {} --lradj {} > log/EncoDeco_Electricity_{}to{}_hid1_lrType{}_lr{}_bt{}_WD1e-5_SLOSS_layer3.log 2>&1""".format(l,b,i,h,t, i,h,t,l,b)
#                     print(cmd)
#                     call(cmd, shell=True)