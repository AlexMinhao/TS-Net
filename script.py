
from subprocess import call
import sys

#
# lr = [0.001]
#
# epoch = [100]
# hid  = [1]
# bt = [8]
# # head = [8,  16 , 32]
# # pred_len = [128, 168]
# stacks = [1]
# type = [6,9]
#
# for t in type:
#     for h in hid:
#         for l in lr:
#             for b in bt:
#                     cmd = """python -u main.py --lr {} --batch_size {} --hidden-size {} --lradj {} > log/FirstConvResReOrderTrue_PeMS08_12to12_bt{}_epoch50_lr{}type{}_hid{}.log 2>&1""".format(l,b,h,t,b,l,t,h)
#                     print(cmd)
#                     call(cmd, shell=True)
# print('Finish!')


# lr = [0.0005]
# bt = [4]
# Hid = [0.5, 1]
# H = [24]
# input = [168, 96]
# type = [6]
# for l in lr:
#     for t in type:
#         for b in bt:
#             for hid in Hid:
#                 for h in H:
#                     for i in input:
#                         cmd = """python -u run_financial_power.py --lr {} --batch_size {} --hidden-size {} --window_size {} --horizon {} --lradj {} > log/Exchange/EncoDecoReOrder_Exchange_NoNormReOrderCodeClear_{}to{}_hid{}_lrType{}_lr{}_bt{}_WD1e-5_SLOSS_layer3.log 2>&1""".format(l,b,hid,i, h,t, i,h,hid,t, l,b)
#                         print(cmd)
#                         call(cmd, shell=True)

# lr = [0.0005]
# bt = [4]
# Hid = [0.5, 1]
# H = [24]
# input = [168, 96]
# type = [6]
# for l in lr:
#     for t in type:
#         for b in bt:
#             for hid in Hid:
#                 for h in H:
#                     for i in input:
#                         cmd = """python -u run_financial_power.py --lr {} --batch_size {} --hidden-size {} --window_size {} --horizon {} --lradj {} > log/Exchange/EncoDecoReOrder_Exchange_NoNormReOrderCodeClear_{}to{}_hid{}_lrType{}_lr{}_bt{}_WD1e-5_SLOSS_layer3.log 2>&1""".format(l,b,hid,i, h,t, i,h,hid,t, l,b)
#                         print(cmd)
#                         call(cmd, shell=True)



# ETTH
cmd = ["python -u run_ETTh.py --model IDCN --data ETTh1 --features M --seq_len 48 --label_len 48 --pred_len 24 --des 'Exp' --itr 5 --train_epochs 30",
"python -u run_ETTh.py --model IDCN --data ETTh1 --features M --seq_len 96 --label_len 48 --pred_len 48 --des 'Exp' --itr 5 --train_epochs 30",
"python -u run_ETTh.py --model IDCN --data ETTh1 --features M --seq_len 168 --label_len 168 --pred_len 168 --des 'Exp' --itr 5 --train_epochs 30",
"python -u run_ETTh.py --model IDCN --data ETTh1 --features M --seq_len 168 --label_len 168 --pred_len 336 --des 'Exp' --itr 5 --train_epochs 30",
"python -u run_ETTh.py --model IDCN --data ETTh1 --features M --seq_len 336 --label_len 336 --pred_len 720 --des 'Exp' --itr 5 --train_epochs 30",

"python -u run_ETTh.py --model IDCN --data ETTh2 --features M --seq_len 48 --label_len 48 --pred_len 24 --des 'Exp' --itr 5 --train_epochs 30",
"python -u run_ETTh.py --model IDCN --data ETTh2 --features M --seq_len 96 --label_len 96 --pred_len 48 --des 'Exp' --itr 5 --train_epochs 30",
"python -u run_ETTh.py --model IDCN --data ETTh2 --features M --seq_len 336 --label_len 336 --pred_len 168 --des 'Exp' --itr 5 --train_epochs 30",
"python -u run_ETTh.py --model IDCN --data ETTh2 --features M --seq_len 336 --label_len 168 --pred_len 336 --des 'Exp' --itr 5 --train_epochs 30",
"python -u run_ETTh.py --model IDCN --data ETTh2 --features M --seq_len 720 --label_len 336 --pred_len 720 --des 'Exp' --itr 5 --train_epochs 30",

"python -u run_ETTh.py --model IDCN --data ETTm1 --features M --seq_len 672 --label_len 96 --pred_len 24 --des 'Exp' --itr 5

"python -u run_ETTh.py --model IDCN --data ETTm1 --features M --seq_len 96 --label_len 48 --pred_len 48 --des 'Exp' --itr 5

"python -u run_ETTh.py --model IDCN --data ETTm1 --features M --seq_len 384 --label_len 384 --pred_len 96 --des 'Exp' --itr 5

"python -u run_ETTh.py --model IDCN --data ETTm1 --features M --seq_len 672 --label_len 288 --pred_len 288 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

"python -u run_ETTh.py --model IDCN --data ETTm1 --features M --seq_len 672 --label_len 384 --pred_len 672 --e_layers 2 --d_layers 1 --attn prob --des 'Exp' --itr 5

       ]

log = [" > ETTH_util/log/ETTh1_seq_len_48_label_len_48_pred_len_24_layer3_2stacks.log 2>&1",
       " > ETTH_util/log/ETTh1_seq_len_96_label_len_48_pred_len_48_layer3_2stacks.log 2>&1",
" > ETTH_util/log/ETTh1_seq_len_168_label_len_168_pred_len_168_layer3_2stacks.log 2>&1",
" > ETTH_util/log/ETTh1_seq_len_168_label_len_168_pred_len_336_layer3_2stacks.log 2>&1",
" > ETTH_util/log/ETTh1_seq_len_336_label_len_336_pred_len_720_layer3_2stacks.log 2>&1",

" > ETTH_util/log/ETTh2_seq_len_48_label_len_48_pred_len_24_layer3_2stacks.log 2>&1",
" > ETTH_util/log/ETTh2_seq_len_96_label_len_96_pred_len_48_layer3_2stacks.log 2>&1",
" > ETTH_util/log/ETTh2_seq_len_336_label_len_336_pred_len_168_layer3_2stacks.log 2>&1",
" > ETTH_util/log/ETTh2_seq_len_336_label_len_168_pred_len_336_layer3_2stacks.log 2>&1",
" > ETTH_util/log/ETTh2_seq_len_720_label_len_336_pred_len_720_layer3_2stacks.log 2>&1",



       ]

cmd = """python -u run_financial_power.py --lr {} --batch_size {} --hidden-size {} --window_size {} --horizon {} --lradj {} > log/Exchange/EncoDecoReOrder_Exchange_NoNormReOrderCodeClear_{}to{}_hid{}_lrType{}_lr{}_bt{}_WD1e-5_SLOSS_layer3.log 2>&1""".format(l,b,hid,i, h,t, i,h,hid,t, l,b)
print(cmd)
call(cmd, shell=True)