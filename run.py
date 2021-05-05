  
from subprocess import call
import sys
data = ['./dataset/solar_AL.txt', './dataset/exchange_rate.txt','./dataset/electricity.txt','./dataset/traffic.txt']
#lr = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
#lr = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3]
#lr = [1e-4, 5e-4, 1e-3]
#lr = [1e-4, 5e-4, 1e-3]
lr = [1e-3,5e-3, 9e-3]
#lr = [4e-4, 3e-4, 6e-4, 7e-4, 8e-4]
#lr = [4e-3, 5e-3,6e-3]
lradj = [1,5,6,9]
#batch = [8, 16, 32, 64, 128, 256, 512, 1024]
batch = [8,16,32]
#batch = [256, 512, 1024]
#batch = [256, 512,1024]
#batch = [32, 64, 128, 256, 512, 1024, 1280]
#batch = [4,5,6,7]
#batch = [4, 8, 12]
kernel = [1,3,7,9]
horizon = [3, 6, 12, 24]
#hid = [0.03125, 0.0625,0.125, 0.25, 0.5, 0.75, 1.25, 1.5, 2, 3, 4]
#hid = [0.125, 0.25, 0.5, 1,2,3,4]
hid = [1,2,4,8]
#hid = [0.5,2]
#hid = [1, 8, 16, 128,256]
input_length = [24, 48, 72, 96, 128]
#output = [3, 6, 12, 24]
output = [3,24]
weight = [0.5,1.0,2,5,10]
norm = ['AugNormAdj','NormAdj','RWalk','AugRWalk','NoNorm','LowPass','NormLap','RWalkLap','FirstOrderGCN']
cat = [144, 120, 96, 72]
norm = [2]
for i, v in enumerate(output):
    for j,k in enumerate(weight):
        cmd = """srun -p vi_x_cerebra_meta  --gres=gpu:1 nohup python -u run_financial_power.py --data ./dataset/solar_AL.txt --hidden-size 2 --model_mode Enco --single_step 0 --lastWeight {} --normalize 2 --window_size 168 --num_concat 0 --lradj 6 --lr 1e-4 --horizon {} --kernel 5 --batch_size 1024 --model_name so_I168_o24_type1_lr_bs1024_dp0.5_h2_norm2_s1l3_o{}_w{} > log/0504_so_I168_o24_type1_lr_bs1024_dp0.5_norm2_h2_s1l3_o{}_w{}.log 2>&1&""".format(k,v,v,k,v,k)
#        cmd = """srun -p vi_x_cerebra_meta -w SH-IDC1-10-198-8-245 --gres=gpu:1 nohup python -u run_financial_power.py --data ./dataset/exchange_rate.txt --single_step 1 --single_step_output_One 0 --epochs 150 --model_mode Enco --hidden-size {} --dropout 0 --groups 8 --window_size 168 --lradj 1 --lr 5e-3 --horizon {} --lastWeight 1.0 --kernel 5 --normalize 2 --batch_size 4 --model_name ex_I168_type1_lr5e-3_bs4_dp0_h{}_1stack3layer_norm2_o{}_w1_g8 > log/0505_ex_I168_type1_lr5e-3_bs4_dp0_h{}_1stack3layer_norm2_o{}_w1_g8.log 2>&1&""".format(k,v,k,v,k,v)
    #cmd = """srun -p x_cerebra -w SH-IDC1-10-198-8-110 --gres=gpu:1 nohup python -u run_financial_power.py  --data ./dataset/electricity.txt --single_step 1 --single_step_output_One 1 --hidden-size 4 --window_size 168  --num_concat 0 --lradj 1 --lr 0.009 --horizon {} --kernel 5 --batch_size 32 --normalize 2 --model_name ele_I168_So{}_c168_type1_lr9e-3_bs32_dp0_h4_s2l3_gp321 > log/0504_ele_I168_c168_type1_lr9e-3_bs32_dp0_h4_s2l3_So{}_gp321.log 2>&1&""".format(v,v,v)
#    cmd = """srun -p vi_x_cerebra_meta --gres=gpu:1 nohup python -u run_financial_power.py --data ./dataset/exchange_rate.txt --hidden-size 0.125 --window_size 168 --lradj 1 --lr 5e-3 --num_concat 0 --horizon 24 --kernel 5 --batch_size {} --model_name ex_I168_o24_type1_lr5e-3_bs{}_dp0.5_c168_h0.125_-+_2s3l > log/0427_ex_I168_o24_type1_lr5e-3_bs4_dp0.5_h0.125_-+_2s3l.log 2>&1&""".format(v,v,v)
    #cmd = """srun -p x_cerebra -w SH-IDC1-10-198-8-109 --gres=gpu:1 nohup python -u run_financial_power.py --data ./dataset/traffic.txt --hidden-size 2 --normalize 0 --single_step 1 --single_step_output_One 1 --window_size 168 --num_concat 0 --lradj 1 --lr 5e-4 --horizon {} --kernel 5 --batch_size 16  --model_name traf_I168_singleo{}_type1_lr5e-4_bs16_dp0.5_h2_s2l3_n0 > log/0503_traf_I168_So{}_type1_lr5e-4_bs16_dp0.5_h2_s2l3_n0.log 2>&1&""".format(v,v,v)
#    cmd = """srun -p x_cerebra --gres=gpu:1 nohup python -u main.py --dataset PEMS07 --input_dim 883 --lradj 6 --batch_size 8 --hidden-size {} --kernel 3 --model_name S7_T6_last_k3_dp0_hid{} > log/0417_S7_T6_last_k3_dp0_hid{}.log 2>&1&""".format(v,v,v2)
        print(cmd)
        call(cmd, shell=True)
print('Finish!')

