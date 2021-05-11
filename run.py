  
from subprocess import call
import sys
data = ['./dataset/solar_AL.txt', './dataset/exchange_rate.txt','./dataset/electricity.txt','./dataset/traffic.txt']
#lr = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
lr = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3]
#lr = [1e-4, 5e-4, 1e-3]
#lr = [1e-4, 5e-4, 1e-3]
#lr = [1e-3,5e-3, 9e-3]
#lr = [4e-4, 3e-4, 6e-4, 7e-4, 8e-4]
#lr = [4e-3, 5e-3,6e-3]
lradj = [1,5,6,9]
#batch = [8, 16, 32, 64, 128, 256, 512, 1024]
batch = [8,32]
#batch = [256, 512, 1024]
#batch = [256, 512,1024]
#batch = [32, 64, 128, 256, 512, 1024, 1280]
#batch = [4,5,6,7]
#batch = [4, 8, 12]
kernel = [1,3,7,9]
horizon = [3, 6, 12, 24]
#hid = [0.03125, 0.0625,0.125, 0.25, 0.5, 0.75, 1.25, 1.5, 2, 3, 4]
#hid = [0.125, 0.25, 0.5, 1,2,3,4]
hid = [2,4,8]
drop = [0,0.1,0.2]
#hid = [0.5,2]
#hid = [1, 8, 16, 128,256]
input_length = [24, 48, 72, 96, 128]
output = [3, 6, 12, 24]
#output = [6,12]
#weight = [0.25,0.125,0.05, 0.01]
weight=[1.0,2]
norm = ['AugNormAdj','NormAdj','RWalk','AugRWalk','NoNorm','LowPass','NormLap','RWalkLap','FirstOrderGCN']
cat = [144, 120, 96, 72]
norm = [2]
oo = [0,1]
for i, v in enumerate(output):
    #for j,k in enumerate(drop):
    #cmd = """srun -p vi_x_cerebra_meta -w SH-IDC1-10-198-8-238 --gres=gpu:1 nohup python -u run_financial_power.py --data ./dataset/solar_AL.txt --hidden-size 2 --single_step 0 --lastWeight 0.5 --normalize 2 --window_size 160 --layers 4 --model_mode Enco --num_concat 0 --lradj 6 --lr 1e-4 --horizon {} --kernel 5 --dropout 0.25 --batch_size 1024 --model_name so_I160_type1_lr_bs1024_dp0.25_h2_norm2_Ts1l4_o{}_w0.5_scale > log/0510_so_I160_type1_lr_bs1024_dp0.25_norm2_h2_Ts1l4_o{}_w0.5.log 2>&1&""".format(v,v,v)
 #       cmd = """srun -p vi_x_cerebra_meta  --gres=gpu:1 nohup python -u run_financial_power.py --data ./dataset/exchange_rate.txt --single_step 0 --single_step_output_One 0 --epochs 150 --model_mode Enco --hidden-size 8 --dropout {} --groups 8 --window_size 168 --lradj 1 --lr 5e-3 --lastWeight 0.5 --horizon {} --kernel 5 --normalize 2 --batch_size 4 --model_name ex_I168_type1_lr5e-3_bs4_dp{}_h8_1stack3layer_norm2_out{}_w0.5_g8 > log/0506_ex_I168_type1_lr5e-3_bs4_dp{}_h8_1stack3layer_norm2_out{}_w0.5_g8.log 2>&1&""".format(k,v,k,v,k,v)
    #cmd = """srun -p vi_x_cerebra_meta --gres=gpu:1 nohup python -u run_financial_power.py  --data ./dataset/electricity.txt --single_step 0 --single_step_output_One 0 --hidden-size 8 --window_size 168 --num_concat 0 --lradj 1 --lr 9e-3 --horizon {} --kernel 5 --batch_size 32 --normalize 2 --groups 321 --epochs 150 --dropout 0 --lastWeight 0.5 --model_name ele_I168_So{}_type1_lr9e-3_bs32_dp0_h8_s2l3_gp321_w0.5 > log/0509_ele_I168_type1_lr9e-3_bs32_dp0_h8_s2l3_So{}_gp321_w0.5.log 2>&1&""".format(v,v,v)
#    cmd = """srun -p vi_x_cerebra_meta --gres=gpu:1 nohup python -u run_financial_power.py --data ./dataset/exchange_rate.txt --hidden-size 0.125 --window_size 168 --lradj 1 --lr 5e-3 --num_concat 0 --horizon 24 --kernel 5 --batch_size {} --model_name ex_I168_o24_type1_lr5e-3_bs{}_dp0.5_c168_h0.125_-+_2s3l > log/0427_ex_I168_o24_type1_lr5e-3_bs4_dp0.5_h0.125_-+_2s3l.log 2>&1&""".format(v,v,v)
    #cmd = """srun -p x_cerebra --gres=gpu:1 nohup python -u run_financial_power.py --data ./dataset/traffic.txt --hidden-size 1 --normalize 2 --single_step 1 --single_step_output_One 0 --layers 2 --window_size 168 --num_concat 0 --lradj 1 --lr 5e-4 --horizon {} --kernel 5 --batch_size 16 --dropout 0.25 --groups 1 --lastWeight 1.0 --model_name traf_I168_out{}_type1_lr5e-4_bs16_dp0.25_h1_s2l2_o1 > log/0510_traf_I168_So{}_type1_lr5e-4_bs16_dp0.25_h1_s2l2_o1.log 2>&1&""".format(v,v,v)
#    cmd = """srun -p x_cerebra --gres=gpu:1 nohup python -u main.py --dataset PEMS07 --input_dim 883 --lradj 6 --batch_size 8 --hidden-size {} --kernel 3 --model_name S7_T6_last_k3_dp0_hid{} > log/0417_S7_T6_last_k3_dp0_hid{}.log 2>&1&""".format(v,v,v2)
    cmd = """srun -p x_cerebra --gres=gpu:1 nohup python -u run_ETTh.py --hidden-size 4 --batchSize 32 --epochs 100 --layers 2 --seq_len 24 --label_len 24 --pred_len 24  --num_concat 0 --lradj 1 --lr 1e-4  --kernel 5 --dropout 0.25 --groups 1 --model_name ett2h_I24_out24_type1_lr1e-4_bs32_dp0.25_h4_s2l2 > log/0510_ett2h_I24_out24_type1_lr1e-4_bs32_dp0.25_h4_s2l2.log 2>&1&""".format(v,v,v)
    print(cmd)
    call(cmd, shell=True)
print('Finish!')

