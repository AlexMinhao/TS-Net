  
from subprocess import call
import sys
data = ['./dataset/solar_AL.txt', './dataset/exchange_rate.txt','./dataset/electricity.txt','./dataset/traffic.txt']
#lr = [5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
#lr = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3]
lr = [1e-4, 3e-4, 5e-4, 7e-4, 1e-3, 5e-3]
lradj = [1,5,6,9]
#batch = [8, 16, 32, 64, 128, 256, 512, 1024, 1280, 1520]
batch = [4,8,16,32]
#batch = [4,5,6,7]
#batch = [8, 16, 32, 64, 128, 256]
kernel = [1,3,7,9]
horizon = [3, 6, 12, 24]
#hid = [0.03125, 0.0625,0.125, 0.25, 0.5, 0.75, 1.25, 1.5, 2, 3, 4]
hid = [0.125, 0.25, 0.5, 1,2,3,4]
#hid = [1.5,2,3,4]
#hid = [1, 8, 16, 128,256]
input_length = [24, 48, 72, 96, 128]
output = [3, 6, 12]
norm = ['AugNormAdj','NormAdj','RWalk','AugRWalk','NoNorm','LowPass','NormLap','RWalkLap','FirstOrderGCN']
cat = [120, 96, 72]
for i, v in enumerate(lr):
#    for j,k in enumerate(hid):
#    cmd = """srun -p vi_x_cerebra_meta --gres=gpu:1 nohup python -u run_financial_power.py --data ./dataset/solar_AL.txt --hidden-size 2 --window_size 168 --num_concat {} --lradj 6 --lr 1e-4 --horizon 24 --kernel 5 --batch_size 1024 --model_name so_I168_o24_type1_lr_bs1024_dp0.25_cat{}_h2+- > log/0425_so_I168_o24_type1_lr_bs1024_dp0.25_h2+-_cat{}.log 2>&1&""".format(v,v,v)
#        cmd = """srun -p x_cerebra --gres=gpu:1 nohup python -u run_financial_power.py --data ./dataset/exchange_rate.txt --hidden-size 0.5 --window_size 168 --lradj 1 --lr {} --horizon 24 --kernel 5 --batch_size {} --model_name ex_I168_o24_type1_lr{}_bs{}_dp0.25_h0.5 > log/0425_ex_I168_o24_type1_lr{}_bs{}_dp0.25_h0.5.log 2>&1&""".format(v,k,v,k,v,k)
#    cmd = """srun -p vi_x_cerebra_meta --gres=gpu:1 nohup python -u run_financial_power.py --data ./dataset/electricity.txt --hidden-size 2 --window_size 168 --num_concat 0 --lradj 1 --lr {} --horizon 24 --kernel 5 --batch_size 32 --model_name ele_I168_o24_c168_type1_lr{}_bs32_dp0.25_h2 > log/0425_ex_I168_o24_type1_c168_lr{}_bs32_dp0.25_h2.log 2>&1&""".format(v,v,v)
#    cmd = """srun -p vi_x_cerebra_meta --gres=gpu:1 nohup python -u run_financial_power.py --data ./dataset/exchange_rate.txt --hidden-size 1 --window_size 168 --lradj 1 --lr 4e-3 --horizon {} --kernel 5 --batch_size 4 --model_name ex_I168_o24_type1_lr4e-3_bs4_dp0.5_c168_h1_-+_o{} > log/0425_ex_I168_o24_type1_lr4e-3_bs4_dp0.5_h1_-+_o{}.log 2>&1&""".format(v,v,v)
    cmd = """srun -p x_cerebra --gres=gpu:1 nohup python -u run_financial_power.py --data ./dataset/traffic.txt --hidden-size 2  --window_size 168 --num_concat 0 --lradj 1 --lr {} --horizon 24 --kernel 5 --batch_size 16 --model_name traf_I168_o24_c168_type1_lr{}_bs16_dp0.25_h2+- > log/0425_traf_I168_o24_type1_c168_lr{}_bs16_dp0.25_h2+-.log 2>&1&""".format(v,v,v)
#    cmd = """srun -p x_cerebra --gres=gpu:1 nohup python -u main.py --dataset PEMS07 --input_dim 883 --lradj 6 --batch_size 8 --hidden-size {} --kernel 3 --model_name S7_T6_last_k3_dp0_hid{} > log/0417_S7_T6_last_k3_dp0_hid{}.log 2>&1&""".format(v,v,v2)
    print(cmd)
    call(cmd, shell=True)
print('Finish!')

