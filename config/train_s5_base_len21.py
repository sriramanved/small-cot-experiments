out_dir = 'out-s5-base-len21-depth1'
eval_interval = 5000
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

wandb_log = False
dataset = 's5_cot'

s5_mode = 'base'
s5_m = 21

n_layer = 1
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

# prompt 148, target 7 => total 155, so x/y length 154
block_size = 154

batch_size = 64
gradient_accumulation_steps = 1
learning_rate = 1e-5
max_iters = 1000000
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# Match the native OPD trainer: warm up from 1e-6 to learning_rate, then stay flat.
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = learning_rate

dtype = 'float16'
compile = True
