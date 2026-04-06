out_dir = 'out-s5-cot-len21-depth1'
eval_interval = 5000
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

wandb_log = False
dataset = 's5_cot'

# task-specific
s5_mode = 'cot'
s5_m = 21

# model
n_layer = 1
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

# For p=5 and m=21:
# prompt length = 7*m + 1 = 148
# CoT length    = 7*m     = 147
# total seq len = 295, so x/y length = 294
block_size = 294

# optimizer / training
batch_size = 64
gradient_accumulation_steps = 1
learning_rate = 1e-5
max_iters = 1000000
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = False
warmup_iters = 2000

# precision
dtype = 'float16'

# system
compile = True