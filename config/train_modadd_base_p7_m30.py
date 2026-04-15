out_dir = 'out-modadd-base-p7-m30-depth1'
eval_interval = 5000
log_interval = 50
eval_iters = 50
always_save_checkpoint = True

wandb_log = True
wandb_project = 'small-cot-experiments'
dataset = 'modadd_base'

modadd_p = 7
# Paper length 31 means 30 digits plus '='.
modadd_m = 30

n_layer = 1
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

block_size = 31

batch_size = 64
gradient_accumulation_steps = 1
learning_rate = 1e-5
max_iters = 200000
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = learning_rate

dtype = 'float16'
compile = False

final_eval_on_exit = True
modadd_eval_metrics = True
s5_eval_n = 256
s5_eval_batch_size = 512
s5_eval_seed = 123
save_every = 0
