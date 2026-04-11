out_dir = 'out-modadd-cot-p7-m21-depth1'
eval_interval = 5000
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

wandb_log = False
dataset = 'modadd_cot'

modadd_p = 7
modadd_m = 21

n_layer = 1
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

block_size = 42

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

dtype = 'float16'
compile = True

modadd_eval_metrics = True
s5_eval_n = 256
s5_eval_seed = 123
save_every = 50000
