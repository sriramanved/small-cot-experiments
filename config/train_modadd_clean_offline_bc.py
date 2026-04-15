out_dir = 'out-modadd-clean-offline-bc'
eval_interval = 5000
log_interval = 50
eval_iters = 200
always_save_checkpoint = True

wandb_log = False
dataset = 'modadd_clean_offline_p7_m21_n6000000'
init_from = 'scratch'

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
decay_lr = True
warmup_iters = 2000

dtype = 'float16'
compile = True

offline_single_epoch = True
offline_eval_full = False
offline_train_subset_size = 0
offline_train_shuffle = False
final_eval_on_exit = True
modadd_eval_metrics = True
modadd_eval_clean_train_loss = True
s5_eval_n = 5000
s5_eval_batch_size = 256
s5_eval_seed = 123
save_every = 0
