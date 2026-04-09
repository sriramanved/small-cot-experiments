out_dir = 'out-s5-clean-offline-bc'
eval_interval = 10000
log_interval = 50
eval_iters = 50
always_save_checkpoint = True

wandb_log = False
dataset = 's5_clean_offline_n6000000'   # override from CLI if needed
init_from = 'scratch'

s5_mode = 'cot'
s5_m = 21

n_layer = 1
n_head = 8
n_embd = 512
dropout = 0.0
bias = False

block_size = 294

batch_size = 64
gradient_accumulation_steps = 1
learning_rate = 1e-5
max_iters = 1000000   # harmless because offline_single_epoch=True will stop earlier
weight_decay = 0.0
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = False
warmup_iters = 2000

dtype = 'float16'
compile = True

offline_single_epoch = True
offline_eval_full = False
offline_train_subset_size = 0
final_eval_on_exit = True
s5_eval_metrics = True
s5_eval_n = 5000
s5_eval_batch_size = 256
s5_eval_seed = 123
save_every = 0
