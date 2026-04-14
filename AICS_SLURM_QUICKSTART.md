# AICS Slurm Quickstart

This repo now includes a minimal AICS workflow for one S5 OPD eta job.

## 0. Start in scratch and enter the repo

If you log in and land somewhere like:

```bash
ved@aics:/scratch/blocklab/ved$
```

that is a good place to keep large working files. First make sure the repo exists there and
enter it:

```bash
pwd
find /scratch/blocklab/ved -maxdepth 4 -type d \( -name nanoGPT -o -name small-cot-experiments \) 2>/dev/null
cd /scratch/blocklab/ved/<your-repo-dir>
```

If the repo is not already under your scratch space, copy or clone it there before launching
long jobs so outputs and caches are written to scratch instead of your home directory.

## 1. Discover what you can use

From the repo root on `aics`:

```bash
bash scripts/aics_slurm_discovery.sh
```

Use the output to confirm:

- which `Account` values are valid for your user
- which partitions look like normal GPU queues
- where your repo lives on the cluster
- which virtualenv you want to use

Useful extra cluster commands from lab notes:

```bash
sinfo -Nl
scontrol show job <jobid>
resource_usage
test -f ~/bin/interactivejob && cat ~/bin/interactivejob
```

What they mean:

- `squeue -u $USER`: list your queued and running jobs
- `sinfo -Nl`: show nodes and their state in more detail than `sinfo -s`
- `scontrol show job <jobid>`: inspect one job's exact allocation and status
- `resource_usage`: lab-specific summary command, if installed on AICS
- `cat ~/bin/interactivejob`: shows a local helper script if your lab set one up

## 2. Do a short interactive GPU smoke test

```bash
bash scripts/aics_gpu_smoke_test.sh \
  --account <ACCOUNT> \
  --partition <GPU_PARTITION>
```

If you want a non-default virtualenv:

```bash
bash scripts/aics_gpu_smoke_test.sh \
  --account <ACCOUNT> \
  --partition <GPU_PARTITION> \
  --venv-path /path/to/venv
```

This allocates a short interactive GPU job, enters the repo, activates the env if requested,
runs:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
nvidia-smi
```

and then leaves you in an interactive shell on the allocated node.

Your friend's direct version is the same idea:

```bash
srun --pty -A <ACCOUNT> -t 60:00 --gpus=1 /bin/bash
```

The helper script is a safer first step because it also drops into the repo, activates your env,
and runs the PyTorch/CUDA checks for you.

## 3. Submit one eta job

Recommended helper:

```bash
bash scripts/aics_submit_s5_opd_eta.sh 0.1 \
  --account <ACCOUNT> \
  --partition <GPU_PARTITION>
```

If you want a non-default virtualenv:

```bash
bash scripts/aics_submit_s5_opd_eta.sh 0.1 \
  --account <ACCOUNT> \
  --partition <GPU_PARTITION> \
  --venv-path /path/to/venv
```

Direct `sbatch` also works:

```bash
mkdir -p logs/slurm logs/opd
sbatch --account=<ACCOUNT> --partition=<GPU_PARTITION> --chdir="$PWD" run_s5_opd_eta.sh 0.1
```

`run_s5_opd_eta.sh` launches `scripts/run_opd_sweep.sh` with:

- `N_TRAIN=15000000`
- `SUBSET_SIZE=8000000`
- `ETAS=<eta>`
- `OBJECTIVE=reverse_kl_tm`
- `TEACHER_LAW=distributional_noise`
- `EVAL_BATCH_SIZE=1024`
- `COMPILE=0`

If you want to try a preemptible queue later, the pattern is:

```bash
sbatch -p preempt --account=<ACCOUNT> --chdir="$PWD" run_s5_opd_eta.sh 0.1
```

Use that only after confirming that the `preempt` partition is valid for your account and that you
are comfortable with the job being interrupted and later resumed or requeued depending on cluster
policy.

## 4. Monitor and debug

```bash
squeue -u $USER
tail -f logs/slurm/s5-opd-eta-<jobid>.out
tail -f logs/opd/s5_opd_reverse_kl_tm_n8000000_eta0p1_distributional_noise_t1p0.log
scancel <jobid>
```

Other notes from the lab:

- `/etc/slurm/slurm.conf` is the cluster-wide Slurm config if you ever need to inspect defaults
- `crf@cs.columbia.edu` is the contact address your friend was given for cluster issues
- if dataset I/O is a bottleneck, copy hot datasets into scratch or node-local storage before training
