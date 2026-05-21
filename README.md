# leonardo-hpc-survival-guide

Leonardo HPC Deep Learning & Vision Research Cheatsheet
A concise, battle-tested engineering guide for running heavy Deep Learning, Video Understanding, and Large Multimodal Model (LMM) inference pipelines on the CINECA Leonardo Supercomputer (NVIDIA A100 Cluster) without container overheads.

Designed for first-time HPC users and computer vision researchers shifting from local GPUs to enterprise-level HPC infrastructure.

---

> ⚠️ **Important:** This guide is under continuous revision. All the info below was tested on **ISCRA CINECA Leonardo A100**.

---

## 📑 Table of Contents

1. [The 2FA Tunneling Maze & SSH Automation](#-1-the-2fa-tunneling-maze--ssh-automation)
2. [Strategic Storage Allocation & Filesystem Blueprint](#-2-strategic-storage-allocation--filesystem-blueprint)
3. [High-Throughput Data Transfer (HPC to Local/WSL via Reverse rsync)](#-3-high-throughput-data-transfer-hpc-to-localwsl-via-reverse-rsync)
4. [Silent Performance Killers: Local vs. Enterprise GPU Optimization](#️-4-silent-performance-killers-local-vs-enterprise-gpu-optimization)
5. [The Geometry of Data Scale: Video Length vs. Model Calls](#-5-the-geometry-of-data-scale-video-length-vs-model-calls)
6. [Two-Layer SSH Port Forwarding for Jupyter Development](#-6-two-layer-ssh-port-forwarding-for-jupyter-development)
7. [Production-Ready Slurm Bash Script (`job.sh`)](#-7-production-ready-slurm-bash-script-jobsh)
8. [Essential Slurm & Cluster Navigation Reference](#-8-essential-slurm--cluster-navigation-reference)
9. [References & External Resources](#-9-references--external-resources)

---

## 🔑 1. The 2FA Tunneling Maze & SSH Automation

Leonardo utilizes a tight two-factor authentication (2FA) mechanism via `step-cli`. Network fluctuations or terminal idling will often throw a `client_loop: send disconnect: Broken pipe` error, cutting your live terminal session.

### Restoring a Broken SSH Session safely

If your terminal disconnects, your background job does NOT die if it was submitted via Slurm (`sbatch`). To reconnect and attach back to your execution logs without disrupting the node:

1. Renew your local step token (Bypass the common flag/issuer errors by running the configuration-agnostic command):

```bash
step ssh login username
```

Select `cineca-hpc (OIDC)` from the interactive menu and complete the browser authentication.

2. Tunnel into the login node using the explicit hostname:

```bash
ssh username@login.leonardo.cineca.it
```

3. Verify job status and re-attach to live logs:

```bash
cd $SCRATCH/your_project_dir
squeue -u username
tail -f logs/your_job_id.out
```

### 🪄 Pro-Tip: SSH Configuration for One-Command Automation

To bypass manual tünel management and login directly using a single `ssh leonardo` command, append the following snippet to your local machine's `~/.ssh/config` file:

```
Host leonardo
  HostName login.leonardo.cineca.it
  User username
  CertificateFile ~/.step/ssh/username-cert.pub
  IdentityFile ~/.step/ssh/username
  ProxyCommand bash -c 'step ssh login username --provisioner cineca-hpc >/dev/null 2>&1; nc %h %p'
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
```

---

## 💾 2. Strategic Storage Allocation & Filesystem Blueprint

Deep Learning workloads—especially multi-scene video processing pipelines—can trigger severe I/O bottlenecks or storage exhaustion if targeted incorrectly. Leonardo enforces strict quotas across partitions:

| Partition | Environment Variable | Best Used For | Quota / Trait | Retention Policy |
|-----------|---------------------|---------------|---------------|-----------------|
| Home | `$HOME` | Dotfiles, personal `.bashrc`, SSH configurations, custom binaries. | 50 GB | Permanent |
| Work | `$WORK` | Shared project codebase, team packages, collaborative models. | 40 TB (shared) | Purged 6 months after project close |
| Scratch | `$SCRATCH` | High-performance dynamic execution, temporary datasets, intermediate training/inference outputs. | High-capacity | Auto-deleted after ~40 days |
| Fast | `$FAST` | High-speed NVMe storage, optimized for I/O-heavy reads (e.g., pre-trained foundation model weights). | 1 TB per project | Same as `$WORK` |

### Critical Practice for Large-Scale Video Processing

Never invoke video parsing pipelines (e.g., PySceneDetect, FFmpeg subprocesses, OpenCV frame decoders) inside `$HOME`. Heavy disk read/write loops must happen exclusively within your high-speed scratch or fast workspace:

```bash
cd /leonardo_scratch/large/userexternal/username/
```

---

## 🚀 3. High-Throughput Data Transfer (HPC to Local/WSL via Reverse rsync)

Due to Leonardo's rigorous security protocols, standard outbound `scp` or direct unauthenticated tunneling requests are blocked. To securely pull massive video weights, frame folders, or `.pt` checkpoints back to your local machine (WSL), data synchronization must be initiated from your local machine, leveraging the Smallstep VPN agent paired with `rsync`.

> ⚠️ **CRITICAL RULE:** Execute the following commands exclusively inside your **local terminal (WSL/Local Linux)**, never inside the remote Leonardo environment.

### Step-by-Step Data Retrieval Protocol

1. Initialize the local SSH Agent and Re-authenticate with CINECA:

```bash
eval $(ssh-agent)
step ssh login username --provisioner cineca-hpc
```

(Complete the CINECA OTP/Password challenge if prompted).

2. Execute Reverse rsync to Pull Data Down: Navigate into your target local directory (`cd ~/socialiq_trial/temporal-si`) and ignite the data sync:

```bash
rsync -avP -e "ssh -o StrictHostKeyChecking=no" username@login.leonardo.cineca.it:/leonardo_scratch/large/userexternal/username/socialiq_scene_processing/ .
```

### 💡 Deep Dive: Decoding the rsync Flags

- **`-avP`**: Transfers files in archive mode (`a`), verbosely displays logs (`v`), and provides a real-time progress bar (`P`). Most importantly, if the connection drops mid-way through a multi-gigabyte transfer, re-running this command resumes precisely where it left off, unlike `scp` which restarts from scratch.
- **`-e "ssh -o StrictHostKeyChecking=no"`**: Completely bypasses host key fingerprint mismatches or handshake locks caused by remote network rotations.
- **📂 Trailing Slash on Source (`/`)**: Appending `/` to the end of the remote path (`.../socialiq_scene_processing/`) ensures that only the *contents* of the folder are copied, preventing unwanted nested folder structures.
- **📍 Current Directory Target (`.`)**: The trailing dot instructs the engine to clone everything directly into your current working directory on the local machine.

---

## 🏎️ 4. Silent Performance Killers: Local vs. Enterprise GPU Optimization

An optimized codebase on a local consumer GPU (e.g., RTX 4050 6GB) can become highly inefficient or downright sluggish when migrated directly to an NVIDIA A100 64GB cluster node.

### 🛑 The `torch.cuda.empty_cache()` Trap

When deploying small local models, developers often insert `torch.cuda.empty_cache()` at the end of custom loops as a safeguard against Out-of-Memory (OOM) errors.

- **The Problem on A100s:** In a high-throughput pipeline processing thousands of clips sequentially, calling `empty_cache()` forces PyTorch to completely stall its asynchronous CUDA execution graph, free physical memory blocks back to the OS, and re-allocate them on the next iteration.
- **The Impact:** It turns enterprise-grade hardware into a sluggish "stop-and-go" processor, causing up to 15% to 30% overhead loss during scaling.
- **The Fix:** Comment it out or remove it completely on Leonardo. Trust the native PyTorch memory caching manager; the A100's massive VRAM pool will easily handle internal caching fragmentation.

```python
# Remove this from iterative inference loops on HPC:
# torch.cuda.empty_cache()
```

### 📡 The Weights & Biases (WandB) Offline Mandate

Compute nodes on Leonardo are highly isolated and restricted from making outbound internet calls. Running online logging experiments will freeze or crash your execution script.

**The Solution:** Force WandB into offline tracking mode inside your Slurm bash environments:

```bash
export WANDB_MODE=offline
```

**Post-Run Sync:** Once your cluster computation finishes, you can synchronize your metrics and logs to the online cloud dashboard from an internet-enabled node using the CLI:

```bash
wandb sync path/to/offline-runs
```

---

## 🎬 5. The Geometry of Data Scale: Video Length vs. Model Calls

When benchmarking Large Multimodal Models (like Qwen2.5-VL-7B), scaling the duration of input videos scales your compute load **geometrically**, not linearly.

- **Short Actions (e.g., MSR-VTT, 15s):** Videos are short, static, and typically single-scene. 7,000 videos ≈ 7,000 distinct model inferences.
- **Complex Scenarios (e.g., SocialIQ, 60s):** Rich, unstructured interactions contain frequent camera cuts. Running a multi-scene parser can yield an average of **10-14 valid sub-scenes** per clip.
- **The Math:** 1,015 videos × 11 scenes = ~11,165 distinct Qwen evaluations. Combined with continuous disk writes via FFmpeg trimming and subsequent text vectorizations (e.g., SBERT), a structurally smaller dataset can demand a vastly superior compute footprint.

---

## 📓 6. Two-Layer SSH Port Forwarding for Jupyter Development

For interactive visual modeling or checking your extracted scenes, you can spawn a Jupyter instance directly inside an allocated GPU compute node. Because compute nodes live behind a private network, you must establish a two-layer tunnel:

1. Request a GPU Interactive Node Allocation:

```bash
salloc --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=02:00:00 --partition=boost_usr_prod
```

2. Identify the allocated Node ID (e.g., `r123c45`) via `echo $HOSTNAME` and spawn the server inside your active environment on a selected port:

```bash
jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
```

3. Establish the Layered Handshake Tunnels from your Local Host:

**Tunnel 1** (Local Terminal → Login Node): Maps local port `8888` to a temporary relay port `44444` on Leonardo.
```bash
ssh -L 8888:localhost:44444 username@login.leonardo.cineca.it
```

**Tunnel 2** (Inside that Login Session → Compute Node): Maps the relay port `44444` to the live port `8888` on the active GPU compute node.
```bash
ssh -L 44444:localhost:8888 r123c45
```

4. Fire up your local browser and point it to `http://localhost:8888`, pasting the execution token generated by step 2.

---

## 📄 7. Production-Ready Slurm Bash Script (`job.sh`)

This template is optimized for launching heavy Python/PyTorch inference or vision pipelines on a single production node equipped with multiple NVIDIA A100 GPUs. It handles purging default module states, loading optimized CUDA/CUDNN binaries, activating environments directly from shared paths, and exporting strict environment routing arrays.

Create a `job.sh` file inside your project workspace and configure the block below:

```bash
#!/bin/bash
#SBATCH --job-name=socialiq_inference   # Descriptive name of your pipeline
#SBATCH --nodes=1                       # Target single-node computation
#SBATCH --ntasks-per-node=1             # Main execution process count
#SBATCH --gres=gpu:4                    # Request all 4 NVIDIA A100 GPUs on the node
#SBATCH --cpus-per-task=32              # Multithreading CPU allocation for I/O operations
#SBATCH --time=12:00:00                 # Heavy safety margin execution walltime limit
#SBATCH --partition=boost_usr_prod      # Production queue for Leonardo Booster partition
#SBATCH --output=logs_%j.out            # Standard execution out log (%j logs unique Slurm Job ID)
#SBATCH --error=logs_%j.err             # Execution tracking error stream output

# 1. Purge default module environment mappings to prevent cross-compilation errors
module purge
ml profile/deeplrn

# 2. Dynamically bind structural high-performance computation dependencies
module load python/3.11.6--gcc--8.5.0
module load openmpi/4.1.6--gcc--12.2.0
module load nccl/2.19.1-1--gcc--12.2.0-cuda-12.1
module load cuda/12.1

# 3. Mount and ignite your lightweight localized python project workspace environment
source $WORK/venv_project/bin/activate

# 4. Bind system runtime environment configurations for offline operation and network routing
export WANDB_MODE=offline
export NCCL_NET=IB
export NCCL_DEBUG=INFO
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# 5. Fire your execution script passing your workspace parameters
python -u main.py
```

To dispatch this pipeline directly to Leonardo's background nodes, execute:

```bash
sbatch job.sh
```

---

## 📋 8. Essential Slurm & Cluster Navigation Reference

Quick reference for navigating the storage partitions and managing your heavy machine learning extraction jobs seamlessly:

| Command | Operational Purpose | Context / Best Practice |
|---------|--------------------|-----------------------|
| `cd $SCRATCH` | 📂 Warp straight to high-speed scratch space. | Always execute this immediately after logging in. Never run your training or processing loops in `$HOME`. |
| `rm -rf <dir_name>` | 🧹 Absolute and permanent directory purge. | Use with caution to clean up failed duman testi trials, cached frame remnants, or conflicting lock files before a major `sbatch` run. |
| `sbatch job.sh` | 🚀 Submit a batch script to the queue. | Dispatches your heavy execution pipeline to the background allocation nodes (e.g., NVIDIA A100 nodes). |
| `squeue -u username` | 📋 Monitor your active and pending jobs. | Check this to ensure your submitted cluster job status is officially set to `R` (Running) or to see your queue priority. |
| `tail -n 30 logs_<ID>.out` | 🔍 Peek at the trailing edge of execution logs. | Essential for structural diagnostics, confirming active checkpointing, or debugging early script failures. |
| `scancel <JOB_ID>` | 🛑 Instant process termination. | Safely kills your running cluster job if logs indicate an unintended infinite loop, bad configuration, or dynamic memory leaks. |

---

## 📚 9. References & External Resources

To deep dive into Leonardo's custom system libraries, environment profiles, and standard network/storage architecture policies, consult the official resources below:

- 📖 [CINECA HPC Documentation] – Official platform infrastructure, partition specifications, and baseline Slurm orchestration rules.
- ⚡ [Best Practice Guide to Running AI Workloads on Leonardo CINECA] – European HPC Application Support Portal guide for optimizing native non-containerized distributed training/inference loops.
