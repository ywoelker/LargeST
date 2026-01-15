import os
import pathlib
import queue
import threading
import subprocess
import time

# ========= CONFIG =========
# GPUs you want to use concurrently (e.g., first five only):
GPU_IDS = list(range(8)) # full set on the box
MAX_CONCURRENT = 6                   # run on 5 GPUs at a time
CONDA_ENV = "pyg"
SHELL = "zsh"                         # you said you use zsh
RC_FILE = "~/.zshrc"                  # so conda activate works
OMP_NUM_THREADS = "8"

# If some models should prefer certain GPUs (e.g., H100s), list their IDs:
PREFERRED_GPUS = {
    # "D2STGNN": [0,1],   # example
    # "GMAN":    [0,1],
}
# ==========================

LOG_DIR = pathlib.Path("scripts_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def build_cmd(model_name: str, mask_name: str, mask_iter: int, div_weight: float) -> tuple[str, str]:
    job_name = f"{model_name}_div{div_weight}_{mask_name}_{mask_iter}"

    # base shell setup (zsh)
    base = (
        f'echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; '
        f'source {RC_FILE}; '
        f'conda activate {CONDA_ENV}; '
    )

    # main training command (mirrors your original)
    cmd = (
        f'python experiments/{model_name.lower()}/main.py '
        f'--dataset SD --years 2019 --device cuda '
        f'--model_name {model_name} --run_description {job_name} '
        f'--mask_name {mask_name} --mask_iter {mask_iter} '
        f'--use_metadata True '
        f'--input_dim 38 '
        f'--static_prefilter_mode static_dsn '
    )

    # extra args like your script
    if model_name.lower() == "dstagnn":
        cmd += " --input_dim 1"
    elif model_name.lower() == "gman":
        cmd += " --bs 16"
    elif model_name.lower() == 'dsgnn':
        cmd += (
            f" --n_rand_dim 16 --static_prefilter_mode static_dsn "
            f'--n_rand_dim 16 '
            f'--n_hid 32 '
            f'--n_context 128 --n_context_emb 128 '
            f'--dsn_div_weight {div_weight} '
        )

    return job_name, base + cmd

# Build the sweep (your same nested loops)
JOBS: list[tuple[str, str, str]] = []
for div_weight in [0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 1]:
    for model_name in ['DSGNN']:
        job_name, cmd = build_cmd(model_name, 'point_missing_050', 2, div_weight)
        JOBS.append((model_name, job_name, cmd))

# GPU pool (all physical GPUs)
gpu_pool = queue.Queue()
for g in GPU_IDS:
    gpu_pool.put(g)

# limit to MAX_CONCURRENT by only spawning that many workers
NUM_WORKERS = min(MAX_CONCURRENT, len(GPU_IDS))

def pick_gpu_for(model: str) -> int:
    """Try to pick a preferred GPU if defined; else any free GPU."""
    preferred = PREFERRED_GPUS.get(model)
    if preferred:
        # attempt to pick one of the preferred from the pool atomically
        grabbed = None
        tmp = []
        while not gpu_pool.empty():
            g = gpu_pool.get()
            if grabbed is None and g in preferred:
                grabbed = g
            else:
                tmp.append(g)
        for g in tmp:
            gpu_pool.put(g)
        if grabbed is not None:
            return grabbed
    # fallback: any GPU
    return gpu_pool.get()

def return_gpu(gid: int):
    gpu_pool.put(gid)

def run_one(model_name: str, job_name: str, cmd: str):
    gid = pick_gpu_for(model_name)
    log_path = LOG_DIR / f"{job_name}.out"
    print(f"[LAUNCH] {job_name} on GPU {gid} -> {log_path}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gid)
    env["OMP_NUM_THREADS"] = OMP_NUM_THREADS

    # use zsh so your conda activate/path matches your shell
    full_cmd = [SHELL, "-lc", cmd]

    # --- write header line(s) to the log first ---
    with open(log_path, "w") as logf:
        logf.write(f"[COMMAND] {cmd}\n")
        logf.write(f"[GPU] {gid}\n")
        logf.write("="*80 + "\n\n")

    # --- append process output to same log file ---
    with open(log_path, "a") as logf:
        proc = subprocess.Popen(
            full_cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        ret = proc.wait()

    print(f"[DONE {ret}] {job_name} (GPU {gid})")
    return_gpu(gid)

def worker(job_q: "queue.Queue[tuple[str,str,str]]"):
    while True:
        try:
            args = job_q.get_nowait()
        except queue.Empty:
            return
        try:
            run_one(*args)
        except Exception as e:
            print(f"[ERROR] {args[1]}: {e}")
        finally:
            job_q.task_done()

def main():
    job_q: "queue.Queue[tuple[str,str,str]]" = queue.Queue()
    for it in JOBS:
        job_q.put(it)

    threads = []
    for _ in range(NUM_WORKERS):
        t = threading.Thread(target=worker, args=(job_q,), daemon=True)
        t.start()
        threads.append(t)

    t0 = time.time()
    job_q.join()
    for t in threads:
        t.join(timeout=0.1)
    print(f"All jobs finished in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
