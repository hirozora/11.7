"""Run a list of model jobs over GPU slots (simplified from graph_prediction_research's scr_queue.py).

--jobs is a JSON file with a list of {"name": ..., "args": [...]} entries; each job runs
`python src/<script> <args> --device cuda:<gpu>` with its log at <log_dir>/<name>.log. Slots = gpus x jobs_per_gpu. After
the first failure no new job starts; the queue waits for the running ones and exits with an error.
"""
import argparse
import json
import subprocess
import sys
import time

from common import ROOT, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--script", required=True)
    parser.add_argument("--gpus", nargs="+", required=True)
    parser.add_argument("--jobs-per-gpu", type=int, required=True)
    parser.add_argument("--log-dir", required=True)
    args = parser.parse_args()
    jobs = load(ROOT / args.jobs)
    log_dir = ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    free = [g for g in args.gpus for _ in range(args.jobs_per_gpu)]
    running, failed, done = {}, [], []
    pending = list(jobs)
    while pending or running:
        while pending and free and not failed:
            job = pending.pop(0)
            gpu = free.pop(0)
            log = open(log_dir / f"{job['name']}.log", "w")
            cmd = [sys.executable, str(ROOT / "src" / args.script), *job["args"], "--device", f"cuda:{gpu}"]
            running[job["name"]] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT), gpu, log)
            print(f"{time.strftime('%H:%M:%S')} start {job['name']} on cuda:{gpu}", flush=True)
        if failed and not running:
            break
        time.sleep(5)
        for name, (proc, gpu, log) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del running[name]
            free.append(gpu)
            (done if code == 0 else failed).append(name)
            print(f"{time.strftime('%H:%M:%S')} {'done' if code == 0 else 'FAILED'} {name} (exit {code})", flush=True)
    with open(log_dir / "queue_status.json", "w", encoding="utf-8") as f:
        json.dump({"done": done, "failed": failed, "not_started": [j["name"] for j in pending]}, f, indent=1)
    if failed:
        raise SystemExit(f"failed jobs: {failed}")
    print("all jobs done", flush=True)


if __name__ == "__main__":
    main()
