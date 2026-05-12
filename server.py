"""reefy-bench backend.

Tiny Flask + gunicorn-served service that fronts four stress tools:
    CPU    - sysbench cpu
    Memory - sysbench memory
    Disk   - fio (random 4K mix; sequential 1M mix)
    GPU    - gpu-fryer (HuggingFace; NVIDIA only)

Each test maps to a POST /api/run/<kind> endpoint. The handler spawns
a subprocess + a daemon thread that pumps stdout into an in-memory
job dict. The UI polls /api/jobs/<id> every 500ms. Jobs run
independently so the user can fire multiple tests at once.

State is intentionally non-persistent: a container restart wipes the
job history. Bench output is meant to be read live or copy-pasted,
not warehoused.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid as _uuid

from flask import Flask, jsonify, render_template, request

app = Flask(__name__, static_folder='static', template_folder='templates')

# fio scratch lives on the app's `data` volume (host path
# /mnt/reefy-data/apps/<instance>/data/) so the test writes hit real
# disk, not the container overlay tmpfs. The reconciler creates the
# mount; we just need the subdir.
FIO_SCRATCH_DIR = '/data/fio-scratch'
FIO_SCRATCH_FILE = os.path.join(FIO_SCRATCH_DIR, 'scratch')

# Cap per-job stdout to avoid unbounded growth from a wedged test.
# sysbench/fio output a few hundred lines max; gpu-fryer is chattier
# but still well under this ceiling for the supported run lengths.
MAX_STDOUT_LINES = 5000

# In-memory job registry. Keyed by short uuid. See _spawn_job for
# the dict shape.
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


# ── HW discovery ────────────────────────────────────────────────────

def _parse_cpuinfo() -> tuple[str, int]:
    """Return (model_name, logical_cpu_count). Model is the first
    `model name` line; cores counts `processor` entries (logical
    threads, what sysbench --threads talks about)."""
    model = ''
    cores = 0
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if not model and line.startswith('model name'):
                    model = line.split(':', 1)[1].strip()
                if line.startswith('processor'):
                    cores += 1
    except OSError:
        pass
    return model, cores or os.cpu_count() or 1


def _parse_meminfo() -> tuple[int, int]:
    """Return (mem_total_mb, mem_available_mb) from /proc/meminfo."""
    total_kb = avail_kb = 0
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    total_kb = int(line.split()[1])
                elif line.startswith('MemAvailable:'):
                    avail_kb = int(line.split()[1])
    except OSError:
        pass
    return total_kb // 1024, avail_kb // 1024


def _list_nvidia_gpus() -> list[dict]:
    """List NVIDIA GPUs via nvidia-smi. Returns [] if the binary is
    missing (no NVIDIA driver mounted into the container) or the
    query fails - we treat both as "no GPU" so the UI hides the
    GPU card without distinguishing the two."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi',
             '--query-gpu=index,name,memory.total',
             '--format=csv,noheader,nounits'],
            stderr=subprocess.DEVNULL, timeout=5).decode()
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 3:
            continue
        try:
            gpus.append({
                'idx': int(parts[0]),
                'vendor': 'NVIDIA',
                'model': parts[1],
                'mem_mb': int(parts[2]),
            })
        except ValueError:
            continue
    return gpus


# ── Job runner ──────────────────────────────────────────────────────

def _spawn_job(kind: str, cmd: list[str],
               env: dict | None = None,
               label: str | None = None) -> str:
    """Start `cmd` in a subprocess, return a job id. A daemon thread
    streams stdout (+ merged stderr) into the job's buffer.

    Jobs are intentionally fire-and-forget: there's no /api/jobs/<id>/cancel
    yet because the bench tests are time-bounded by their own flags
    (--time, --runtime, --duration) and a poll-driven UI surfaces
    failure quickly enough."""
    job_id = _uuid.uuid4().hex[:12]
    job: dict = {
        'id': job_id,
        'kind': kind,
        'label': label or kind,
        'cmd': cmd,
        'status': 'running',
        'returncode': None,
        'started_at': time.time(),
        'ended_at': None,
        'stdout': [],
    }
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    def _run() -> None:
        try:
            proc = subprocess.Popen(
                cmd,
                env=env or os.environ.copy(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                with _JOBS_LOCK:
                    job['stdout'].append(line.rstrip('\n'))
                    # Cap to bound memory if a test wedges and floods.
                    overflow = len(job['stdout']) - MAX_STDOUT_LINES
                    if overflow > 0:
                        del job['stdout'][:overflow]
            proc.wait()
            job['returncode'] = proc.returncode
            job['status'] = 'done' if proc.returncode == 0 else 'error'
        except Exception as exc:  # noqa: BLE001
            with _JOBS_LOCK:
                job['stdout'].append(
                    f'[bench-runner] {type(exc).__name__}: {exc}')
            job['status'] = 'error'
        finally:
            job['ended_at'] = time.time()

    threading.Thread(target=_run, daemon=True).start()
    return job_id


# ── HTTP routes ─────────────────────────────────────────────────────

@app.get('/')
def index():
    return render_template('index.html')


@app.get('/api/hw')
def hw():
    cpu_model, cores = _parse_cpuinfo()
    total_mb, avail_mb = _parse_meminfo()
    return jsonify(
        cpu_model=cpu_model,
        cores=cores,
        mem_total_mb=total_mb,
        mem_available_mb=avail_mb,
        gpus=_list_nvidia_gpus(),
    )


@app.post('/api/run/cpu')
def run_cpu():
    data = request.get_json(silent=True) or {}
    threads = max(1, int(data.get('threads') or 1))
    seconds = max(1, int(data.get('seconds') or 60))
    cmd = ['sysbench', 'cpu',
           f'--threads={threads}', f'--time={seconds}', 'run']
    return jsonify(job_id=_spawn_job('cpu', cmd,
                                     label=f'CPU x{threads}'))


@app.post('/api/run/mem')
def run_mem():
    data = request.get_json(silent=True) or {}
    size_gb = max(1, int(data.get('size_gb') or 2))
    threads = os.cpu_count() or 1
    # sysbench memory loops to hit --memory-total-size, so passing a
    # value larger than RAM is the standard way to exercise throughput
    # rather than measuring "first allocation".
    cmd = ['sysbench', 'memory',
           f'--threads={threads}',
           f'--memory-total-size={size_gb}G',
           '--memory-block-size=1M',
           'run']
    return jsonify(job_id=_spawn_job('mem', cmd,
                                     label=f'Memory {size_gb}G'))


@app.post('/api/run/disk')
def run_disk():
    data = request.get_json(silent=True) or {}
    seconds = max(5, int(data.get('seconds') or 60))
    profile = data.get('profile') or 'randrw_4k'
    os.makedirs(FIO_SCRATCH_DIR, exist_ok=True)

    common = [
        f'--filename={FIO_SCRATCH_FILE}',
        '--size=2G',
        '--ioengine=libaio', '--direct=1',
        f'--runtime={seconds}', '--time_based',
        '--group_reporting',
    ]
    if profile == 'randrw_4k':
        # 4k random 70/30 read/write with iodepth 32 - the canonical
        # "what IOPS can this disk sustain" probe.
        cmd = ['fio', '--name=randrw_4k',
               '--rw=randrw', '--rwmixread=70',
               '--bs=4k', '--iodepth=32', *common]
        label = 'Disk rand-4K'
    elif profile == 'seq_1m':
        # 1M sequential 50/50, lower iodepth - measures throughput
        # ceiling rather than IOPS.
        cmd = ['fio', '--name=seq_1m',
               '--rw=readwrite',
               '--bs=1M', '--iodepth=8', *common]
        label = 'Disk seq-1M'
    else:
        return jsonify(error=f'unknown profile {profile!r}'), 400
    return jsonify(job_id=_spawn_job('disk', cmd, label=label))


@app.post('/api/run/gpu')
def run_gpu():
    gpus = _list_nvidia_gpus()
    if not gpus:
        return jsonify(error='no NVIDIA GPU detected'), 404
    data = request.get_json(silent=True) or {}
    seconds = max(5, int(data.get('seconds') or 60))
    requested = data.get('gpu_indices')
    if requested is None:
        indices = [g['idx'] for g in gpus]
    else:
        try:
            indices = [int(i) for i in requested]
        except (TypeError, ValueError):
            return jsonify(error='gpu_indices must be a list of ints'), 400
    valid = {g['idx'] for g in gpus}
    bad = [i for i in indices if i not in valid]
    if bad:
        return jsonify(error=f'unknown gpu indices: {bad}'), 400
    if not indices:
        return jsonify(error='select at least one GPU'), 400

    # One gpu-fryer process per selected GPU, isolated via
    # CUDA_VISIBLE_DEVICES so each sees exactly one device. Running
    # them in parallel exercises the PSU + cooling envelope and gives
    # unambiguous per-GPU TFLOPs output in the UI.
    job_ids = []
    for idx in indices:
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(idx)
        model = next(g['model'] for g in gpus if g['idx'] == idx)
        label = f'GPU {idx} ({model})'
        cmd = ['gpu-fryer', '--duration', str(seconds)]
        job_ids.append(_spawn_job('gpu', cmd, env=env, label=label))
    return jsonify(job_ids=job_ids)


@app.get('/api/jobs/<job_id>')
def get_job(job_id: str):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return jsonify(error='job not found'), 404
        return jsonify(
            id=job['id'],
            kind=job['kind'],
            label=job['label'],
            cmd=job['cmd'],
            status=job['status'],
            returncode=job['returncode'],
            started_at=job['started_at'],
            ended_at=job['ended_at'],
            stdout='\n'.join(job['stdout']),
        )


if __name__ == '__main__':
    # Standalone dev mode; production runs under gunicorn (see Dockerfile).
    app.run(host='0.0.0.0', port=8500, debug=False)
