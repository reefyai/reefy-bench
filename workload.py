"""Simulated mixed-workload waveforms.

Each `*_wave` function runs for a fixed total duration, cycling through
short slices that pick different load levels per slice. The goal is a
plausibly-realistic mixed load - not a flat stress - so the host's
metrics graph wiggles like a real app would.

Per subsystem:
    cpu  - sysbench cpu with thread-count cycling 20% -> 100% of cores
    mem  - stress-ng --vm-keep holding 20%-70% of RAM in slices
    disk - fio randrw bursts (8-20s) with short idles between
    gpu  - gpu-fryer bursts (8-25s) with short idles between

Invoked as a CLI module by the Flask backend's
POST /api/run/workload route: each subsystem becomes its own
subprocess so the UI can stream their stdouts side-by-side.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

# Disk file lives on the app's data volume (host LV), not container
# overlay. Same path the standalone Disk card uses.
DATA_DIR = '/data'
DISK_FILE = os.path.join(DATA_DIR, 'workload')

# How much of the data volume's free space the disk wave is allowed
# to touch. 70% of free was chosen as a balance: realistic for a
# device with users actually using their disk; still leaves room
# for OS bookkeeping, snapshots, and other apps.
DISK_FRACTION = 0.70
DISK_MIN_BYTES = 256 * 1024 * 1024  # don't run trivially small


def _print(tag: str, msg: str) -> None:
    print(f'[{tag}] {msg}', flush=True)


def _slice_remaining(end_ts: float, max_s: int) -> int:
    """Return the lesser of `max_s` and (end - now), at least 1s."""
    return max(1, min(max_s, int(end_ts - time.time())))


def cpu_wave(duration: int) -> None:
    """Cycle sysbench cpu through thread counts at 20% -> 100% of
    cores in 10s slices. Each slice runs to completion; if the
    duration runs out mid-slice the loop breaks at the next boundary."""
    end = time.time() + duration
    nproc = os.cpu_count() or 1
    levels = [20, 40, 60, 80, 100]
    i = 0
    while time.time() < end:
        pct = levels[i % len(levels)]
        threads = max(1, round(nproc * pct / 100))
        slice_s = _slice_remaining(end, 10)
        _print('cpu-wave',
               f'slice {i}: {threads} threads ({pct}% of {nproc} cores) '
               f'for {slice_s}s')
        r = subprocess.run(
            ['sysbench', 'cpu',
             f'--threads={threads}', f'--time={slice_s}', 'run'],
            capture_output=True, text=True)
        # Show the most interesting line of sysbench's output - the
        # events-per-second summary line - so the stream is readable
        # without scrolling through prime-finding boilerplate.
        for line in r.stdout.splitlines():
            if 'events per second' in line:
                _print('cpu-wave', '  ' + line.strip())
                break
        i += 1
    _print('cpu-wave', 'done')


def mem_wave(duration: int) -> None:
    """Hold varying fractions of RAM in 12s slices. stress-ng with
    --vm-keep holds the allocation for the slice duration rather than
    looping bandwidth tests."""
    end = time.time() + duration
    levels = [20, 50, 70, 35, 60]
    i = 0
    while time.time() < end:
        pct = levels[i % len(levels)]
        slice_s = _slice_remaining(end, 12)
        _print('mem-wave',
               f'slice {i}: holding {pct}% of RAM for {slice_s}s')
        subprocess.run(
            ['stress-ng', '--vm', '1',
             f'--vm-bytes', f'{pct}%',
             '--vm-keep',
             '--timeout', str(slice_s),
             '--metrics-brief'])
        i += 1
    _print('mem-wave', 'done')


def disk_wave(duration: int) -> None:
    """Random-rw bursts on the data volume with short idle gaps.
    File size is a fixed fraction of free space, computed once at
    start; fio re-uses that file across bursts so we don't pay
    fallocate cost on every slice."""
    end = time.time() + duration
    try:
        free = shutil.disk_usage(DATA_DIR).free
    except OSError as exc:
        _print('disk-wave', f'cannot stat {DATA_DIR}: {exc}')
        return
    target = max(DISK_MIN_BYTES, int(free * DISK_FRACTION))
    size_gb = max(1, target // (1024 ** 3))
    os.makedirs(DATA_DIR, exist_ok=True)
    _print('disk-wave',
           f'using {size_gb}G file at {DISK_FILE} ({DISK_FRACTION:.0%} '
           f'of {free // (1024 ** 3)}G free)')

    bursts = [8, 12, 15, 6, 20]
    idles = [4, 5, 3, 8, 4]
    i = 0
    while time.time() < end:
        burst = _slice_remaining(end, bursts[i % len(bursts)])
        _print('disk-wave',
               f'burst {i}: {burst}s random rw 50/50 on {size_gb}G file')
        subprocess.run(
            ['fio', f'--name=workload-{i}',
             f'--filename={DISK_FILE}',
             f'--size={size_gb}G',
             '--rw=randrw', '--rwmixread=50',
             '--bs=1M', '--ioengine=libaio', '--direct=1',
             '--iodepth=16',
             f'--runtime={burst}', '--time_based',
             '--group_reporting', '--minimal'])
        if time.time() >= end:
            break
        idle = _slice_remaining(end, idles[i % len(idles)])
        if idle > 0:
            _print('disk-wave', f'idle: {idle}s')
            time.sleep(idle)
        i += 1
    _print('disk-wave', 'done')


def gpu_wave(duration: int) -> None:
    """gpu-fryer bursts (8-25s) interleaved with idle gaps (3-8s).
    Caller pins the process to one GPU via CUDA_VISIBLE_DEVICES so
    multiple gpu_wave instances run in parallel exercise different
    cards independently."""
    end = time.time() + duration
    bursts = [15, 10, 20, 8, 25]
    idles = [5, 3, 6, 4, 5]
    i = 0
    while time.time() < end:
        burst = _slice_remaining(end, bursts[i % len(bursts)])
        _print('gpu-wave', f'burst {i}: {burst}s of gpu-fryer')
        subprocess.run(['gpu-fryer', str(burst)])
        if time.time() >= end:
            break
        idle = _slice_remaining(end, idles[i % len(idles)])
        if idle > 0:
            _print('gpu-wave', f'idle: {idle}s')
            time.sleep(idle)
        i += 1
    _print('gpu-wave', 'done')


_DISPATCH = {
    'cpu':  cpu_wave,
    'mem':  mem_wave,
    'disk': disk_wave,
    'gpu':  gpu_wave,
}


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in _DISPATCH:
        print('usage: workload.py <cpu|mem|disk|gpu> <duration_s>',
              file=sys.stderr)
        return 2
    try:
        dur = int(argv[2])
    except ValueError:
        print(f'invalid duration: {argv[2]!r}', file=sys.stderr)
        return 2
    if dur < 5:
        print('duration must be >= 5 seconds', file=sys.stderr)
        return 2
    _DISPATCH[argv[1]](dur)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
