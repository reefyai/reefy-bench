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
    """Cycle disk-space allocation up and down using fallocate.

    The device-side metric for the Disk panel is `reefy_disk_used_pct`
    - a capacity gauge, not an IO meter. So to make that graph rise
    and fall during the workload, we don't need to push real IO; we
    just need to occupy and release blocks. `fallocate -l <N>` reserves
    real blocks on ext4/xfs (counts toward `df`) without writing any
    bytes, so the operation is near-instant and the metric tracks it.

    On each slice we unlink + fallocate to a new target size; this
    keeps the algorithm simple (always-fresh file, no shrink-vs-grow
    branching) and the change is immediately visible to statvfs."""
    end = time.time() + duration
    try:
        free = shutil.disk_usage(DATA_DIR).free
    except OSError as exc:
        _print('disk-wave', f'cannot stat {DATA_DIR}: {exc}')
        return
    # Upper bound: never allocate more than 70% of what was free
    # at start. Leaves headroom for OS bookkeeping + other apps.
    max_bytes = max(DISK_MIN_BYTES, int(free * DISK_FRACTION))
    max_gb = max(1, max_bytes // (1024 ** 3))
    os.makedirs(DATA_DIR, exist_ok=True)
    _print('disk-wave',
           f'cycling allocation up to {max_gb}G ({DISK_FRACTION:.0%} of '
           f'{free // (1024 ** 3)}G free); '
           f'fallocate reserves blocks without writing bytes')

    # Fractions of max_bytes per slice. Mix of small and large so the
    # graph traces a clear sawtooth rather than a monotonic ramp.
    levels = [20, 60, 10, 80, 40, 70, 30]
    i = 0
    while time.time() < end:
        pct = levels[i % len(levels)]
        target = max(DISK_MIN_BYTES, int(max_bytes * pct / 100))
        target_gb = max(1, target // (1024 ** 3))
        slice_s = _slice_remaining(end, 10)
        _print('disk-wave',
               f'slice {i}: fallocate {target_gb}G ({pct}% of allowance) '
               f'for {slice_s}s')
        try:
            os.unlink(DISK_FILE)
        except FileNotFoundError:
            pass
        r = subprocess.run(
            ['fallocate', '-l', str(target), DISK_FILE],
            capture_output=True, text=True)
        if r.returncode != 0:
            _print('disk-wave',
                   f'fallocate failed (rc={r.returncode}): '
                   f'{r.stderr.strip() or r.stdout.strip()}')
            return
        time.sleep(slice_s)
        i += 1
    # Best-effort cleanup so we don't squat on space after the run.
    try:
        os.unlink(DISK_FILE)
        _print('disk-wave', 'released allocation')
    except FileNotFoundError:
        pass
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
