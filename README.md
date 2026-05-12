# reefy-bench

CPU / memory / disk / GPU stress tests with a small web UI.

| Test   | Tool                                  | Notes                                                  |
|--------|---------------------------------------|--------------------------------------------------------|
| CPU    | `sysbench cpu`                        | Pick threads + duration.                               |
| Memory | `sysbench memory`                     | 1M block writes; default size auto-tuned to 2× RAM.    |
| Disk   | `fio`                                 | Random 4K IOPS profile + sequential 1M throughput.     |
| GPU    | [gpu-fryer](https://github.com/huggingface/gpu-fryer) | fp16 matmul on tensor cores; NVIDIA only. |

Each card has its own Run button; tests run concurrently if you fire
several at once. Live output streams into each card's panel. State is
in-memory only - restarting the container clears job history.

## Two ways to use it

**1. Inside the Reefy app catalog** (zero setup). Boot a machine into
Reefy OS, adopt it on [reefy.ai](https://reefy.ai), open the device's
**Install app** menu, pick **Bench**. A few clicks and you're in.
Reefy provisions the container, attaches the GPU via CDI on hosts that
have one, sets up the tunnel + access link, and hands you back a URL.

**2. Standalone on any Docker host**. The image is published publicly
to `ghcr.io/reefyai/reefy-bench:latest` - run it anywhere. GPU access
goes through CDI (the same mechanism Reefy itself uses), so you need
the NVIDIA Container Toolkit configured to generate a CDI spec at
`/etc/cdi/nvidia.yaml` (see [NVIDIA's CDI guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/cdi-support.html)).
Once that's in place:

```bash
# GPU machine (NVIDIA CDI configured)
docker run --rm -p 8500:8500 \
    --device nvidia.com/gpu=all \
    ghcr.io/reefyai/reefy-bench:latest

# CPU / disk / mem only - no GPU flag needed
docker run --rm -p 8500:8500 ghcr.io/reefyai/reefy-bench:latest

open http://localhost:8500
```

Without `--device nvidia.com/gpu=all` the GPU card auto-hides; the
other three tests keep working.

No auth: inside the Reefy app catalog the per-device tunnel + service
token handle access control; standalone you're responsible for not
exposing port 8500 to the open internet.
