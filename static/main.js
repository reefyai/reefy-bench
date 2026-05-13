/* reefy-bench frontend. Drives the 4 stress-test cards.
 *
 * Each card is independent: clicking Run posts to /api/run/<kind>
 * and starts a polling loop on the returned job_id. Multiple cards
 * can be in flight at once - the only coupling is the per-card
 * Run-button disabled state while THAT card has a job running.
 *
 * State lives in the DOM (form values, button.disabled, output text).
 * No framework, no client-side store. Output is replaced wholesale
 * each poll because the server caps job stdout at MAX_STDOUT_LINES
 * so the payload stays small.
 */

const POLL_MS = 500;

function $(sel, root) { return (root || document).querySelector(sel); }
function $$(sel, root) {
    return Array.from((root || document).querySelectorAll(sel));
}

function fmtMB(mb) {
    if (typeof mb !== 'number') return '?';
    if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
    return mb + ' MB';
}

async function loadHw() {
    let hw;
    try {
        const r = await fetch('/api/hw');
        if (!r.ok) throw new Error('HTTP ' + r.status);
        hw = await r.json();
    } catch (e) {
        $('#hw-banner').textContent = 'hw probe failed: ' + e.message;
        return;
    }

    const parts = [
        hw.cpu_model || 'unknown CPU',
        hw.cores + ' cores',
        'RAM ' + fmtMB(hw.mem_total_mb) +
            ' (' + fmtMB(hw.mem_available_mb) + ' free)',
    ];
    if (hw.gpus && hw.gpus.length) {
        parts.push(
            hw.gpus.length + '× GPU: ' +
            hw.gpus
                .map((g) => g.model + ' ' + fmtMB(g.mem_mb))
                .join(', '),
        );
    } else {
        parts.push('no NVIDIA GPU');
    }
    $('#hw-banner').textContent = parts.join(' · ');

    // Default CPU threads to total logical cores - what the user
    // typically wants ("burn the whole CPU"). They can dial it back
    // to test ratios.
    const cpuInput = $(
        '[data-card="cpu"] input[name="threads"]');
    if (cpuInput) cpuInput.value = hw.cores || 1;

    // Default memory size to 2× MemAvailable so sysbench loops
    // through more than RAM, exercising throughput rather than
    // measuring a single allocation. Floor at 2 GB so tiny VMs still
    // run a meaningful test; round to GB.
    const memInput = $(
        '[data-card="mem"] input[name="size_gb"]');
    if (memInput) {
        const target = Math.max(
            2,
            Math.round(((hw.mem_available_mb || 0) / 1024) * 2),
        );
        memInput.value = target;
    }

    // GPU card: only show when an NVIDIA GPU is present. Build one
    // checkbox per detected GPU; all default checked so a single
    // "Run GPU" click stresses everything.
    const gpuCard = $('section.card[data-card="gpu"]');
    if (hw.gpus && hw.gpus.length) {
        gpuCard.hidden = false;
        const fs = $('#gpu-fieldset');
        // Clear (after the <legend>), then append fresh checkboxes.
        Array.from(fs.querySelectorAll('label')).forEach((el) =>
            el.remove());
        hw.gpus.forEach((g) => {
            const lbl = document.createElement('label');
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.name = 'gpu';
            cb.value = String(g.idx);
            cb.checked = true;
            lbl.appendChild(cb);
            lbl.appendChild(document.createTextNode(
                ' GPU ' + g.idx + ' (' + g.model + ')'));
            fs.appendChild(lbl);
        });
    }
}

// ── Card open/close ───────────────────────────────────────────────

function bindCards() {
    $$('.card .card-head').forEach((head) => {
        head.addEventListener('click', () => {
            const card = head.closest('.card');
            const body = $('.card-body', card);
            const open = card.classList.toggle('open');
            body.hidden = !open;
        });
    });
}

// ── Run / poll ────────────────────────────────────────────────────

function setStatus(card, label, cls) {
    const s = $('.status', card);
    s.textContent = label;
    s.className = 'status' + (cls ? ' ' + cls : '');
}

async function pollJob(jobId, onUpdate, onDone) {
    while (true) {
        let job;
        try {
            const r = await fetch('/api/jobs/' + jobId);
            if (!r.ok) throw new Error('HTTP ' + r.status);
            job = await r.json();
        } catch (e) {
            onUpdate({ stdout: '[poll] ' + e.message });
            onDone({ status: 'error' });
            return;
        }
        onUpdate(job);
        if (job.status !== 'running') {
            onDone(job);
            return;
        }
        await new Promise((res) => setTimeout(res, POLL_MS));
    }
}

function collectPayload(card, kind) {
    const payload = {};
    $$('input[type="number"], select', card).forEach((el) => {
        if (el.value === '') return;
        payload[el.name] =
            el.type === 'number' ? Number(el.value) : el.value;
    });
    if (kind === 'gpu') {
        payload.gpu_indices = $$(
            '#gpu-fieldset input[name="gpu"]:checked')
            .map((el) => Number(el.value));
    }
    return payload;
}

async function runSingle(card, kind, payload) {
    const pre = $('.output', card);
    pre.textContent = '';
    let resp;
    try {
        const r = await fetch('/api/run/' + kind, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        resp = await r.json();
        if (!r.ok) throw new Error(resp.error || ('HTTP ' + r.status));
    } catch (e) {
        setStatus(card, e.message, 'error');
        return false;
    }
    setStatus(card, 'running…', 'running');
    pollJob(
        resp.job_id,
        (job) => {
            pre.textContent = job.stdout || '';
            pre.scrollTop = pre.scrollHeight;
        },
        (job) => {
            setStatus(
                card,
                job.status + (job.returncode != null
                    ? ' (rc=' + job.returncode + ')'
                    : ''),
                job.status);
            $('.run-btn', card).disabled = false;
        },
    );
    return true;
}

async function runMulti(card, endpoint, payload, opts) {
    /* Multi-job runner: POSTs the payload, expects {job_ids: [...]},
     * renders one output block per job inside the card's .gpu-outputs
     * container, polls each in parallel. Used by both the GPU card
     * (one job per selected device) and the Simulate Workload card
     * (one job per subsystem). opts.runningLabel = string shown in
     * the card status while jobs are in flight. */
    const outsWrap = $('.gpu-outputs', card);
    outsWrap.innerHTML = '';

    let resp;
    try {
        const r = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        resp = await r.json();
        if (!r.ok) throw new Error(resp.error || ('HTTP ' + r.status));
    } catch (e) {
        setStatus(card, e.message, 'error');
        return false;
    }

    const total = resp.job_ids.length;
    setStatus(card,
        (opts && opts.runningLabel || 'running') +
            ' (' + total + ' job' + (total === 1 ? '' : 's') + ')…',
        'running');

    const finished = new Map();
    resp.job_ids.forEach((jid) => {
        const wrap = document.createElement('div');
        const lbl = document.createElement('div');
        lbl.className = 'gpu-output-label';
        lbl.textContent = 'job ' + jid;
        const status = document.createElement('span');
        status.className = 'gpu-output-status running';
        status.textContent = 'running';
        lbl.appendChild(status);
        const pre = document.createElement('pre');
        pre.className = 'gpu-output';
        wrap.append(lbl, pre);
        outsWrap.appendChild(wrap);

        pollJob(
            jid,
            (job) => {
                if (job.label) {
                    lbl.firstChild.nodeValue = job.label + ' ';
                }
                pre.textContent = job.stdout || '';
                pre.scrollTop = pre.scrollHeight;
            },
            (job) => {
                status.className = 'gpu-output-status ' + job.status;
                status.textContent = job.status +
                    (job.returncode != null
                        ? ' (rc=' + job.returncode + ')'
                        : '');
                finished.set(jid, job);
                if (finished.size === total) {
                    const anyErr = Array.from(finished.values())
                        .some((j) => j.status === 'error');
                    setStatus(card,
                        anyErr ? 'one or more failed' : 'all done',
                        anyErr ? 'error' : 'done');
                    $('.run-btn', card).disabled = false;
                }
            },
        );
    });
    return true;
}

async function runGpu(card, payload) {
    if (!payload.gpu_indices.length) {
        setStatus(card, 'pick at least one GPU', 'error');
        return false;
    }
    return runMulti(card, '/api/run/gpu', payload,
        { runningLabel: 'running on GPU' });
}

async function runWorkload(card, payload) {
    return runMulti(card, '/api/run/workload', payload,
        { runningLabel: 'simulating workload' });
}

function bindRunButtons() {
    $$('.run-btn').forEach((btn) => {
        btn.addEventListener('click', async () => {
            const card = btn.closest('.card');
            const kind = btn.dataset.kind;
            btn.disabled = true;
            setStatus(card, 'starting…');
            const payload = collectPayload(card, kind);
            let ok;
            if (kind === 'gpu')           ok = await runGpu(card, payload);
            else if (kind === 'workload') ok = await runWorkload(card, payload);
            else                          ok = await runSingle(card, kind, payload);
            if (!ok) btn.disabled = false;
        });
    });
}

document.addEventListener('DOMContentLoaded', () => {
    loadHw();
    bindCards();
    bindRunButtons();
});
