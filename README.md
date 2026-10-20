# p8-hygda-pixel-colab-bus

Control plane for driving a Google Colab GPU session from a local agent.

**This repository contains job descriptions and code only.** No datasets, no model weights,
no credentials. It is public precisely so the Colab worker needs no token to read it.

## Why a repo instead of just Google Drive

Google Drive is asymmetric. `Colab → Drive → local` is fast and reliable: the FUSE client
uploads on `close()` and Drive Desktop pulls the change down within seconds. The reverse,
`local → Drive → Colab`, depends on Colab's FUSE client invalidating a cached directory
listing, which has no documented TTL and is not verifiable in advance.

So the two legs use different transports:

| Leg | Carries | Transport |
|---|---|---|
| down | job JSON, agent code, payload code | this repo (`git ls-remote` — a direct, uncached request) |
| up | logs, progress, results, checkpoints | Google Drive |

## Layout

```
agent/
  supervisor.py    keeps the agent alive; re-pulls and relaunches when agent code changes
  colab_agent.py   polls for jobs, runs them, streams logs to Drive, writes result.json
  schema.py        job/result dataclasses          ) mirrored verbatim from the main repo's
  driveio.py       sentinel-verified log chunks    ) hygdap/bus/ — never edit them here
  classify.py      failure taxonomy                )
payload/
  probe.py         report the runtime; proves the bus works
  selftest.py      mint a known-good dependency lock, resolve the SD-1.5 mirror
  train_lora.py    the actual training job
jobs/              one JSON per submitted job; <job_id>.cancel requests cancellation
```

## How a session starts

In Colab, open the Drive-hosted notebook `p8-hygda-pixel/p8_colab_agent.ipynb`, set
Runtime → Change runtime type → GPU, and run its single cell:

```python
from google.colab import drive; drive.mount('/content/drive')
exec(open('/content/drive/MyDrive/p8-hygda-pixel/bootstrap.py').read())
```

That cell never changes. `bootstrap.py` clones this repo and hands off to the supervisor,
which then picks up whatever work is pushed here — including updates to its own code.

## Contracts worth knowing

- A job is bound to code by **landing in the same commit** as any fix it exercises. A retry
  always gets a new `job_id`; the agent never re-runs a `job_id` it has already claimed.
- **Every terminal path writes `result.json`** — success, failure, timeout, cancellation,
  or an agent crash. The failures that motivated this design left no trace at all.
- Logs are **immutable numbered chunks** ending in a `#EOC <seq> <sha1> <lines>` sentinel.
  A reader that cannot verify the sentinel skips the chunk and retries, so a partially
  synced file can never be misread as truncated output.
- Exit codes: agent `75` = relaunch me, `70` = fatal, `0` = clean.

Part of the P8 HyGDA-Pixel project (prompt-to-layout design of pixelated circularly polarized patch antennas).
