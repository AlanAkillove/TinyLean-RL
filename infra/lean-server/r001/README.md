# V3-R001 dedicated verifier instance (C′)

This stack is the **formal verifier for V3-R001 attempt-2** and nothing else. It is the same pinned
image as the shared server (`projectnumina/kimina-lean-server:2.0.0`,
`sha256:588a2cbbd10da509ed13f53ac136f8463fabff02dfe4eca535e7c47ae6e3ffd9` on fly122), started with the
smallest viable pool:

| setting | value | why |
| --- | --- | --- |
| `LEAN_SERVER_MAX_REPLS` | `1` | one reusable REPL: a pathological candidate, or a client-abandoned request whose Lean computation is still alive, can occupy at most one slot, and the runner's `docker restart` restores capacity by construction |
| `LEAN_SERVER_MAX_REPL_USES` | `-1` | no TTL recycling: a fresh REPL's Mathlib import would eat into the **frozen** 120 s candidate budget (owner section 4 forbids changing the solving budget) |
| `LEAN_SERVER_MAX_REPL_MEM` | `8G` | per-REPL `RLIMIT_AS`, as on the shared instance |
| `LEAN_SERVER_MAX_WAIT` | `60` | pool wait before HTTP 429, as on the shared instance |
| `LEAN_SERVER_ENVIRONMENT` | `prod` | same logging personality as the instance attempt-1 used |
| `mem_limit` / `memswap_limit` | `16g` / `16g` | covers one 8 GiB REPL + the server process + the import peak; swap disabled so a runaway is killed rather than thrashing the host |
| host port | `127.0.0.1:8010` | loopback only, distinct from the shared instance (8000) and the B0 scratch server (8001) |

The rationale, the attempt-1 forensics and the audit that produced these choices are in
`docs/v3/V3-R001_infrastructure_amendment_Cprime.md`.

## Runbook

```bash
# start (first time, or after a host reboot)
docker compose -f infra/lean-server/r001/compose.yaml up -d

# state + the pool size the server actually reports
docker compose -f infra/lean-server/r001/compose.yaml ps
docker exec tinylean-rl-lean-server-r001 env | grep LEAN_SERVER_MAX_REPLS

# the runner refuses to start unless this endpoint is this container:
curl -s http://127.0.0.1:8010/health
```

The runner (`scripts/v3_r001_rollout.py`) checks the instance identity before generating anything
(pinned image, `MAX_REPLS=1`, port `8010` published by *this* container) and restarts it only via
`docker restart tinylean-rl-lean-server-r001`. A formal launch therefore needs no docker privileges
inside the runner beyond that one command.

## What not to do

- Do not delete or `docker compose up -d --force-recreate` this container: its writable layer holds
  the image's Mathlib build cache, and a wipe re-runs that build (observed 2026-09-17 on the shared
  instance: ~500% CPU and >25 GiB RSS for tens of minutes, during which every `/verify` fails).
- Do not point the runner at the shared instance's port 8000: the identity check will abort the run
  (this is deliberate -- the shared 16-slot pool is a different, uncontrolled verifier).
- Do not lower `mem_limit` below the 8 GiB REPL cap plus the server process: the shared instance
  once had its whole pool killed by a too-tight live limit (2026-09-17).
