# Handoff to fly90: unpushed E024 pause commits and fly122 push blocker

- Date: 2026-09-19 (UTC)
- From: fly122 worker session
- To: fly90 canonical session

## 1. Why this handoff exists

fly122 committed the E023 audit / E024 pause record but cannot push to
GitHub. The fly122 deploy key is still not registered on the repository
(git@github.com answers Permission denied (publickey) even though the key is
offered through the SOCKS proxy), the HTTPS remote has no stored credentials,
and the standing protocol forbids using a PAT from fly122. Please pick the
commits up over LAN and push them, or register the fly122 key (section 5).

## 2. Commits waiting on fly122 (branch p3-linux)

Base: b4196b2b9b0765f88602976a067cfb6782710029, currently equal to
origin/p3-linux, so the pickup is a clean fast-forward.

- b7267e4 exp: record E023 by-prefix audit (clean) and pause E024 theta0 after
  Lean server wedge — adds docs/e024_status.md (86 lines)
- the follow-up docs commit on top — adds this handoff file

Working tree is clean; no other tracked changes exist.

## 3. Suggested pickup procedure (run on fly90)

    git fetch ssh://fly@10.3.25.122/home/fly/ZJQ/TinyLean-RL p3-linux
    git merge --ff-only FETCH_HEAD
    git push origin p3-linux

Use the fly90 deploy key path for the final push; it works there.

## 4. Untracked evidence available for rsync

Under experiments/results/ on fly122 (that directory is gitignored by design):

- e023_by_prefix_bug_audit.json (3,382 bytes, sha256 cf116ee5...) — E023
  retrospective audit artifact (affected_total = 0)
- e024_minif2f_theta0.partial.json (6,536,537 bytes, sha256 e79f3d68...) —
  paused theta0 partial (chunks 1-8: 128/244 theorems, 512 records)
- e024_minif2f_theta0_run.log (82,749 bytes, sha256 59db3fb7...) — preserved
  run log of the Lean server wedge incident

E023 primary artifacts were not modified (sha256 verified against
experiments/manifests/e023_holdout.yaml).

## 5. fly122 deploy key (for future direct pushes)

Public key file on fly122: ~/.ssh/id_ed25519_github.pub

    ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJzwlQGX04ozymBJfFA/5LMSbZWJ6hDQyogNbi1gXr4K fly@122-deploykey

Fingerprint: SHA256:Eo1gX0LjyDIr/LY+b0DX19HgoTjhSSY3YjEtyTlO8DA

Registering it as a repository deploy key with write access would give fly122
a direct push path for future worker commits.

## 6. fly122 operational state at handoff

- E023 RETRO AUDIT: CLEAN — affected_total = 0 across all four primary
  artifacts; no E023 statistics recomputation required.
- E024 theta0: PAUSED by operator instruction after the Lean server wedge;
  no adjudication and no paired analysis were run; no E024 result is claimed.
  A rerun from scratch is required (see docs/e024_status.md section 5).
- Lean server: restarted 13:06 UTC; healthy (health endpoint returns ok,
  trivial verification 4.7 s).
- GPU: 0 compute processes; systemd unit fly122-e024-theta0 is inactive.
