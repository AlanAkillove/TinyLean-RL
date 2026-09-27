# V5_R001_PRELAUNCH_AUDIT_V2

Status: **engineering feasibility only** — `S0`–`S3` executed under the owner's 2026-09-27 pre-launch
directive §10 authorization (fly122, consumed training prompts only). No capability holdout was
contacted, the 93-component reserve stays sealed, no scientific training result exists, and
`V5_R001_FORMAL_TRAINING_AUTHORIZED: NO`.

Host of record: **fly122** (RTX 3080 10 GB, 48 SM, 62 GiB RAM), stack pinned by
`experiments/manifests/p3_0_environment.yaml`. Docker oracle `tinylean-rl-lean-oracle-v5`
(`sha256:588a2cbb…3ffd9`, `LEAN_SERVER_MAX_REPLS=1`) healthy throughout.

Executed against `915622e` (S1 attempts and every labelled diagnostic), `616c0b5` (this report; S0
re-run + artifact commit). The dict below is the §16 field list, exactly.

```yaml
V5_R001_PRELAUNCH_AUDIT_V2:

  reward_definition:            # frozen owner intervention (draft §3/§4/§7), implemented + tested
    advantage: "A_i,t = A_outcome_i + lambda * I[t == first_token(tactic_i,j)] * (phi_i,j - g_bar)"
    outcome_term: "A_outcome_i = g_i - g_bar, g_bar over VALID candidates only"
    lambda_primary: 1
    phi: {success: +1, d1: -0.05, d2: -0.10}
    precedence: "d2 > d1 > success, exactly one write per token position"
    control: "lambda = 0 through the same R001 estimator (V1-equivalent outcome-only objective
              under the shared execution stack; never 'byte-identical to V1')"
    infra_censoring: "excluded from g_bar, loss masked to zero, infra counter,
                      n_valid < 2 => GROUP_SKIPPED_INFRA, identical in both arms"
    modules: [src/tinylean_rl/rl/process_credit.py, src/tinylean_rl/rl/r001_advantage.py,
              scripts/v5_r001_reward.py, scripts/v5_r001_online_credit.py,
              scripts/v5_r001_manager.py]
    prohibitions_respected: "no lambda tuning, no phi centering, no response-level summation,
                             no normalization search"

  holdout_H1:
    registry: experiments/manifests/v5/h1_holdout_pool.json
    sha256: 38939258f04df5d900e37010d57fbbf27737cf44a2121ac36b98eadc8a5dcdf1
    capacities: {128: supported, 192: supported, 256: supported}   # 4094 family components
    theta0_outcomes_used: 0
    sealed_reserve_touched: 0

  power_v2:
    artifact: experiments/manifests/v5/v5_r001_power_v2.json
    sha256: 549a9abe759a72f329fb876fa43ee3144f9cc06f99997f987c0798577fb09be8
    endpoints: [theorem-level rate delta (paired family bootstrap + exact sign-flip),
                pass@8 McNemar (exact, one-sided)]
    detectable_80pct:
      N128: {rate_delta_pp: 3.62-4.19, pass_at_n_pp: 5.42-7.65}
      N192: {rate_delta_pp: 3.13-3.47, pass_at_n_pp: 4.24-5.97}
      N256: {rate_delta_pp: 2.72-3.21, pass_at_n_pp: 3.65-5.34}
    recommended_N: "N = 192 planning target (owner decision; N = 256 buys ~1 - sqrt(192/256)
                    in the detectable delta)"
    proposed_practical_gate: "NOT FROZEN (owner §9); a threshold needs S1's between-arm noise"

  S0:
    artifact: experiments/results/v5_r001_smoke/s0/config_audit.json   # regenerated 2026-09-27 at
                                    # 616c0b5 with max_num_seqs audited: passed, 0 violations
    arms: [control(lambda=0), treatment(lambda=1)], arm_diff: {algorithm.r001_lambda: [0, 1]}
    parameters: {trainable: 596049920, dtype: bfloat16, arch: Qwen3ForCausalLM,
                 vocab: 151936, hidden: 1024, layers: 28, tied_embeddings: true}
    static_bytes: {bf16_params: 1192099840, bf16_grads: 1192099840, fp32_master: 2384199680,
                   adam_m_v: 4768399360, total: 9536798720, total_gib: 8.88}
    memory_settings_both_arms: {gradient_checkpointing: true, activation_offload: true,
                                param_offload: true, optimizer_offload: true,
                                use_dynamic_bsz: true, ppo_micro_batch_size_per_gpu: 1,
                                ppo_max_token_len_per_gpu: 5120,
                                log_prob_micro_batch_size_per_gpu: 1,
                                gpu_memory_utilization: 0.30, max_num_batched_tokens: 5120,
                                max_model_len: 5120, max_num_seqs: 1024}
    capture: {batch: 4 prompts x n=4, max_prompt_length: 1024, max_response_length: 4096,
              save_freq: -1, test_freq: -1}
    checks_passed: true   # reward identity (endpoint/phi/lambda/sentinels/tokenizer),
                          # arm expectations, no leftover GPU processes
    correction_vs_draft: "the draft §10.5 table assumes 751 632 384 parameters; the measured
                          model is 596 049 920 (tied embeddings), so the static subtotal is
                          8.88 GiB, not 11.2 GiB - the draft's number overstates by ~25 %"

  S1:
    artifact: experiments/results/v5_r001_smoke/s1/attempts.json
    attempts: 2, outcome_control_fit_10GB: NO
    diff_table:
      - attempt: 1
        knobs: "S1 baseline (all §10.2 memory knobs: gradient checkpointing, activation offload,
                param offload, optimizer offload, dynamic bsz, micro 1, token caps 5120,
                vLLM util 0.30)"
        outcome: FAILED_OOM   # vLLM profile run (gpu_model_runner.profile_run -> _dummy_run
                              # with max_num_tokens=5120 -> top-k/top-p sort workspace 1.74 GiB)
        peak_vram_mib: 8190
        seconds: 148.6
        evidence: "Tried to allocate 1.74 GiB, 1.65 GiB free, 7.97 GiB in use, 7.19 GiB by
                   PyTorch; max_num_seqs=1024"
      - attempt: 2
        knobs: "gpu_memory_utilization 0.30 -> 0.25 (one knob, §12)"
        outcome: FAILED_OOM
        peak_vram_mib: 8190
        seconds: 149.5
        evidence: "byte-identical allocation numbers to attempt 1 - the profile run happens
                   before the KV cache is sized, so the util knob is inert for this OOM"
    final_memory_settings: "none - no configuration inside the §10.2 allowed list, and no
                            labelled diagnostic on top of max_num_seqs=16, completes an
                            optimizer step on the 10 GB card"
    seconds_per_step: "not measurable: no recorded attempt or diagnostic completed a step
                       (149 s and 150 s to the attempt-side OOM; probe4 reached the first
                       optimizer step after 329 s, probe6 after 445 s)"
    diagnostics_outside_the_allowed_list:   # labelled diagnostics, never S1 attempts
      - tag: max_num_seqs16/attempt1
        knobs: "max_num_seqs 1024 -> 16 (NOT on the §10.2 list)"
        outcome: "vLLM profile run cleared; died in the reward row guards (numpy truth-value
                  bug, fixed in 915622e) - no memory evidence (peak 8306 MiB)"
      - tag: max_num_seqs16/attempt2
        knobs: "max_num_seqs 1024 -> 16 (NOT on the §10.2 list)"
        outcome: "OOM moved to loss.backward() (dp_actor.py:466): needs 1.93 GiB with 363 MiB
                  free; peak 9512 MiB - the 5120-token lm_head logits term"
      - tag: probe3
        knobs: "max_num_seqs=16 + ppo_max_token_len_per_gpu 5120 -> 2048 (allowed knob)"
        outcome: "AssertionError in rearrange_micro_batches: max_token_len must be >=
                  max_seq_len (2048 < 5120) - in this fork the token cap cannot go below the
                  padded sequence length, so it is not a lever here. The online reward DID run
                  end-to-end on this attempt: 16 rows, 7 submitted, 2 verified, 0 infra-censored"
      - tag: probe4
        knobs: "max_num_seqs=16 + use_fused_kernels=True + fused_kernel_options.impl_backend=torch"
        outcome: "backward CLEARED (the chunked lm_head removes the 1.93 GiB logits term); OOM
                  moves to the first optimizer step (dp_actor.py:478 -> adam.py:176
                  _init_group: state['exp_avg'] = torch.zeros_like(...)): needs 62.00 MiB with
                  27.06 MiB free, 45.37 MiB reserved-unallocated; peak 9848 of 9871 MiB;
                  rollout+reward completed (rows 16, submitted 7, verified 2, mapped 82,
                  no_code 9, infra_censored 0)"
      - tag: probe5
        knobs: "probe4 + PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (allocator env var,
                zero science impact; the allocator's own OOM hint)"
        outcome: "hard stack incompatibility: vLLM's sleep-mode allocator asserts at
                  device_allocator/cumem.py:147 (CuMemAllocator.get_instance) during model init,
                  before any rollout - expandable segments cannot be used with this stack"
      - tag: probe6
        knobs: "probe4 + actor_rollout_ref.rollout.enforce_eager=True (allowed knob)"
        outcome: "same 62.00 MiB Adam-state allocation fails again, now with 55.06 MiB free
                  (enforce_eager frees ~28 MiB of CUDA-graph pool; the deficit is larger); peak
                  9820 MiB; rollout+reward completed (rows 16, submitted 6, verified 2)"
    allowed_knob_closure: "every §12-allowed knob is at its memory-optimal value, tried, or
                           structurally inert: gradient checkpointing, activation offload,
                           param/optimizer offload, dynamic bsz, micro-batch 1 and the 5120
                           token caps are the frozen baseline; gpu_memory_utilization is inert
                           for the profile-run OOM (identical allocation numbers at 0.30/0.25);
                           ppo_max_token_len_per_gpu cannot go below 5120
                           (seqlen_balancing.py:277 assert); entropy_from_logits_with_chunking
                           and entropy_checkpointing configure only the non-fused logits path
                           (dp_actor.py:78-81 reaching :207-210), which use_fused_kernels
                           bypasses (dp_actor.py:188); enforce_eager was tried in probe6"
    finding: "the binding constraint is the update phase: with max_num_seqs=16 and the
              fused/chunked lm_head the rollout, reward and backward all run, but the first
              optimizer step cannot materialize its Adam state on the 9.64 GiB card (dies 62 MiB
              short at ~99.5 % occupancy). The static training footprint alone (bf16 params +
              grads + fp32 master + Adam m/v) is 8.88 GiB, so under the frozen science stack the
              10 GB card has no headroom for the update"

  S2:
    status: NOT RUN - blocked by S1's STOP_FOR_OWNER (both conditions need the GPU stack that
            OOMs; re-running them on the diagnostic stack would make the parity claim about a
            stack the owner has not authorized)
    lambda0_parity: "offline fixtures pass bit-for-bit at lambda=0 vs the pinned estimator and vs
                     the V1 outcome-only formula (tests/test_v5_r001_process_credit.py,
                     script v5_r001_online_credit replay: 0 mismatches over 5488 stored
                     candidates)"
    lambda1_tensor_valid: "constructed + pinned by fixtures (counters non-zero, precedence
                           respected, mask respected); no live-stack evidence yet"
    loss_delta: null
    gradient_delta: null

  S3:
    status: "stored-data variant COMPLETE (CPU/verifier, no generation, no GPU);
             smoke-generated variant blocked with S1"
    artifact: experiments/manifests/v5/v5_r001_online_replay.json
    data: "5488 stored consumed-training candidates (Phase-B raw items + V1 rollout sources)"
    counters: {mapped: 25197, ambiguous: 0, outside_response: 56, retokenization_mismatch: 0,
               conflict: 56, infra_censored: 3048, no_code: 0}
    chain_mismatches_vs_frozen_phase_b: 0
    advantage_tensor_probe: {rows: 4, width: 3720, credit_tokens_applied: 53,
                             dropped_by_mask: 0, out_of_range: 0,
                             lambda0_equals_v1_bit_for_bit: true}
    live_rollout_counters: "one live step's online reward ran on the diagnostic stack:
                            rows 16, submitted 7, verified 2, no_code 9, mapped 82,
                            retokenization_mismatch 2, infra_censored 0"

  compute_projection:
    structure: "40-60 steps x 2 arms x 2 seeds = 4 runs"
    measured_cost: "not derivable - no optimizer step completed on this card"
    verifier_floor: "16 candidates/step x 9.85 s/candidate (measured, serialized) ~ 2.6 min/step
                     of verifier time alone; 50 steps x 4 runs ~ 8.8 h verifier-bound, before
                     generation, log-prob and update phases"

  recommended_seeds_per_arm: "2 (unchanged from §16); the S1-cost-based revision cannot be
                              computed - no step cost exists yet"
  recommended_horizon: "40-60 steps (unchanged); same reason"
  recommended_holdout_N: "192 (power v2 planning target, H1 capacity supports 128/192/256)"

  remaining_blockers:
    - "10 GB fit (STOP condition, §12): no §12-compliant configuration completes an optimizer
       step. Both recorded attempts OOM at the vLLM profile run; every labelled diagnostic on
       top of max_num_seqs=16 clears the rollout, reward and backward phases and then dies in
       the first optimizer step's lazy Adam allocation. The deficit is structural - 8.88 GiB of
       static training state on a 9.64 GiB-usable card - not a tunable"
    - "the pivotal value max_num_seqs=1024 (pinned by third_party/.../rollout/rollout.yaml:59)
       is outside the §10.2 allowed-knob list - owner decision required; with max_num_seqs=16
       the vLLM-side OOM is fully cleared"
    - "if the 10 GB card stays the target, a fitting update needs a change outside the
       memory-knob space (e.g. optimizer-state precision, fp32 master weights, a bigger card).
       Identified only; NOT implemented or probed - each would alter the frozen training stack
       and is an owner decision"
    - "S2 (lambda=0 parity with loss/gradient deltas, lambda=1 tensor construction) blocked:
       both conditions need the GPU stack that OOMs"
    - "S3 smoke-generated variant blocked; the stored-data variant is complete"
    - "thresholds/gates, horizon, seeds and holdout N remain owner decisions"

  V5_R001_FORMAL_TRAINING_AUTHORIZED: NO
```

## Notes on scope and integrity

* **Diagnostic discipline.** Every run that includes a knob outside the §10.2 list is labelled a
  diagnostic (`experiments/results/v5_r001_smoke/s1/diagnostic/`), is never appended to
  `attempts.json`, and is reported only as memory-engineering evidence. `attempts.json` holds
  exactly the two §12-compliant attempts. All diagnostics ran at `915622e`, so their parsed
  `reward_summaries`/`r001_stats` fields are empty - the Ray-prefix parser fix (616c0b5) landed
  after them; the raw `R001_REWARD` lines are in every diagnostic log and were read directly.
* **No checkpoints exist.** `trainer.save_freq=-1` in the frozen stack; nothing was written under
  `checkpoints/`, so §19's "engineering checkpoints are deleted" is satisfied by construction.
* **Consumed prompts only.** `data.train_files` and `data.val_files` both point at the consumed
  training parquet; no validation row was iterated (`val_before_train=False`, `test_freq=-1`).
* **Reward-path bug fixed during the audit.** The first live rollout crashed in the reward's row
  guards (`ValueError: truth value of an array ... is ambiguous`) because the pinned batch manager
  hands `non_tensor_batch` fields over as numpy object arrays; fixed in `915622e` with a regression
  fixture. The parser bug that hid worker-emitted markers was fixed in `616c0b5`.
* **Descriptor note.** `R001_ENTRY` prints `"runner": "verl.trainer.main_ppo.TaskRunner"` because
  Ray keeps the parent's creation descriptor; the worker-side `(R001TaskRunner pid=…)` prefix and
  the `R001_STACK` line prove the derived runner executed. Cosmetic, not a blocker.
