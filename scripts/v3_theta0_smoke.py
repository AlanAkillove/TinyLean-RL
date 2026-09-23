#!/usr/bin/env python3
"""NON-FORMAL fly90/RTX 3090 theta0 hidden-state feasibility smoke. See note below."""

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# NON-FORMAL feasibility smoke.
# This was run on fly90 / RTX 3090 24GB. It establishes architecture feasibility
# ONLY and its timing/VRAM are NOT a formal V3-D001 compute result. The formal
# theta0 representation extraction runs on fly122 / RTX 3080 10GB (owner directive
# 2026-09-23 §0/§8), where a fresh 1-3 prompt smoke is re-measured. Primary
# representation for V3-D001 is block 18 last-token only (9/27 = robustness).
mp = 'models/weights/kimina_distill_0_6b'
prompts = []
for p in ['runs/m1_seed3/rollout_data/3.jsonl', 'runs/m1_seed2/rollout_data/1.jsonl']:
    for line in Path(p).read_text().split('\n'):
        if line.strip():
            prompts.append(json.loads(line)['input'])
        if len(prompts) >= 3:
            break
    if len(prompts) >= 3:
        break

tok = AutoTokenizer.from_pretrained(mp)
encs = [tok(pr, add_special_tokens=False) for pr in prompts]
lens = [len(e['input_ids']) for e in encs]

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
if dev == 'cuda':
    torch.cuda.reset_peak_memory_stats()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(mp, torch_dtype=torch.bfloat16, device_map=dev).eval()
load_gb = torch.cuda.max_memory_allocated() / 1e9 if dev == 'cuda' else 0
NL = model.config.num_hidden_layers
print(f'arch: Qwen3 num_layers={NL} hidden={model.config.hidden_size} '
      f'| load_time_s={time.time() - t0:.1f} peak_load_GB={load_gb:.2f}')

layers_idx = [NL // 3, (2 * NL) // 3, NL - 1]  # early/mid, ~2/3, final (0-based block idx)
peak = 0
t1 = time.time()
for e in encs:
    x = torch.tensor([e['input_ids']], device=dev)
    with torch.no_grad():
        out = model(input_ids=x, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states  # len NL+1, index 0 = embeddings
    vecs = [hs[l + 1][0, -1, :].float() for l in layers_idx]  # last non-pad token pooling
    if dev == 'cuda':
        peak = max(peak, torch.cuda.max_memory_allocated())
print('prompt token lens:', lens)
print('pre-registered block layers (0-based):', layers_idx, '-> pooled dim:', [v.shape[0] for v in vecs])
print(f'3-prompt forward time_s: {time.time() - t1:.2f} | single_forward_peak_GB: {peak / 1e9:.2f}')
print(f'hidden vectors: {len(layers_idx)} layers x {vecs[0].shape[0]} dims, dtype fp32 for heads')
