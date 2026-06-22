# LOOP-STATE — QUEST-30B-RL Turbo

Goal (session-scoped): **optimize QUEST per the TURBO systems and push the Quest variant to its
optimum on the B70.** Working autonomously toward it. Methodology mirrors the prior campaign.

## Pipeline status (2026-06-22)
- [✅] **Phase 0** baseline — DONE. B70 target band (MoE-reorder tree, Q5_K, -ngl 99): **pp512 1154.6 t/s,
  tg128 79.9 t/s** (main tree: 993 / 74.8). This is the speed ceiling QUEST must reach. (`results/00_baseline.txt`)
- [✅] **Phase 1** fetch → BF16 — DONE & VERIFIED. `artifacts/quest-30b-rl-bf16.gguf` (57 GB, 579 tensors).
  Metadata confirmed: qwen3moe, 48L, ctx 262144, 128E/8A, 32/4 GQA, ffn 6144/expert 768, rope 1e7.
  MoE tensors named `blk.N.ffn_{down,gate,up}_exps.weight` (quantizer/imatrix-recognized).
- [✅] **Phase 2** imatrix — DONE. `artifacts/quest-30b-rl.imatrix` (117 MB), calib PPL 7.44.
  Coverage: NO dead experts; 42/144 expert tensors (14/48 layers) have 1–6 uncovered long-tail
  experts (95.3–99.2%). Strong → proceed. CAVEAT: generic corpus may leave tool-call/`<think>`
  experts uncovered → agentic gate (04b) is decisive, not PPL. Details: `results/02_coverage_summary.txt`.
- [✅→Q5] **Phase 3** quant sweep — REFOCUSED: user directive **ship Q5_K_M exclusively**. Q5_K_M built
  (20.2 GB, fits B70 -ngl 99 fully). Sweep killed after Q5/Q6 to free GPU; Q4 kept as harmless leftover,
  Q6/Q8 removed. BF16 = gate reference.
- [✅] **Phase 4** PPL sanity — Q5_K_M wikitext PPL **7.19** (BF16 calib anchor 7.44) → no gross breakage.
  (Ran direct w/ 2>&1; BF16 wikitext PPL skipped to save GPU for the gate.)
- [✅ SHIP] **Phase 4b** AGENTIC GATE — **Q5_K_M PASSES**, behaviorally identical to BF16:
  TOOL 8/9 (agree 8/9), REASON 10/10 (agree 10/10), THINK 10/10 — same scores as BF16, no H1/H2/H3.
  `abstain` fails on BOTH (base-model tool-eagerness, not quant damage). 95–99% imatrix coverage caused
  NO behavioral cliff. Summary: `results/04b_summary.txt`. **Q5_K_M certified shippable.**
  Subagent fleet (10 agents) found+verified 3 script bugs (PPL-parse HIGH, gate-zombie MED, ub-sweep MED) —
  all FIXED. Probes hardened: TOOL 5→9 + `<think>` integrity. Plan: `results/_prep/Q5_ACTION_PLAN.md`.
- [staged] **Phase 3-orig note** (superseded): full multi-quant cliff sweep dropped per Q5-exclusive directive.
- [staged] **Phase 4** accuracy — `scripts/04_accuracy.sh` (PPL + parity; necessary-not-sufficient).
- [staged] **Phase 4b** AGENTIC GATE — `scripts/04b_agentic_gate.sh` + `04b_probes.py` (tool-call integrity +
  reasoning, graded vs BF16) → **the ship decision**.
- [✅] **Phase 5** speed-tune — Q5_K_M on B70 (-ngl 99): ship (FA on, KV q8_0, -ub 4096) pp512 ~1380 /
  tg128 ~90; peak tg128 109 (KV f16); peak pp512 1597 (FA off); -ub lever pp4096 1130→1703→1804.
  MoE-reorder build +7.9% decode → ship binary. Depth falloff: tg d0 90 / d16k 17.7 / d32k 9.9.
  Logs `results/05_*.txt`. `scripts/05_focused.sh`.
- [✅] **Phase 6** serve — smoke ALL PASS (S1 tool-route, S2 tool round-trip→391, S3 think→120) at
  CTX 65536/KV q8_0, no OOM. `scripts/06_serve.sh` (:8093) + `scripts/06_smoke.sh`.

## RELEASE (2026-06-22) — user directive: new repo + HF card, max quality, Q5-exclusive
- HF (PRIVATE): **Frosty40/QUEST-30B-RL-B70-Turbo-GGUF** — card UPLOADED; Q5_K_M GGUF + imatrix UPLOADING (bg `b16i300kx`).
  User flips PUBLIC when satisfied. Card: `release/README.md` (gate+speed tables filled, honest depth curve).
- GitHub (PRIVATE): **github.com/newjordan/quest-turbo** — PUSHED (pipeline + standardized `tests/` suite). Flip public alongside HF.
- Standardized pytest cert suite: ✅ DONE (`tests/`). 92 pass / 35 skip / 127 collected; `certify.sh` -> PASS/FAIL.
  Self-certifies **SHIP** on the real gate JSON (`results/04b_answers/*.json`) — agrees with the manual verdict.
  Built by 5-agent fleet (contract-first). Two fleets total this session: 3 verified script bugs fixed + gate hardened.
- ✅ COMPLETE & PUBLIC (verified against repo file list): HF repo has hero.jpg, README (card), imatrix,
  and quest-30b-rl-Q5_K_M.gguf (21.73 GB). private=False. GitHub public-pending (still private; flip when ready).
  NOTE: xet upload stalled twice on the 21 GB file over wifi → succeeded via HF_HUB_DISABLE_XET=1 (plain LFS).
  Speed claims corrected to MEASURED real numbers (decode ~80 short / ~31 @8K, prefill ~1380); 109 was bogus.

## Decisions / facts locked
- QUEST = qwen3_moe, 48L/128E-8A, 262144 ctx, `<think>`+tool-calling, RL deep-research (see AUDIT.md).
- Main llama.cpp converter supports Qwen3MoeForCausalLM → no MoE-tree fallback needed for convert.
- Both llama.cpp trees built; RUN_BIN = the MoE-reorder speed tree.
- the prior campaign's gated-delta-net decode fusion does NOT apply (QUEST = standard GQA). Router/expert-combine
  fusions + `-ub` prefill are expected to carry the B70 speed win.

## Guardrails (carried from the prior campaign autonomy rules)
- Do NOT push to public HF, touch other projects' shipped GGUFs, or cycle a live server w/o user OK.
- Gate before ship: no quant ships without passing 04b within ~1-2 of BF16 on TOOL + REASON.

## Log
- 2026-06-22: identified osunlp/QUEST-30B-RL; scaffolded repo; wrote+syntax-checked phases 00–06 +
  agentic gate; launched Phase 1 fetch. Next: when BF16 lands → run 02→03→04→04b → pick ship quant → 05 → 06.
- 2026-06-22 (recovery): connection dropped mid-fetch; fetch process had died leaving 7/13 shards +
  ~16 GB of partial shards in the hf xet cache. Phase 0 had finished cleanly (baseline captured).
  Verified hf/converter/torch deps, relaunched resumable `01_fetch_source.sh` in background → confirmed
  ~30 MB/s and all 13 shards being reconciled. No data loss; pipeline unblocked, awaiting BF16.
