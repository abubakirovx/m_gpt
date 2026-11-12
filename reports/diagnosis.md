# Investigation: local Qwen2.5 QLoRA project

Investigated on 2026-11-11. No retraining was started. The original adapter and checkpoints were preserved. The base model and installed dependencies were not changed. All quality findings below concern this small local evaluation, not general benchmark performance.

## Confirmed findings and likely contributors

1. **Failures occur with the adapter disabled too.** Twelve fixed prompts were evaluated with identical messages, quantization, attention backend, greedy decoding, and a 192-token output budget. Both variants hallucinated technical facts, mishandled Uzbek, and repeated phrases. The checked raw base output exactly matched the adapter-disabled output. The problem is therefore not confined to an incorrectly loaded adapter. This establishes a weakness of the evaluated base/runtime combination; it does not isolate quantization from inherent base-model capability.
2. **UltraChat is a poor match for Uzbek adaptation.** The [dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) identifies it as English synthetic conversation data. The training pipeline supplied no dedicated Uzbek supervision, terminology validation, factuality filter, or held-out quality evaluation. Generic 7,000-example SFT is not evidence that Uzbek or technical reliability should improve. The [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) describes multilingual capabilities but does not establish quality on these Uzbek tasks.
3. **Severe right truncation is confirmed in the actual cached old training data.** Of 7,000 conversations, 6,469 exceeded 512 tokens before truncation (92.41%). The final cached training sequences contain 3,522,307 tokens; the original untruncated cache contains 8,453,899. Many conversations/answers were cut. Under corrected direct tokenization, 707 selected conversations contain no assistant targets in the retained first 512 tokens. The old full-conversation objective therefore trained those examples only on prompts/headers. Right truncation also removes many assistant endings and can plausibly weaken stopping behavior; it has not been proved to cause the particular exclamation-mark incident.
4. **Duplicate EOS formatting is confirmed.** The old code rendered each conversation to a `text` field ending in `<|im_end|>\n`. TRL 0.17 then classified the field as plain text and appended another EOS because the string ended in a newline. The cached text literally ends in `<|im_end|>\n<|im_end|>`. Direct chat-template tokenization fixes this. Qwen's template and EOS themselves were correct. A duplicate EOS is a formatting defect, but its independent effect on quality was not measured.
5. **The old comparison was nondeterministic.** It generated a sampled base answer and a different random sample after attaching the adapter, without resetting RNG state. The two answers were not a controlled quality comparison. The chat and comparison scripts also used different system prompts and generation settings; chat did not explicitly match training's FP16 model dtype. Importing either script loaded models as a side effect, and chat ignored the documented CLI arguments.
6. **Existing adapter identity/loading is correct.** Saved config names `Qwen/Qwen2.5-1.5B-Instruct`, rank 8, alpha 16, dropout 0.05, `q_proj`/`v_proj`, causal LM. All 112 saved tensors are finite. The corrected loader verified that every loaded tensor exactly matched its saved value, that the adapter was active, and that all inference parameters were frozen. Saved/base vocabularies, chat templates, padding IDs, and EOS IDs match. Finite weights and matching configuration do not establish useful learned behavior.
7. **No training numerical collapse was found in the recorded logs.** The checkpoint reached step 875/875 and epoch 1. Logged losses range from 1.2757 to 1.758; logged gradient norms range from 0.1855 to 0.6551. These finite training metrics do not measure held-out factuality or language quality. The original run had no held-out evaluation. Rank 8 and q/v projection targets are legitimate LoRA settings; there is no evidence that changing rank/targets alone fixes the failures.
8. **Warning causes were separated from model quality.** Both original inference scripts already supplied attention masks. A missing attention mask is not a confirmed cause in those files. They did override the pad ID with EOS, and greedy chat inherited the base model's sampling defaults. The new shared generator uses explicit masks, actual padding ID 151643, EOS 151645 plus the base's additional stop ID 151643, and a fresh configuration. Sampling controls are populated only when sampling is enabled.
9. **Sliding Window warning is misleading for this configuration.** Installed Transformers 4.51.3 `models/qwen2/modeling_qwen2.py` warns when `config.sliding_window` is truthy and the backend is not FlashAttention 2. Its actual attention path additionally requires `config.use_sliding_window`. The cached model sets `sliding_window=32768`, `use_sliding_window=false`. The warning remains visible; neither architecture nor window settings were modified. SDPA is used consistently; eager is available for diagnostics. No FlashAttention installation is required for this inactive-window case.
10. **The current Python environment is broken independently of model quality.** `.venv/Scripts/python.exe` points to a missing full Python 3.11 installation. PATH's Anaconda Python fails to import `encodings`. Validation used an isolated official portable CPython 3.11.9 runtime inside `.tools`, reading the existing package directory. No packages were replaced. Long stalls while generating/reusing disk-backed shuffle indices were avoided by keeping the small dataset permutation in RAM; this also avoids writing indices into a read-only/shared HF cache.

## Evaluation evidence

See [all 24 completions and settings](evaluation.json), [repeatability/adapter-disable checks](reproducibility.json), [old cache inspection](cache-audit.json), and [corrected tokenization audit](data-audit.json). JSON files are local ignored artifacts. Run the commands in the README to produce new reports.

| Test | Base | Existing adapter |
|---|---|---|
| RAG explanation | Incorrectly says retrieval-side weights are updated from generated text | Incorrectly says RAG updates model weights by fine-tuning on generated text |
| LoRA / QLoRA | Expands LoRA as Low-Rank Auto-Encoder and gives incorrect explanation | Also expands it as Low-Rank Auto-Encoder and gives incorrect explanation |
| Attention mask / -100 | Incorrectly restricts -100 labels to sequence-to-sequence models | Incorrectly claims -100 is a sentence/context marker and is not used in causal LM training |
| Invented PyTorch API | Invents an implementation/library for the nonexistent API | Also invents an implementation/library |
| Uzbek RAG | Repetitive output, reaches token budget | Repetitive output, reaches token budget |
| Uzbek letter | Responds in English asking for clarification | Responds in English asking for clarification |
| Uzbek translation | English paraphrase, not Uzbek | Same English paraphrase |
| Exact READY | Pass | Pass |
| JSON only | Fails because of Markdown fencing | Pass |
| Exact Salom | Fails: adds `!` | Fails: adds `!` |
| Uzbek water sentence | Repeats phrases | Repeats `suva`; reaches token budget |

Objective instruction checks: base 1/3, adapter 2/3. These three tasks are too few to establish overall improvement. On the Uzbek water prompt, duplicate token 4-gram fractions were 0.673 for base and 0.894 for adapter. On the Uzbek RAG prompt they were 0.741 and 0.677. These are descriptive repetition metrics, not language-quality scores.

Hundreds of exclamation marks were **not reproduced** in the 24 completions. Phrase repetition was reproduced in both variants, including an adapter regression on the water prompt. No hidden repetition stopper or forced no-repeat n-gram rule was added. Output budgets bound runtime, and limit hits are reported.

A second run on the first prompt verified token-identical repeated greedy responses for both variants, and verified raw-base versus disabled-adapter equivalence. Its outputs also match the first run on that prompt. Determinism across different libraries, hardware, or attention backends is not claimed.

Maximum measured PyTorch memory across the 24 completions: **1.1423 GiB allocated, 1.7032 GiB reserved** (rounded upward). This stayed below 8 GB on the RTX 4060. PyTorch figures exclude other processes and driver/external allocations. The allocator is capped at 7 GiB, but cannot cap other applications. Training memory was not remeasured.

An unquantized precision diagnostic was attempted but not run: fewer than 4 GiB GPU memory and only about 3.24 GiB system RAM were free at the respective preflight checks. No external workloads were stopped. Thus NF4/runtime numerical effects remain unisolated; no dependency or precision change is justified by the completed tests alone.

## Corrected dataset audit

All 7,000 selected training conversations and 64 separate `test_sft` conversations passed role/content/template-prefix validation. Selection uses the same seed 42; masks are explicit and are checked through the actual TRL 0.17 collator.

| Measurement | Selected train | Selected held-out |
|---|---:|---:|
| Conversations | 7,000 | 64 |
| Original tokens, without duplicate EOS | 8,446,899 | 72,936 |
| Tokens retained at length 512, before row filtering | 3,521,776 | 32,256 |
| Truncated conversations | 6,468 | 60 |
| No assistant targets after truncation | 707 | 8 |
| Usable conversations | 6,293 | 56 |
| Supervised assistant tokens | 2,170,884 | 19,311 |
| Retained supervised EOS tokens | 5,645 | 57 |

The corrected pipeline drops zero-target rows, masks prompt/header/padding labels, and preserves real assistant EOS targets. It does not append EOS to an unfinished truncated answer. The high truncation rate remains a dataset preparation limitation; review shorter or complete-turn examples before another run. The corrected default may therefore run fewer optimizer steps than the old 875-step run.

## Changes

- `src/runtime.py` (new): shared model/tokenizer loading, strict adapter identity/tensor verification, NF4 FP16 profile, explicit masks/stops, fresh generation configuration, bounded input/output, GPU budget/logging, and clear errors.
- `src/chat.py`: functional CLI, configurable system/sampling settings, base-only option, one-shot and interactive history/reset, no model loading on import.
- `src/evaluate.py` and `configs/eval_prompts.json` (new): deterministic paired evaluation, rubrics, objective checks, repetition metrics, provenance, incremental reports, optional repeated-generation/raw-base verification.
- `src/test_model.py`: compatibility entry point for the evaluation.
- `src/data.py` (new): role validation, direct chat-template tokens, assistant/full loss masks, truncation audit, zero-target filtering.
- `src/train.py`: no-training audit mode, explicit offline Arrow input, RAM shuffle indices, output protection, new-run manifest, separate held-out loss, real collator validation, seeded trainer/data, current supported TRL argument names, and Windows worker settings.
- `configs/train.yaml`: new protected v2 output, observed base commit pinned, explicit targets, assistant loss, held-out split, 3% warmup, and memory/backend configuration. Assistant loss/warmup are prospective training changes, not evaluated quality improvements.
- `tests/test_project.py` (new): 12 meaningful CPU checks with installed libraries, real tokenizer, and a tiny Qwen forward pass.
- `README.md`: exact PowerShell commands, interpreter recovery, evaluation/audit/training instructions, memory limits, warning diagnosis, and Uzbek recommendations.
- `.gitignore`: ignores local portable tooling and generated JSON/log reports.
- `reports/diagnosis.md` (this file): investigation record. `requirements.txt` is unchanged.

Public Transformers classes are imported from their defining modules to avoid lazy top-level export ambiguity in Pylance. No private underscored imports, package stub modifications, or global diagnostic suppression were introduced. Runtime imports pass. Pylance itself was not available for an IDE diagnostic check.

## Validation and preservation

- `python -m unittest discover -s tests -v`: **12 passed**; covers real tokenizer/template/EOS, multi-turn assistant masks, actual SFTTrainer collation and finite CPU forward loss, padding -100, truncation, bad roles/config, generation settings, limits, missing/wrong adapter, output protection, metrics, import safety.
- `python -m compileall -q src tests`: passed.
- `python -m pip check`: no broken requirements.
- Twelve-prompt GPU evaluation: complete, 24 completions saved.
- GPU reproducibility/raw-base check: complete, all token comparisons passed.
- Corrected adapter chat CLI smoke: returned exactly `READY`, 42 input tokens, 2 generated tokens; all 112 adapter tensors matched and no generation-parameter/mask warning occurred.
- Full 7,000 + 64 data audit: completed without loading GPU model weights or training.
- Training CLI directed at the original output: correctly refused it before model loading, exit 1.
- `git diff --check`: passed; Git only reported its usual LF/CRLF conversion notices.
- Original adapter SHA-256 before and after investigation: `10718047be84faaf4c8bb6cef3ffa3ebc7d9cb61fe38265c7d3fe504edf4ef38` (unchanged). No existing output artifact was deleted or overwritten.

## Retraining decision and remaining limitations

Retraining is unnecessary to use the corrected inference/evaluation tools. The existing adapter remains usable and inspectable, but inference fixes cannot undo its learned behavior. Corrected formatting and loss masks affect only future adapters. A new training experiment is reasonable only after addressing the dataset mismatch and truncation, with native-speaker-reviewed Uzbek instruction data, consistent script/terminology, a held-out Uzbek set, and English regression checks. Repeating the same English UltraChat run is not a supported Uzbek remedy.

The small evaluation is diagnostic, not a benchmark. Prompt wording/system prompts can change outputs; no sweep was run. No held-out loss for the old adapter, dense precision comparison, training backward pass, new training run, or native-speaker Uzbek assessment was completed. Full loss is a legitimate objective, not inherently a bug; assistant-only loss is chosen for future instruction tuning and must be evaluated. RAG can provide verified context for technical tasks but cannot guarantee factuality or teach Uzbek fluency. A larger or more Uzbek-capable base can be considered only after a separate memory/quality comparison; the current base was retained.

Use the exact commands in [README.md](../README.md) to run corrected chat, compare both variants, audit data, and test. Future training commands use new directories and are explicitly separated from the non-training workflow.
