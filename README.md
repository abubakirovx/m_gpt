# Local Qwen2.5 QLoRA (Windows 11, RTX 4060 8 GB)

The base remains `Qwen/Qwen2.5-1.5B-Instruct`. Dependencies in `requirements.txt` are unchanged. The original adapter in `outputs/qwen25-15b-ultrachat` is preserved. Inference fixes do not change its learned weights or establish better model quality. See [the investigation](reports/diagnosis.md) and the generated JSON reports for evidence.

## Python on this machine

The existing `.venv` was created with Python 3.11.9 at `C:\Users\oem\AppData\Local\Programs\Python\Python311\python.exe`. That interpreter is missing. The `python` on PATH points to an Anaconda installation that also fails to import its standard library. This is separate from the earlier training/quality problems.

For validation, an official portable Python 3.11.9 was downloaded to `.tools/python311` and configured to read the existing `.venv/Lib/site-packages` plus the project root and `src`. No existing packages were installed, removed, or upgraded. `.tools` is a local, ignored validation aid, not a distributable virtual environment.

Use these **PowerShell** commands from this workspace now:

```powershell
Set-Location 'C:\Users\oem\OneDrive\Desktop\m_gpt'
$python = '.\.tools\python311\python.exe'
$env:PYTHONIOENCODING = 'utf-8'
& $python -c "import sys, torch; print(sys.version); print(torch.__version__, torch.cuda.is_available())"
& $python -m pip check
nvidia-smi
```

For a permanent environment, restore full CPython 3.11.9 at the original installation path, then set `$python = '.\.venv\Scripts\python.exe'` and run the checks above. In VS Code, run **Python: Select Interpreter** and select that working interpreter. Do not delete the existing environment to repair a missing interpreter. On another machine, create a Python 3.11 environment, install a suitable CUDA PyTorch wheel using the [official selector](https://pytorch.org/get-started/locally/), then `python -m pip install -r requirements.txt`. CUDA availability and `pip check` must pass before loading GPU models. Native Windows is supported; WSL2 is optional.

## Inference

```powershell
# Saved original adapter, greedy generation
& $python src/chat.py --local-files-only --prompt "Explain retrieval-augmented generation in English."
& $python src/chat.py --local-files-only --prompt "RAG nima? O'zbek tilida tushuntir."

# Original base model, same inference settings
& $python src/chat.py --local-files-only --base-only --prompt "RAG nima? O'zbek tilida tushuntir."

# Interactive history: /reset starts over; /quit exits
& $python src/chat.py --local-files-only

# Configurable system prompt and optional sampling
& $python src/chat.py --local-files-only --system 'Answer briefly in English. State uncertainty when needed.' --prompt 'What is QLoRA?' --max-new-tokens 128
& $python src/chat.py --local-files-only --prompt 'Write a short story.' --sample --temperature 0.7 --top-p 0.9 --top-k 40 --repetition-penalty 1.05
```

Remove `--local-files-only` to permit Hugging Face downloads. Paths are resolved from the project root. `--adapter` selects a different adapter; `--revision` can pin the base commit. Training defaults to the observed base commit `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.

Both modes use the original Qwen chat template, an explicit attention mask, real padding ID, and both base generation stop IDs including `<|im_end|>`. A fresh generation configuration prevents inherited sampling settings from conflicting with greedy generation. Greedy decoding and repetition penalty 1.0 are defaults; penalties can change responses and should be evaluated. They cannot establish factual accuracy or Uzbek fluency. Long prompts fail with a clear error instead of losing chat boundaries through silent truncation.

## Paired evaluation

```powershell
& $python src/evaluate.py --local-files-only
# Equivalent legacy entry point:
& $python src/test_model.py --local-files-only

# Verify repeatability and adapter disabling on the first prompt:
& $python src/evaluate.py --local-files-only --limit 1 --verify-base --verify-repeat
```

The default 12 prompts cover English technical facts, a nonexistent API, Uzbek explanations/translation, exact instructions, JSON output, and repetition. Reports receive unique names under `reports/`; an explicit `--output` must name a new file. Reports include prompts, completions, token IDs, generation settings, versions, base commit, template/adapter hashes, objective instruction checks, repetition metrics, output-limit flags, and peak PyTorch memory. Technical correctness and Uzbek fluency need human review against the included rubrics; there is no automated factuality/language-quality score.

Evaluation uses **one** 4-bit base with the same saved adapter enabled or disabled. Both variants receive identical messages, tokenization, greedy settings, and token budget. `--verify-base` checks that disabling the adapter matches the raw base before attachment. Exact reproducibility is checked within the same hardware/software environment; different CUDA/library/backend versions may change tokens. `--attn eager` is available for a controlled backend diagnostic; SDPA is the default.

## Audit before considering another training run

No training is performed by these commands:

```powershell
# Cached prepared Arrow files are present on this machine:
$datasetCache = Join-Path $env:USERPROFILE '.cache\huggingface\datasets\HuggingFaceH4___ultrachat_200k\default\0.0.0\8049631c405ae6576f93f445c6b8166f76f5505a'
& $python src/train.py --audit-only --local-files-only --dataset-cache $datasetCache

# On a machine with normal Hub connectivity:
& $python src/train.py --audit-only
```

`--audit-report reports/new-audit.json` saves to a new report file. The explicit Arrow option exists because this machine has prepared Arrow data but `datasets.load_dataset` cannot resolve the Hub builder in offline mode. It logs the selected files. No fallback silently substitutes another dataset.

The audit validates alternating `user`/`assistant` roles (optional first `system`), nonempty content, chat-template prefix boundaries, truncation, supervised tokens, and retained supervised EOS tokens. It refuses silent role remapping. Direct template tokenization prevents extra EOS/BOS additions by a second formatting pipeline. Assistant-only loss masks system/user/assistant-header tokens, preserves assistant content and EOS, and masks padding to -100 through the actual TRL 0.17 collator. `loss_mode: full` reproduces the previous full-conversation objective while retaining corrected formatting. Right truncation can still cut an assistant answer; the audit measures this and never fabricates EOS for an unfinished answer. Rows with no remaining supervised targets are dropped and counted.

## Future training (optional; not run during this investigation)

Only run after reviewing the audit/evaluation and preparing data suited to the task:

```powershell
# Separate, new output for a small training smoke run:
& $python src/train.py --samples 20 --output-dir outputs/qwen25-15b-smoke-new

# Full run uses outputs/qwen25-15b-ultrachat-v2, never the original adapter:
& $python src/train.py
```

Every nonempty training output directory is refused, even if it contains only a prior run manifest. Choose another `--output-dir`; do not delete the original adapter. New runs record tokenizer/config/version/dataset fingerprints and base revision in `run_manifest.json`, use held-out `test_sft` loss, seed the trainer and data order, and retain rank 8 on `q_proj`/`v_proj`. No resume/overwrite option is provided. Evaluation loss is useful for monitoring but does not measure factual correctness or Uzbek quality.

512-token truncation is severe for these multi-turn conversations. Inspect complete-turn or shorter single-turn examples before retraining; increasing sequence length alone increases VRAM and does not supply Uzbek supervision. The new default changes the loss to assistant-only and adds a 3% warmup. Neither change has been trained or quality-validated here. Batch 1, accumulation 8, one epoch, NF4 double quantization, FP16 compute, paged AdamW, and gradient checkpointing remain suitable starting constraints for this GPU.

## Memory and warnings

`--gpu-memory-gb 7` caps the PyTorch allocator below 8 GiB; prompt and generation lengths are bounded and models stay on one GPU. This cap does not include CUDA driver allocations, external library allocations, the Windows desktop, or other processes. It cannot guarantee total GPU usage while other workloads run. `nvidia-smi` and the logged free-memory/allocated/reserved values are the practical checks. Close other GPU workloads yourself if necessary. The program never stops them and never silently switches to CPU. OOM errors propagate with context. Training memory has not been remeasured because no retraining was authorized for this investigation.

Transformers 4.51.3 warns about Sliding Window Attention because its warning tests `config.sliding_window` while actual attention also tests `config.use_sliding_window`. The cached Qwen config sets `sliding_window=32768` and `use_sliding_window=false`. The architecture/config are preserved and the upstream warning remains visible. Switching to eager also triggers this warning; installing FlashAttention or changing architecture is unnecessary for this inactive-window case.

The previous `chat.py` and `test_model.py` already supplied attention masks. Their masks were not a confirmed cause of poor quality. Shared generation now requires a mask, including if padding equals EOS. The actual pad token is used instead of substituting EOS.

Transformers top-level imports are valid at runtime. To address Pylance's lazy-export `reportPrivateImportUsage` reports, imports now reference the modules defining the same public classes (`models.auto.*`, `generation.configuration_utils`, `utils.quantization_config`, `trainer_utils`). No private `_` imports, global warning suppression, stub edits, or dependency changes are used. The Pylance UI itself was not available to validate its diagnostics.

## Verification

```powershell
& $python -m unittest discover -s tests -v
& $python -m compileall -q src tests
& $python -m pip check
```

The tests use the real cached tokenizer and TRL collator, including a tiny CPU Qwen forward pass. They check assistant/EOS/padding labels, multi-turn formatting, truncation, invalid roles, generation settings, prompt limits, adapter validation, output protection, metrics, and import safety. No existing adapter is modified and no training is started. The tokenizer integration tests explicitly skip if the tokenizer is not cached.

For Uzbek, start with native-speaker reviewed Uzbek instruction/response data and a held-out Uzbek test set; balance it with English examples to monitor regressions. UltraChat is an English synthetic conversation dataset and does not supply the requested Uzbek adaptation. Review Latin/Cyrillic consistency and terminology, remove repetitive/corrupt examples, and compare every new adapter with the base. RAG may ground technical content in verified documents but does not itself teach language fluency or guarantee truth. A more capable base is an option only after a separate, memory-budgeted evaluation; this project keeps the current base.

References: [Qwen model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [UltraChat dataset card](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k), [Transformers 4.51.3 templates](https://huggingface.co/docs/transformers/v4.51.3/en/chat_templating), [TRL 0.17 SFT](https://huggingface.co/docs/trl/v0.17.0/en/sft_trainer).
