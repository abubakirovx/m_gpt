# Qwen2.5-1.5B-Instruct QLoRA — RTX 4060 8 GB

Ready-to-run VS Code / WSL2 Ubuntu starter project. Dataset: `HuggingFaceH4/ultrachat_200k`, `train_sft` split, default 7000 samples. Uses 4-bit NF4 + LoRA, not full model fine-tuning.

## Requirements
- Windows 11 + WSL2 Ubuntu + VS Code Remote WSL
- NVIDIA driver with WSL CUDA support (`nvidia-smi` inside Ubuntu)
- Python 3.11 recommended, internet access for model and dataset downloads
- Enough free disk space for model downloads, datasets, and checkpoints (recommend 15+ GB)

## Install (WSL Ubuntu terminal)
```bash
cd m_gpt
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install CUDA PyTorch first; check https://pytorch.org/get-started/locally/ for current wheel selection.
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
If `python3.11` is missing, install Python 3.11 or use a supported Python 3.12 environment. Don't use Colab's Python 3.13 for this pinned setup.

## Train
First run a 20-sample smoke test:
```bash
python src/train.py --samples 20
```
Then run the full dataset:
```bash
python src/train.py
```
Both runs use the same output directory: smoke-test artifacts may be overwritten by full training. For clean runs, delete or rename `outputs/qwen25-15b-ultrachat` between runs. Training checkpoints are LoRA adapter checkpoints.

## Chat with trained adapter
```bash
python src/chat.py --prompt "Explain RAG in Uzbek"
```

## Customize
Edit `configs/train.yaml`: lower `max_seq_length` to 256 if GPU runs out of memory; lower `max_samples` to test faster. Default LoRA `r=8`, `q_proj`/`v_proj`, batch size 1, gradient accumulation 8, 1 epoch. Close GPU-heavy apps while training.

## Notes
- `train_sft` contains conversations; `apply_chat_template` formats them before training.
- Loss is computed on all non-padding tokens of the formatted conversations; this is not assistant-only masking.
- The output directory contains **LoRA adapters**, not standalone base model weights. `chat.py` downloads the base model as needed.
- Pinning libraries avoids the previously encountered TRL 1.17.1 / Transformers 5.22.1 fused-forward path. Installation and hardware-dependent compatibility still need verification on your PC.
- Hugging Face dataset/model downloads require internet. Review dataset licensing and terms before commercial use.
