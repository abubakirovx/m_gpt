"""Shared Qwen loading and bounded generation. No model loads at import time."""
import hashlib
import logging
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file
from transformers.generation.configuration_utils import GenerationConfig
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.utils.quantization_config import BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = ROOT / "outputs/qwen25-15b-ultrachat"
SYSTEM = "You are a helpful assistant. Follow the requested language. Be accurate; state uncertainty when needed."
LOG = logging.getLogger(__name__)


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def adapter_info(path, base_model=BASE_MODEL):
    path = project_path(path).resolve()
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json in {path}. Use --base-only for the original model.")
    weights = path / "adapter_model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing adapter weights: {weights}")
    cfg = PeftConfig.from_pretrained(str(path))
    if cfg.base_model_name_or_path != base_model:
        raise ValueError(f"Adapter expects {cfg.base_model_name_or_path!r}, requested {base_model!r}.")
    if cfg.peft_type != "LORA" or cfg.task_type != "CAUSAL_LM":
        raise ValueError("Expected a causal language model LoRA adapter.")
    tensors = load_file(str(weights), device="cpu")
    if not tensors or any(not torch.isfinite(t).all().item() for t in tensors.values()):
        raise ValueError("Adapter contains missing or non-finite tensors.")
    info = {"path": str(path), "sha256": sha256(weights), "r": cfg.r,
            "targets": sorted(cfg.target_modules), "tensor_count": len(tensors)}
    LOG.info("Adapter: %s", info)
    return info


def cuda_setup(memory_gb=7.0):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check the selected Python interpreter, CUDA-enabled PyTorch, and NVIDIA driver. Native Windows is supported; WSL is optional.")
    if not 0 < memory_gb < 8:
        raise ValueError("GPU memory budget must be greater than zero and below 8 GiB.")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(memory_gb * 1024**3 / total, 0.95), 0)
    free, _ = torch.cuda.mem_get_info(0)
    LOG.info("GPU %s; free %.2f GiB; allocator cap %.2f GiB", torch.cuda.get_device_name(0), free / 1024**3, memory_gb)
    if free < 2 * 1024**3:
        raise RuntimeError("Less than 2 GiB VRAM free. Close other GPU workloads and retry. No processes were stopped.")


def quantization():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)


def load_base(base_model=BASE_MODEL, local_files_only=False, attn="sdpa", revision=None, memory_gb=7.0):
    cuda_setup(memory_gb)
    kwargs = {"local_files_only": local_files_only, "revision": revision}
    tokenizer = AutoTokenizer.from_pretrained(base_model, **kwargs)
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer requires its original chat template and EOS token.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    cfg = AutoConfig.from_pretrained(base_model, **kwargs)
    LOG.info("Attention=%s use_sliding_window=%s sliding_window=%s revision=%s", attn,
             getattr(cfg, "use_sliding_window", None), getattr(cfg, "sliding_window", None), getattr(cfg, "_commit_hash", None))
    # Preserve upstream architecture. 4.51.3 warns even when use_sliding_window=False.
    model = AutoModelForCausalLM.from_pretrained(
        base_model, config=cfg, quantization_config=quantization(), device_map={"": 0},
        torch_dtype=torch.float16, attn_implementation=attn, **kwargs)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def attach_adapter(model, tokenizer, path, base_model=BASE_MODEL, info=None):
    info = info or adapter_info(path, base_model)
    path = info["path"]
    if (Path(path) / "tokenizer_config.json").is_file():
        saved = AutoTokenizer.from_pretrained(path, local_files_only=True)
        if (saved.get_vocab() != tokenizer.get_vocab() or saved.chat_template != tokenizer.chat_template
                or saved.eos_token_id != tokenizer.eos_token_id or saved.pad_token_id != tokenizer.pad_token_id):
            raise ValueError("Saved tokenizer differs from base vocabulary/template/special tokens. Investigate before inference.")
    model = PeftModel.from_pretrained(model, path, is_trainable=False)
    model.eval()
    if model.active_adapter != "default" or not model.peft_config:
        raise RuntimeError("LoRA adapter did not activate.")
    saved_weights = load_file(str(Path(path) / "adapter_model.safetensors"), device="cpu")
    # Compare the adapter itself; do not ask PEFT to probe the Hub for optional
    # embedding exports during an offline inference validation.
    loaded_weights = get_peft_model_state_dict(model, adapter_name="default", save_embedding_layers=False)
    if saved_weights.keys() != loaded_weights.keys():
        raise RuntimeError("Loaded adapter tensor keys differ from the saved adapter; refusing partial loading.")
    for key, tensor in loaded_weights.items():
        loaded = tensor.detach().cpu()
        if not torch.equal(loaded, saved_weights[key].to(dtype=loaded.dtype)):
            raise RuntimeError(f"Loaded adapter tensor does not match saved weights: {key}")
    LOG.info("Verified all %d loaded adapter tensors against saved weights", len(loaded_weights))
    LOG.info("Active adapter=%s; trainable parameters=%d", model.active_adapter,
             sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def generation_config(tokenizer, model, max_new_tokens=192, sample=False,
                      temperature=0.7, top_p=0.9, top_k=40, repetition_penalty=1.0):
    if not 1 <= max_new_tokens <= 1024:
        raise ValueError("max_new_tokens must be between 1 and 1024.")
    if temperature <= 0 or not 0 < top_p <= 1 or top_k < 0 or repetition_penalty <= 0:
        raise ValueError("Require temperature>0, 0<top_p<=1, top_k>=0, repetition_penalty>0.")
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    eos = list(dict.fromkeys([tokenizer.eos_token_id, *eos]))
    # Fresh config avoids inheriting incompatible Qwen sampling defaults in greedy mode.
    options = dict(max_new_tokens=max_new_tokens, do_sample=sample, num_beams=1,
                   pad_token_id=tokenizer.pad_token_id, eos_token_id=eos,
                   bos_token_id=tokenizer.bos_token_id, repetition_penalty=repetition_penalty,
                   use_cache=True)
    if sample:
        options.update(temperature=temperature, top_p=top_p, top_k=top_k)
    return GenerationConfig(**options)


def generate(model, tokenizer, messages, config, max_input_tokens=2048):
    if not 1 <= max_input_tokens <= 3072:
        raise ValueError("max_input_tokens must be between 1 and 3072 for the 8 GB profile.")
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt")
    if "attention_mask" not in inputs:
        raise RuntimeError("Tokenizer failed to return an explicit attention_mask.")
    length = inputs["input_ids"].shape[-1]
    if length > max_input_tokens:
        raise ValueError(f"Prompt has {length} tokens, limit {max_input_tokens}. Shorten prompt/history; it was not silently truncated.")
    if length + config.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus completion exceeds model context length.")
    inputs = inputs.to(model.device)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=config)
    tokens = output[0, length:].tolist()
    return {"text": tokenizer.decode(tokens, skip_special_tokens=True), "token_ids": tokens,
            "input_tokens": length, "generated_tokens": len(tokens),
            "hit_token_limit": len(tokens) == config.max_new_tokens and tokens[-1] not in config.eos_token_id,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}


def add_model_arguments(parser):
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--adapter", default=str(ADAPTER))
    parser.add_argument("--revision", help="Optional base model commit SHA")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn", choices=["sdpa", "eager"], default="sdpa")
    parser.add_argument("--gpu-memory-gb", type=float, default=7.0)
    parser.add_argument("--system", default=SYSTEM)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def load_from_args(args):
    return load_base(args.base_model, args.local_files_only, args.attn, args.revision, args.gpu_memory_gb)
