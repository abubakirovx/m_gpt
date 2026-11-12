"""Deterministic paired evaluation using one quantized base model."""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import logging
import os

import torch
from transformers.trainer_utils import set_seed
from runtime import (ROOT, add_model_arguments, adapter_info, attach_adapter, generate,
                     generation_config, load_from_args, project_path, setup_logging, sha256)


def repetition_metrics(text, token_ids):
    def longest(items):
        return max((sum(1 for _ in group) for _, group in itertools.groupby(items)), default=0)
    grams = [tuple(token_ids[i:i + 4]) for i in range(max(0, len(token_ids) - 3))]
    return {"longest_character_run": longest(text), "longest_token_run": longest(token_ids),
            "duplicate_4gram_fraction": 1 - len(set(grams)) / len(grams) if grams else 0.0,
            "exclamation_marks": text.count("!")}


def automatic_check(case, text):
    # Only objective constrained tasks are scored. Technical truth and Uzbek
    # fluency require human review; keywords cannot establish either.
    if "expected_exact" in case:
        return text.strip() == case["expected_exact"]
    if "expected_json" in case:
        try:
            return json.loads(text) == case["expected_json"]
        except (ValueError, TypeError):
            return False
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument("--prompts", default=str(ROOT / "configs/eval_prompts.json"))
    parser.add_argument("--output", help="New JSON report path (existing files are refused)")
    parser.add_argument("--limit", type=int, help="Optional smoke evaluation limit")
    parser.add_argument("--verify-repeat", action="store_true", help="Generate each response twice and require identical token ids")
    parser.add_argument("--verify-base", action="store_true", help="Check first disabled-adapter response against the raw model before adapter attachment")
    args = parser.parse_args()
    setup_logging()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    cases = json.loads(project_path(args.prompts).read_text(encoding="utf-8-sig"))
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Evaluation cases must be nonempty and have unique ids.")
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        cases = cases[:args.limit]
    info = adapter_info(args.adapter, args.base_model)
    output = project_path(args.output) if args.output else ROOT / "reports" / (datetime.now(timezone.utc).strftime("eval-%Y%m%dT%H%M%S%fZ.json"))
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {output}")
    model, tokenizer = load_from_args(args)
    raw_base_tokens = None
    if args.verify_base:
        messages = ([{"role": "system", "content": args.system}] if args.system else [])
        messages += [{"role": "user", "content": cases[0]["prompt"]}]
        config = generation_config(tokenizer, model, args.max_new_tokens)
        raw_base_tokens = generate(model, tokenizer, messages, config, args.max_input_tokens)["token_ids"]
    model = attach_adapter(model, tokenizer, args.adapter, args.base_model, info)
    config = generation_config(tokenizer, model, args.max_new_tokens)
    report = {"status": "running", "base_model": args.base_model, "adapter": info,
              "base_commit": getattr(model.config, "_commit_hash", None),
              "seed": args.seed, "attention": args.attn, "quantization": "4bit NF4 double quant FP16 compute",
              "generation": config.to_dict(), "system": args.system,
              "prompts_sha256": sha256(project_path(args.prompts)),
              "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
              "versions": {name: importlib.metadata.version(name) for name in
                           ["torch", "transformers", "peft", "trl", "bitsandbytes", "datasets"]},
              "gpu": torch.cuda.get_device_name(0), "results": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep completed pairs if generation fails; the failure still propagates.
    with output.open("x", encoding="utf-8") as stream:
        def save():
            stream.seek(0)
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.truncate()
            stream.flush()
        save()
        for case in cases:
            messages = ([{"role": "system", "content": args.system}] if args.system else [])
            messages += [{"role": "user", "content": case["prompt"]}]
            pair = {"case": case, "messages": messages}
            for variant in ["base", "adapter"]:
                set_seed(args.seed)
                context = model.disable_adapter() if variant == "base" else nullcontext()
                with context:
                    result = generate(model, tokenizer, messages, config, args.max_input_tokens)
                    if args.verify_repeat:
                        repeated = generate(model, tokenizer, messages, config, args.max_input_tokens)
                        if repeated["token_ids"] != result["token_ids"]:
                            raise RuntimeError(f"Generation was not repeatable: {case['id']}/{variant}")
                        result["repeat_verified"] = True
                if raw_base_tokens is not None and case is cases[0] and variant == "base":
                    if raw_base_tokens != result["token_ids"]:
                        raise RuntimeError("Disabled adapter output differs from the raw base model")
                    report["raw_base_verified"] = True
                result["repetition"] = repetition_metrics(result["text"], result["token_ids"])
                result["automatic_pass"] = automatic_check(case, result["text"])
                pair[variant] = result
                print(f"\n[{case['id']} / {variant}]\n{result['text']}", flush=True)
            pair["identical_tokens"] = pair["base"]["token_ids"] == pair["adapter"]["token_ids"]
            report["results"].append(pair)
            save()
        report["status"] = "complete"
        save()
    logging.info("Saved paired evaluation: %s", output)


if __name__ == "__main__":
    main()
