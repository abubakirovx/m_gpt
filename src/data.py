"""Validated, directly tokenized Qwen conversations with optional assistant loss."""
from collections import Counter
import logging

LOG = logging.getLogger(__name__)


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a nonempty list of role/content objects")
    expected = "user"
    for i, message in enumerate(messages):
        if not isinstance(message, dict) or not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"Message {i}: content must be a nonempty string")
        role = message.get("role")
        if i == 0 and role == "system":
            continue
        if role != expected:
            raise ValueError(f"Message {i}: expected {expected!r}, got {role!r}; roles are not silently remapped")
        expected = "assistant" if role == "user" else "user"
    if messages[-1]["role"] != "assistant":
        raise ValueError("Training conversations must end with an assistant response")


def encode_conversation(tokenizer, messages, max_length, loss_mode="assistant"):
    validate_messages(messages)
    if max_length < 2 or loss_mode not in {"assistant", "full"}:
        raise ValueError("Require max_length>=2 and loss_mode assistant or full")
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    mask = [int(loss_mode == "full")] * len(ids)
    if loss_mode == "assistant":
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            prompt = tokenizer.apply_chat_template(messages[:index], tokenize=True, add_generation_prompt=True)
            if len(prompt) >= max_length:
                break
            completed = tokenizer.apply_chat_template(messages[:index + 1], tokenize=True, add_generation_prompt=False)
            if ids[:len(prompt)] != prompt or ids[:len(completed)] != completed:
                raise ValueError("Chat template is not prefix-stable; cannot safely build assistant labels")
            mask[len(prompt):len(completed)] = [1] * (len(completed) - len(prompt))
            if len(completed) >= max_length:
                break
    # Right truncation retains original token identities. Never fabricate an EOS
    # for a response cut in mid-sentence. Audit how often this happens.
    feature = {"input_ids": ids[:max_length], "attention_mask": [1] * min(len(ids), max_length),
               "completion_mask": mask[:max_length]}
    stats = {"examples": 1, "original_tokens": len(ids), "retained_tokens": len(feature["input_ids"]),
             "truncated": int(len(ids) > max_length),
             "supervised_tokens": sum(feature["completion_mask"]),
             "retained_supervised_eos": sum(t == tokenizer.eos_token_id and m for t, m in zip(feature["input_ids"], feature["completion_mask"])),
             "no_supervised_tokens": int(not any(feature["completion_mask"][1:]))}
    return feature, stats


def prepare_dataset(raw, tokenizer, max_length, loss_mode="assistant"):
    from datasets import Dataset
    totals = Counter()
    rows = []
    for index, example in enumerate(raw):
        if index and index % 500 == 0:
            LOG.info("Audited %d/%d conversations", index, len(raw))
        try:
            feature, stats = encode_conversation(tokenizer, example["messages"], max_length, loss_mode)
        except (ValueError, KeyError) as exc:
            raise ValueError(f"Invalid dataset row {index}: {exc}") from exc
        totals.update(stats)
        if not stats["no_supervised_tokens"]:
            rows.append(feature)
    totals["kept_examples"] = len(rows)
    LOG.info("Tokenization audit (%s): %s", loss_mode, dict(totals))
    if not rows:
        raise ValueError("No examples have supervised targets after truncation. Increase max_seq_length or shorten prompts.")
    if totals["truncated"]:
        LOG.warning("%d/%d conversations were right-truncated at %d tokens; inspect audit before training.", totals["truncated"], totals["examples"], max_length)
    return Dataset.from_list(rows), dict(totals)
