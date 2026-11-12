"""One-shot or interactive inference; importing never loads a model."""
import argparse
import logging

import torch
from transformers.trainer_utils import set_seed
from runtime import (add_model_arguments, adapter_info, attach_adapter, generate,
                     generation_config, load_from_args, setup_logging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    parser.add_argument('--base-only', action='store_true')
    parser.add_argument('--prompt', help='Omit for interactive mode; /reset clears history, /quit exits')
    parser.add_argument('--sample', action='store_true', help='Enable sampling (default: greedy)')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--top-k', type=int, default=40)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    args = parser.parse_args()
    setup_logging()
    set_seed(args.seed)
    info = None if args.base_only else adapter_info(args.adapter, args.base_model)
    model, tokenizer = load_from_args(args)
    if info:
        model = attach_adapter(model, tokenizer, args.adapter, args.base_model, info)
    config = generation_config(tokenizer, model, args.max_new_tokens, args.sample,
                               args.temperature, args.top_p, args.top_k, args.repetition_penalty)
    messages = [{'role': 'system', 'content': args.system}] if args.system else []
    while True:
        try:
            prompt = args.prompt if args.prompt is not None else input('You: ')
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip() == '/quit':
            break
        if prompt.strip() == '/reset':
            messages = [{'role': 'system', 'content': args.system}] if args.system else []
            if args.prompt is not None:
                break
            continue
        if not prompt.strip():
            if args.prompt is not None:
                parser.error('--prompt must not be empty')
            continue
        request = messages + [{'role': 'user', 'content': prompt}]
        result = generate(model, tokenizer, request, config, args.max_input_tokens)
        print(result['text'], flush=True)
        logging.info('input=%d output=%d limit_reached=%s peak_allocated=%.2f GiB peak_reserved=%.2f GiB',
                     result['input_tokens'], result['generated_tokens'], result['hit_token_limit'],
                     result['peak_allocated_gib'], result['peak_reserved_gib'])
        messages = request + [{'role': 'assistant', 'content': result['text']}]
        if args.prompt is not None:
            break


if __name__ == '__main__':
    try:
        main()
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError('CUDA out of memory. Close other GPU workloads and reduce input/output tokens.') from exc
