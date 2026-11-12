"""QLoRA training with a no-training audit mode and protected output directories."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import logging

import torch
import yaml
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.trainer_utils import set_seed
from trl import SFTConfig, SFTTrainer

from data import prepare_dataset
from runtime import ROOT, load_base, project_path, setup_logging


def validate_config(cfg):
    integers = ('max_samples', 'max_seq_length', 'lora_r', 'batch_size',
                'gradient_accumulation_steps', 'logging_steps', 'save_steps', 'eval_samples')
    for key in (*integers, 'lora_alpha', 'epochs', 'learning_rate'):
        if isinstance(cfg.get(key), bool) or not isinstance(cfg.get(key), (int, float)) or cfg[key] <= 0:
            raise ValueError(f'Configuration {key} must be positive')
        if key in integers and not isinstance(cfg[key], int):
            raise ValueError(f'Configuration {key} must be an integer')
    if cfg.get('eval_split') == cfg.get('dataset_split'):
        raise ValueError('eval_split must be separate from dataset_split')
    if not isinstance(cfg.get('target_modules'), list) or not cfg['target_modules']:
        raise ValueError('target_modules must be a nonempty list')
    if cfg['max_seq_length'] > 1024 or cfg['batch_size'] != 1:
        raise ValueError('This 8 GB training profile requires batch_size=1 and max_seq_length<=1024')
    if not 0 <= cfg['lora_dropout'] < 1:
        raise ValueError('lora_dropout must be in [0, 1)')
    if cfg.get('loss_mode') not in ('assistant', 'full'):
        raise ValueError('loss_mode must be assistant or full')


def protect_output(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {path}. Choose a new --output-dir. Existing adapters/checkpoints are preserved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/train.yaml'))
    parser.add_argument('--samples', type=int, help='Override training sample count')
    parser.add_argument('--output-dir', help='Must be empty or new')
    parser.add_argument('--audit-only', action='store_true', help='Tokenize and validate; never load GPU weights or train')
    parser.add_argument('--audit-report', help='New JSON report file')
    parser.add_argument('--local-files-only', action='store_true', help='Use cached Hugging Face assets only')
    parser.add_argument('--dataset-cache', help='Explicit prepared HF dataset cache directory containing ultrachat_200k-<split>*.arrow; useful for offline audits')
    args = parser.parse_args()
    setup_logging()
    cfg = yaml.safe_load(project_path(args.config).read_text(encoding='utf-8-sig'))
    if args.samples is not None:
        cfg['max_samples'] = args.samples
    validate_config(cfg)
    output = project_path(args.output_dir or cfg['output_dir']).resolve()
    if not args.audit_only:
        protect_output(output)
    set_seed(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'], local_files_only=args.local_files_only,
                                             revision=cfg.get('revision'))
    if not tokenizer.chat_template or tokenizer.eos_token_id is None:
        raise ValueError('Base tokenizer must include its chat template and EOS token')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    data_kwargs = {'download_mode': 'reuse_dataset_if_exists'}
    if args.local_files_only:
        from datasets import DownloadConfig
        data_kwargs['download_config'] = DownloadConfig(local_files_only=True)
    def read_split(split):
        if args.dataset_cache:
            directory = project_path(args.dataset_cache)
            files = sorted(directory.glob(f'ultrachat_200k-{split}*.arrow'))
            if not files:
                raise FileNotFoundError(f'No prepared Arrow files for {split} in {directory}')
            logging.info('Reading explicit offline split %s from %s', split, [str(p) for p in files])
            return concatenate_datasets([Dataset.from_file(str(p)) for p in files])
        return load_dataset(cfg['dataset_id'], split=split, **data_kwargs)
    raw = read_split(cfg['dataset_split'])
    # Keep the small permutation in RAM: cached Arrow files may live in a
    # read-only directory, and shared cache locks must not stall local audits.
    raw = raw.shuffle(seed=cfg['seed'], keep_in_memory=True).select(range(min(cfg['max_samples'], len(raw))))
    train_ds, train_audit = prepare_dataset(raw, tokenizer, cfg['max_seq_length'], cfg['loss_mode'])
    heldout = read_split(cfg['eval_split'])
    heldout = heldout.shuffle(seed=cfg['seed'], keep_in_memory=True).select(range(min(cfg['eval_samples'], len(heldout))))
    eval_ds, eval_audit = prepare_dataset(heldout, tokenizer, cfg['max_seq_length'], cfg['loss_mode'])
    audit = {'config': cfg, 'train': train_audit, 'heldout': eval_audit,
             'dataset_cache': args.dataset_cache,
             'train_fingerprint': raw._fingerprint, 'heldout_fingerprint': heldout._fingerprint,
             'template': tokenizer.chat_template, 'eos_token_id': tokenizer.eos_token_id,
             'pad_token_id': tokenizer.pad_token_id}
    if args.audit_report:
        destination = project_path(args.audit_report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8') as stream:
            json.dump(audit, stream, ensure_ascii=False, indent=2)
    if args.audit_only:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return
    model, _ = load_base(cfg['model_id'], args.local_files_only, cfg.get('attn_implementation', 'sdpa'),
                         cfg.get('revision'), cfg.get('gpu_memory_gb', 7.0))
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant': False})
    lora = LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'], lora_dropout=cfg['lora_dropout'],
                      bias='none', task_type='CAUSAL_LM', target_modules=cfg['target_modules'])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    training = SFTConfig(
        output_dir=str(output), per_device_train_batch_size=cfg['batch_size'], per_device_eval_batch_size=1,
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'], num_train_epochs=cfg['epochs'],
        learning_rate=cfg['learning_rate'], warmup_ratio=cfg.get('warmup_ratio', 0.03),
        max_length=cfg['max_seq_length'], completion_only_loss=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        fp16=True, bf16=False, optim='paged_adamw_8bit',
        logging_steps=cfg['logging_steps'], save_strategy='steps', save_steps=cfg['save_steps'],
        save_total_limit=2, report_to='none', packing=False, eval_strategy='epoch',
        prediction_loss_only=True, dataloader_num_workers=0, seed=cfg['seed'], data_seed=cfg['seed'],
    )
    trainer = SFTTrainer(model=model, train_dataset=train_ds, eval_dataset=eval_ds,
                         args=training, processing_class=tokenizer)
    # Check the actual trainer collator, not a parallel implementation of labels.
    batch = trainer.data_collator([train_ds[0]])
    expected = torch.tensor(train_ds[0]['completion_mask'], dtype=torch.bool)
    if not torch.equal(batch['labels'][0] != -100, expected):
        raise RuntimeError('Trainer labels do not match the audited loss mask')
    protect_output(output)
    output.mkdir(parents=True, exist_ok=True)
    audit['base_commit'] = getattr(model.config, '_commit_hash', None)
    audit['versions'] = {name: importlib.metadata.version(name) for name in
                         ['torch', 'transformers', 'trl', 'peft', 'bitsandbytes', 'datasets']}
    audit['created_utc'] = datetime.now(timezone.utc).isoformat()
    (output / 'run_manifest.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    trainer.save_state()
    logging.info('Saved adapter to %s; peak allocated %.2f GiB, reserved %.2f GiB', output,
                 torch.cuda.max_memory_allocated() / 1024**3, torch.cuda.max_memory_reserved() / 1024**3)


if __name__ == '__main__':
    main()
