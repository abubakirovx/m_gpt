import argparse
from pathlib import Path
import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/train.yaml')
    parser.add_argument('--samples', type=int, default=None, help='Override number of examples for smoke test')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not found. Run inside WSL2 Ubuntu with NVIDIA drivers.')
    print('GPU:', torch.cuda.get_device_name(0))
    set_seed(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_id'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                               bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(cfg['model_id'], quantization_config=quant,
                                                 device_map={'': 0}, torch_dtype=torch.float16)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
                      lora_dropout=cfg['lora_dropout'], bias='none', task_type='CAUSAL_LM',
                      target_modules=['q_proj', 'v_proj'])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    count = args.samples if args.samples is not None else cfg['max_samples']
    raw = load_dataset(cfg['dataset_id'], split=cfg['dataset_split'])
    raw = raw.shuffle(seed=cfg['seed']).select(range(min(count, len(raw))))
    def format_chat(example):
        return {'text': tokenizer.apply_chat_template(example['messages'], tokenize=False,
                                                      add_generation_prompt=False)}
    ds = raw.map(format_chat, remove_columns=raw.column_names)
    training = SFTConfig(
        output_dir=cfg['output_dir'], per_device_train_batch_size=cfg['batch_size'],
        gradient_accumulation_steps=cfg['gradient_accumulation_steps'],
        num_train_epochs=cfg['epochs'], learning_rate=cfg['learning_rate'],
        max_seq_length=cfg['max_seq_length'], dataset_text_field='text',
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        fp16=True, bf16=False, optim='paged_adamw_8bit',
        logging_steps=cfg['logging_steps'], save_strategy='steps',
        save_steps=cfg['save_steps'], save_total_limit=2, report_to='none',
        packing=False,
    )
    trainer = SFTTrainer(model=model, train_dataset=ds, args=training,
                         processing_class=tokenizer)
    trainer.train()
    trainer.save_model(cfg['output_dir'])
    tokenizer.save_pretrained(cfg['output_dir'])
    print('Saved LoRA adapter to', cfg['output_dir'])


if __name__ == '__main__':
    main()
