"""CPU contract checks using the installed pinned libraries and real tokenizer."""
import importlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from trl import SFTConfig, SFTTrainer
from data import encode_conversation, prepare_dataset, validate_messages
from evaluate import automatic_check, repetition_metrics
from runtime import ADAPTER, BASE_MODEL, adapter_info, generation_config, generate
from train import protect_output, validate_config


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
        except OSError:
            raise unittest.SkipTest("Real tokenizer unavailable locally; cache the base tokenizer to run integration checks")
        cls.messages = [
            {"role": "system", "content": "Answer briefly."},
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Say goodbye."},
            {"role": "assistant", "content": "Goodbye!"},
        ]

    def test_role_validation(self):
        for messages in ([], [{"role": "human", "content": "x"}],
                         [{"role": "user", "content": " "}], self.messages[:-1]):
            with self.assertRaises(ValueError):
                validate_messages(messages)

    def test_assistant_labels_and_eos(self):
        feature, stats = encode_conversation(self.tokenizer, self.messages, 512)
        supervised = [t for t, m in zip(feature['input_ids'], feature['completion_mask']) if m]
        decoded = self.tokenizer.decode(supervised, skip_special_tokens=False)
        self.assertIn('Hello!', decoded)
        self.assertIn('Goodbye!', decoded)
        self.assertNotIn('Say hello', decoded)
        self.assertNotIn('Answer briefly', decoded)
        self.assertNotIn('<|im_start|>', decoded)
        self.assertEqual(supervised.count(self.tokenizer.eos_token_id), 2)
        self.assertEqual(stats['retained_supervised_eos'], 2)
        self.assertEqual(feature['input_ids'], self.tokenizer.apply_chat_template(self.messages, tokenize=True, add_generation_prompt=False))

    def test_truncation_and_empty_targets(self):
        feature, stats = encode_conversation(self.tokenizer, self.messages, 3)
        self.assertEqual(len(feature['input_ids']), 3)
        self.assertEqual(stats['truncated'], 1)
        self.assertEqual(stats['no_supervised_tokens'], 1)
        with self.assertRaises(ValueError):
            prepare_dataset([{'messages': self.messages}], self.tokenizer, 3)

    def test_full_loss(self):
        feature, _ = encode_conversation(self.tokenizer, self.messages, 512, 'full')
        self.assertTrue(all(feature['completion_mask']))

    def test_real_trainer_collator_and_forward(self):
        dataset, _ = prepare_dataset([{'messages': self.messages}, {'messages': self.messages[:3]}], self.tokenizer, 512)
        config = Qwen2Config(vocab_size=len(self.tokenizer), hidden_size=16, intermediate_size=32,
                            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                            sliding_window=None, use_cache=False)
        model = Qwen2ForCausalLM(config)
        with tempfile.TemporaryDirectory() as directory:
            args = SFTConfig(output_dir=directory, use_cpu=True, report_to='none', fp16=False,
                             bf16=False, max_length=512, completion_only_loss=True)
            trainer = SFTTrainer(model=model, args=args, train_dataset=dataset, processing_class=self.tokenizer)
            batch = trainer.data_collator([dataset[0], dataset[1]])
            for i, row in enumerate(dataset):
                n = len(row['input_ids'])
                self.assertEqual((batch['labels'][i, :n] != -100).tolist(), [bool(x) for x in row['completion_mask']])
                self.assertTrue((batch['labels'][i, n:] == -100).all())
                self.assertTrue((batch['attention_mask'][i, n:] == 0).all())
            with torch.inference_mode():
                loss = model(**batch).loss
            self.assertTrue(torch.isfinite(loss))

    def test_generation_configuration(self):
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=[151645, 151643]))
        with warnings.catch_warnings(record=True) as caught:
            config = generation_config(self.tokenizer, model)
            config.validate()
        self.assertFalse(caught)
        self.assertFalse(config.do_sample)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_p, 1.0)
        self.assertEqual(config.repetition_penalty, 1.0)
        self.assertEqual(config.pad_token_id, self.tokenizer.pad_token_id)
        self.assertIn(self.tokenizer.eos_token_id, config.eos_token_id)
        sampled = generation_config(self.tokenizer, model, sample=True, temperature=0.6)
        self.assertEqual(sampled.temperature, 0.6)
        with self.assertRaises(ValueError):
            generation_config(self.tokenizer, model, max_new_tokens=0)

    def test_prompt_limit_before_gpu(self):
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=151645))
        config = generation_config(self.tokenizer, model)
        with self.assertRaisesRegex(ValueError, 'Prompt has'):
            generate(model, self.tokenizer, self.messages[:-1], config, max_input_tokens=1)

    def test_missing_and_wrong_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                adapter_info(directory)
        if ADAPTER.exists():
            info = adapter_info(ADAPTER)
            self.assertEqual(info['r'], 8)
            self.assertEqual(info['tensor_count'], 112)
            with self.assertRaises(ValueError):
                adapter_info(ADAPTER, 'wrong/model')

    def test_output_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            protect_output(path)
            (path / 'adapter_config.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                protect_output(path)

    def test_training_config_validation(self):
        import yaml
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/train.yaml').read_text(encoding='utf-8-sig'))
        validate_config(cfg)
        for key, value in [('max_samples', 0), ('max_samples', 1.5), ('batch_size', 2), ('eval_split', cfg['dataset_split'])]:
            with self.assertRaises(ValueError):
                validate_config({**cfg, key: value})

    def test_metrics_and_checks(self):
        self.assertEqual(repetition_metrics('!!!!!', [1] * 8)['longest_character_run'], 5)
        self.assertEqual(repetition_metrics('', [])['duplicate_4gram_fraction'], 0)
        self.assertTrue(automatic_check({'expected_exact': 'READY'}, 'READY\n'))
        self.assertFalse(automatic_check({'expected_json': {'x': 1}}, '```json\n{}\n```'))
        self.assertIsNone(automatic_check({}, 'hello'))

    def test_imports_do_not_load_models(self):
        with patch('runtime.load_base', side_effect=AssertionError('unexpected model loading')):
            importlib.reload(importlib.import_module('chat'))
            importlib.reload(importlib.import_module('test_model'))


if __name__ == '__main__':
    unittest.main()
