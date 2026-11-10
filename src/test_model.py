import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "outputs/qwen25-15b-ultrachat"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading original Qwen model...")

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

base_model.eval()

messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant. Answer in Uzbek.",
    },
    {
        "role": "user",
        "content": "RAG nima? O'zbek tilida tushuntir.",
    },
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(base_model.device)


def generate_response(model):
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )


print("\n" + "=" * 50)
print("ORIGINAL QWEN RESPONSE")
print("=" * 50)

print(generate_response(base_model))


print("\nLoading fine-tuned LoRA adapter...")

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model.eval()

print("\n" + "=" * 50)
print("FINE-TUNED QWEN RESPONSE")
print("=" * 50)

print(generate_response(model))