import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "outputs/qwen25-15b-ultrachat"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization_config,
    device_map="auto",
    attn_implementation="eager",
)

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
)

model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful AI assistant. "
            "Always respond in Uzbek when the user requests Uzbek. "
            "Be accurate and avoid inventing information."
        ),
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
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)

print(response)