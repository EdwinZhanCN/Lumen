import torch
from transformers import AutoTokenizer, ChineseCLIPTextModel, ChineseCLIPModel

tokenizer = AutoTokenizer.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16", max_length=77)
clip_model = ChineseCLIPModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16")
model = ChineseCLIPTextModel.from_pretrained("OFA-Sys/chinese-clip-vit-base-patch16", torch_dtype=torch.bfloat16)
model.eval().requires_grad_(False)

inputs = tokenizer("一只猴子", truncation=True, max_length=77, return_length=False, return_overflowing_tokens=False, padding="max_length", return_tensors="pt")

outputs = model(input_ids=inputs["input_ids"], attention_mask=None, output_hidden_states=False)

print(outputs["pooler_output"])

