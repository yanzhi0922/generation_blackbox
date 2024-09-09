import os
os.environ["GPU_MEMORY_UTILIZATION"] = "1"
import transformers
import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# model_id = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct", cache_dir="D:/tmp/cache/Meta-Llama-3.1-8B-Instruct", local_dir="D:/tmp/cache/Meta-Llama-3.1-8B-Instruct", local_files_only=False)

# 假设你的模型文件存放在 "path/to/your/model/directory/"
#model_id = "/hy-tmp/local"

#tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModel.from_pretrained(model_id)


model = "/hy-tmp/local"
tokenizer1 = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
)
data = load_dataset("abisee/cnn_dailymail", "3.0.0", cache_dir="/hy-tmp/cache")
train_data = data['train'] # 取0.001的数据
train_data = train_data.select(range(0, int(len(train_data) * 0.001)))
print(len(train_data))
print(len(train_data))
for item in train_data:
    inputs = "Write a summary of the text below." + item['article']
    sequences = pipeline(
            inputs,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer1.eos_token_id,
            truncation=True,
            max_new_tokens=512,
            return_full_text=False
    )
    print("output:", sequences[0]['generated_text'])
    print("item['article']:", item['article'])
    print("\n\n\n")
