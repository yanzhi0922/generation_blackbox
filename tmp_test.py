import transformers
import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModel
model_id = snapshot_download("LLM-Research/Meta-Llama-3.1-8B-Instruct", cache_dir="/hy-tmp/cache", local_dir="/hy-tmp/local", local_files_only=True)

# 假设你的模型文件存放在 "path/to/your/model/directory/"
#model_id = "/hy-tmp/local"

#tokenizer = AutoTokenizer.from_pretrained(model_id)
#model = AutoModel.from_pretrained(model_id)


pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": "people A so if are for the she at the of the of the I'm by Mr be A they But out time just do people into his He my are time on the me have from the their not wouldWrite a summary of the text below.Spain's Carla Suarez Navarro came back from a dreadful first set to defeat three-times Miami Open champion Venus Williams 0-6, 6-1, 7-5 at Key Biscayne on Tuesday and set up a semi-final clash with Germany's ninth seed Andrea Petkovic. Petkovic booked her place in the last four after a 6-4, 6-2 win over 14th seeded Czech Karolina Pliskova. Williams utterly dominated the first set, crushing the Spaniard 6-0 in less than half an hour but her opponent was able to regroup and battle back in impressive style. Venus Williams shows her frustrations as she loses to Carla Suarez Navarro in the Miami Open last eight Suarez Navarro celebrates as she completes her comeback against three-time Miami champion Williams Suddenly regaining her confidence Suarez Navarro won the first four games and went on to take the second set 6-1. The third set bucked the trend for one-sided exchanges until Suarez Navarro broke to go 5-4 up only for Venus to break back straight away. The Spaniard won the last two games however to continue an impressive season that has seen her reach two other semi-finals and make the last eight at Indian Wells for the first time. 'It was a crazy match, crazy first two sets. Venus was unbelievable at the beginning. I started a bit nervous but even if you lose the first set you are still in the competition and I am happy with the way I came back,' said Suarez Navarro. Williams chases to her right to play a forehand during her quarter-final on Tuesday evening Williams (left) shakes hands across the net with Suarez Navarro as the Spaniard goes through Williams said she had struggled to find consistency. 'I just made a little too many errors and I was going for it the whole match. Towards the end just never found the happy medium between being aggressive and putting the ball in the court,' she said. Petkovic made a blistering start, going up 4-1 in the first set but the 23-year-old Pliskova powered her way back to 4-4 and then had three break points, all of which were saved. Petkovic, the 2011 Miami semi-finalist, won the next five games and wrapped up the match in one hour, 16 minutes. Andrea Petkovic has her eye on the ball as she plays a forehand during her quarter-final victory Czech Karolina Pliskova hits a serve but later..."}
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
outputs = pipeline(
    messages,
    max_new_tokens=256,
)
outputs = pipeline(
    messages,
    max_new_tokens=256,
)
outputs = pipeline(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"][-1]["content"])