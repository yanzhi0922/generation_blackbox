from openai import OpenAI
import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'      # AutoDL
#os.environ['HF_HOME'] = 'D:/tmp/cache'     # win11
os.environ["HF_HOME"] = "/mnt/d/tmp/cache"  # wsl2
import time
from tigerscore import TIGERScorer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from torch.optim import AdamW
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

def solve_v_total_exact(prompt_emb):
    k = 1
    a, b = -3, 0

    b = prompt_emb.max()

    def f(v):
        s = (prompt_emb - v).clamp(0, 1).sum()
        return s - k

    itr = 0

    v = 0
    while (1):
        itr += 1
        v = (a + b) / 2
        obj = f(v)
        if abs(obj) < 1e-3 or itr > 20:
            break
        if obj < 0:
            b = v
        else:
            a = v
    return v, itr


def constrainScoreByWholeExact(prompt_embeds):
    for i in range(len(prompt_embeds)):
        v, itr = solve_v_total_exact(prompt_embeds[i])
        prompt_embeds[i].sub_(v).clamp_(1e-7, 1)


def query(input):
    message = [
        {"role": "system", "content": ""},
        {"role": "user", "content": input},
    ]
    hypo_output = None
    received = False
    while (not received) or (hypo_output is None):
        try:
            hypo_output = client.chat.completions.create(
                model=chatbot,
                messages=message,
                stream=False,
                max_tokens=1024
            )
            received = True
        except:
            time.sleep(1)
    return hypo_output.choices[0].message.content

def query_concurrently(prompts, instruction, input_context):
    with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [executor.submit(query, p + instruction + input_context) for p in prompts]
        results = [future.result() for future in futures]
    return results

def query_concurrently_sample_size(sample_size, instruction, input_context):
    with ThreadPoolExecutor(max_workers=sample_size) as executor:
        futures = [executor.submit(query, instruction + input_context) for _ in range(sample_size)]
        results = [future.result() for future in futures]
    return results

def tigerloss(instruction, input_context, output):
    received = False
    cnt = 0
    while not received and cnt < 10:
        try:
            result = scorer.score([instruction], [output], [input_context])
            received = True
        except:
            time.sleep(1)
            cnt += 1
    score = result[0].get('score', 0)  # 使用 get 方法提供默认值 0
    loss = - score if score is not None else 0
    return loss


def pmi():
    result = []
    with open(pmi_data, 'r', encoding='utf-8') as f:
        for line in f:
            result.append(line.strip('\n'))
    unique = []
    [unique.append(i) for i in result if not i in unique]
    ngram_index_list = list(unique)
    return ngram_index_list


def test(sample_size=5):
    test_result = []
    origin_result = []
    for item in test_data:
        prompts_dist = torch.distributions.Categorical(prompts_probs)
        loss_list = []
        origin_loss_list = []
        prompts = []
        for k in range(sample_size):
            prompts_discrete_indices = prompts_dist.sample()
            prompts_discrete_ngram_list = []
            indices_list = prompts_discrete_indices.int().tolist()
            for idx in indices_list:
                prompts_discrete_ngram_list.append(ngram_list[idx])
            prompts_discrete = ' '.join(prompts_discrete_ngram_list)
            prompts.append(prompts_discrete)
        outputs = query_concurrently(prompts, item['instruction'], item['input_context'])
        outputs_origin = query_concurrently_sample_size(sample_size, item['instruction'], item['input_context'])
        for output, output_origin in zip(outputs, outputs_origin):
            loss = tigerloss(item['instruction'], item['input_context'], output)
            origin_loss = tigerloss(item['instruction'], item['input_context'], output_origin)
            loss_list.append(loss)
            origin_loss_list.append(origin_loss)

        loss_avg = sum(loss_list) / sample_size
        loss_avg /= len(test_data)
        origin_loss_avg = sum(origin_loss_list) / sample_size
        origin_loss_avg /= len(test_data)
        test_result.append(loss_avg)
        origin_result.append(origin_loss_avg)
        print("test_loss_avg:", loss_avg)
        print("origin_loss_avg:", origin_loss_avg)
    print("Finished testing")
    return test_result, origin_result



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 设置--max-model-len为2848
    # set up scorer
    scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B")  # on GPU
    # scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", quantized=True) # 4 bit quantization on GPU
    # scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", use_vllm=True) # VLLM on GPU, about 5 instances per seconds
    # scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B-GGUF", use_llamacpp=True) # 4 bit quantization on CPU
    client = OpenAI(api_key="sk-0c2e4c0ec7444bc7924a645788c4dd24", base_url="https://api.deepseek.com/v1")
    chatbot = "deepseek-chat"
    # client = OpenAI(api_key="sk-HPOmC99SEkbTxygFd28Nba6785yOocrSpDqzLu94FafdXqOW", base_url="https://api.moonshot.cn/v1")
    # chatbot = "moonshot-v1-8k"
    # 读取数据
    data = json.load(open("data/translation_data200.json", 'r', encoding='utf-8'))
    pmi_data = "data/vocab_translation_all.txt"
    train_data = data[:int(0.7 * len(data))]
    test_data = data[int(0.7 * len(data)):]

    # 设置超参数
    learning_rate = 1e-3
    ngram_list = pmi()
    prompt_length = 35
    prompt_search_space = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    prompts_probs = torch.load("data/prompt_probs_pair.pt",weights_only=True)

    test_result, origin_result = test(5)

    print("test_result:", sum(test_result))
    print("origin_result:", sum(origin_result))
    print("Finished")