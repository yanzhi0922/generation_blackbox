from openai import OpenAI
import os
import json
import subprocess
import os
import time
from tigerscore import TIGERScorer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from torch.optim import Adam


def query(instruction, input_context, prompt=""):
    message = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt + instruction + input_context},
    ]
    hypo_output = None
    received = False
    while not received:
        try:
            hypo_output = client.chat.completions.create(
                model=chatbot,
                messages=message,
                stream=False
            )
            received = True
        except:
            time.sleep(1)
    return hypo_output.choices[0].message.content


def tigerloss(instruction, input_context, output):
    received = False
    while not received:
        try:
            results = scorer.score([instruction], [output], [input_context])
            received = True
        except:
            time.sleep(1)
    score = results[0].get('score', 0)  # 使用 get 方法提供默认值 0
    loss = - score if score is not None else 0
    return loss


def pmi():
    result = []
    with open(pmi_data, 'r') as f:
        for line in f:
            result.append(line.strip('\n'))
    unique = []
    [unique.append(i) for i in result if not i in unique]
    ngram_index_list = list(unique)
    return ngram_index_list


def solve_v_total_exact(prompt_emb):
    k = 1
    a, b = 0, 0

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
        new_embed = prompt_embeds[i].sub(v).clamp(0, 1)
        prompt_embeds[i].data = new_embed.data


def train(epochs, sample_size):
    prompt_optimizer = Adam([{
        "params": [prompts_probs],
        "weight_decay": 0,
        "lr": prompt_learning_rate
    }, ])

    best_eval_result = 0.0
    best_prompts_probs = None
    best_epoch = 0
    eval_results = []
    test_results = []

    for epoch in range(epochs):
        random.shuffle(train_data)
        for item in train_data:
            prompts_dist = torch.distributions.Categorical(prompts_probs)
            loss_list = []
            prompts_discrete_indices_list = []
            for k in range(sample_size):
                prompts_discrete_indices = prompts_dist.sample()
                prompts_discrete_indices_list.append(prompts_discrete_indices)
                prompts_discrete_ngram_list = []
                indices_list = prompts_discrete_indices.int().tolist()
                for idx in indices_list:
                    prompts_discrete_ngram_list.append(ngram_list[idx])
                prompts_discrete = ' '.join(prompts_discrete_ngram_list)
                instruction = item['instruction']
                input_context = item['input_context']
                hypo_output = query(instruction, input_context, prompts_discrete)
                loss = tigerloss(instruction, input_context, hypo_output)
                loss_list.append(loss)
            loss_avg = sum(loss_list) / sample_size

            prompt_optimizer.zero_grad()

            derivative = (-1 / prompts_probs*temperature).repeat(sample_size, 1, 1)

            for k, prompts_discrete_indices in enumerate(prompts_discrete_indices_list):
                for i in range(prompt_length):
                    derivative[k][i][prompts_discrete_indices[i]] += 1/temperature

            prompts_probs.grad = torch.zeros_like(prompts_probs)

            for k in range(sample_size):
                prompts_probs.grad += 1 / (sample_size - 1) * (loss_list[k] - loss_avg) * derivative[k]

            torch.nn.utils.clip_grad_norm_(prompts_probs, 3)
            prompt_optimizer.step()
            constrainScoreByWholeExact(prompts_probs)
            print("epoch:", epoch, "loss_avg:", loss_avg)

        eval_result = evaluate()

        # 保存最好的结果
        if eval_result > best_eval_result:
            best_eval_result = eval_result
            best_prompts_probs = prompts_probs

        # 如果cuda内存不够，清理一下
        if 'cuda' in str(device):
            torch.cuda.empty_cache()
    save_path = "data/best_prompts_probs.pt"
    torch.save(best_prompts_probs, save_path)
    return


def evaluate(sample_size=5):
    eval_result = 0
    for item in test_data:
        prompts_dist = torch.distributions.Categorical(prompts_probs)
        loss_list = []
        for k in range(sample_size):
            prompts_discrete_indices = prompts_dist.sample()
            prompts_discrete_ngram_list = []
            indices_list = prompts_discrete_indices.int().tolist()
            for idx in indices_list:
                prompts_discrete_ngram_list.append(ngram_list[idx])
            prompts_discrete = ' '.join(prompts_discrete_ngram_list)
            instruction = item['instruction']
            input_context = item['input_context']
            hypo_output = query(instruction, input_context, prompts_discrete)
            loss = tigerloss(instruction, input_context, hypo_output)
            loss_list.append(loss)
        loss_avg = sum(loss_list) / sample_size
        print("eval_loss_avg:", loss_avg)
        eval_result += loss_avg
    eval_result /= len(test_data)
    return eval_result


def test(sample_size=10):
    test_result = []
    origin_result = []
    for item in test_data:
        prompts_dist = torch.distributions.Categorical(prompts_probs)
        loss_list = []
        origin_loss_list = []
        for k in range(sample_size):
            prompts_discrete_indices = prompts_dist.sample()
            prompts_discrete_ngram_list = []
            indices_list = prompts_discrete_indices.int().tolist()
            for idx in indices_list:
                prompts_discrete_ngram_list.append(ngram_list[idx])
            prompts_discrete = ' '.join(prompts_discrete_ngram_list)
            instruction = item['instruction']
            input_context = item['input_context']
            output = query(instruction, input_context, prompts_discrete)
            output_origin = query(instruction, input_context)
            loss = tigerloss(instruction, input_context, output)
            origin_loss = tigerloss(instruction, input_context, output_origin)
            loss_list.append(loss)
            origin_loss_list.append(origin_loss)
        loss_avg = sum(loss_list) / sample_size
        origin_loss_avg = sum(origin_loss_list) / sample_size
        test_result.append(loss_avg)
        origin_result.append(origin_loss_avg)
        print("test_loss_avg:", loss_avg)
        print("origin_loss_avg:", origin_loss_avg)
    return test_result, origin_result


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # set up scorer
    scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B")  # on GPU
    # scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", quantized=True) # 4 bit quantization on GPU
    # scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", use_vllm=True) # VLLM on GPU, about 5 instances per seconds
    # scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B-GGUF", use_llamacpp=True) # 4 bit quantization on CPU

    client = OpenAI(api_key="sk-608af9ac56514bba9ffa078eca84fde7", base_url="https://api.deepseek.com/v1")
    chatbot = "deepseek-chat"
    # client = OpenAI(api_key="sk-HPOmC99SEkbTxygFd28Nba6785yOocrSpDqzLu94FafdXqOW", base_url="https://api.moonshot.cn/v1")
    # chatbot = "moonshot-v1-8k"
    data = json.load(open("data/cut_data.json", 'r', encoding='utf-8'))
    pmi_data = "data/pmi_mrpc_gpt.txt"
    train_data = data[:int(0.1 * len(data))]
    test_data = data[int(0.9 * len(data)):]
    prompt_learning_rate = 1e-3
    ngram_list = pmi()
    prompt_length = 10
    prompt_search_space = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    prompts_probs = torch.FloatTensor([[1 / prompt_search_space] * prompt_search_space] * prompt_length)
    prompts_probs.requires_grad = True
    temperature = 1
    train(epochs=1, sample_size=10)
    print("Finished training")
    test_result, origin_result = test(10)
    print("Finished testing")
    print("test_result:", sum(test_result))
    print("origin_result:", sum(origin_result))
    print("Finished")