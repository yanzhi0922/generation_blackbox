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
from torch import autograd


def query(instruction, input_context, prompt=""):
    message = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt + instruction + input_context},
    ]
    hypo_output = None
    received = False
    while (not received) or (hypo_output is None):
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
            result = scorer.score([instruction], [output], [input_context])
            received = True
        except:
            time.sleep(1)
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

def train(epochs, sample_size):
    alphas_optimizer = Adam([{
        "params": [alphas],
        "weight_decay": 0,
        "lr": learning_rate
    }, ])

    best_eval_result = 0.0
    best_alphas = None
    best_epoch = 0
    eval_results = []
    test_results = []
    prev_alphas = alphas  # 用于存储上一次迭代的alphas参数

    for epoch in range(epochs):
        random.shuffle(train_data)
        for item in train_data:
            prompts_probs = F.gumbel_softmax(alphas, tau=temperature, hard=False)
            prompts_dist = torch.distributions.Categorical(prompts_probs)
            tigerscore_list = []
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
                tigerscore = tigerloss(instruction, input_context, hypo_output)
                tigerscore_list.append(tigerscore)
            tigerscore_avg = sum(tigerscore_list) / sample_size
            
            print(torch.norm(alphas- prev_alphas))
            print("epoch:", epoch, "loss_avg:", tigerscore_avg)

        # 计算alphas参数的变化
        alphas_change = torch.norm(alphas - prev_alphas)
        print(f"Alphas change: {alphas_change}")
        prev_alphas = alphas.detach().clone()  # 更新prev_alphas为当前的alphas

        eval_result = evaluate()

        # 检查是否有性能提升或alphas参数变化是否小于阈值
        if eval_result > best_eval_result or alphas_change < alpha_change_threshold:
            best_eval_result = eval_result
            best_alphas = alphas.detach().clone()
            print(f"New best eval result: {best_eval_result} at epoch {epoch} with alphas change: {alphas_change}")

            # 如果alphas参数变化小于阈值，则停止训练
            if alphas_change < alpha_change_threshold:
                print(f"Alphas change is below threshold {alpha_change_threshold}, stopping training.")
                break

        # 如果cuda内存不够，清理一下
        if 'cuda' in str(device):
            torch.cuda.empty_cache()

    save_path = "data/best_alphas.pt"
    torch.save(best_alphas, save_path)
    print("Finished training")
    return

def evaluate(sample_size=10):
    eval_result = 0
    for item in test_data:
        prompts_probs = F.gumbel_softmax(alphas, tau=temperature, hard=False)
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
        prompts_probs = F.gumbel_softmax(alphas, tau=temperature, hard=False)
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
    print("Finished testing")
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
    # 读取数据
    data = json.load(open("data/cut_data.json", 'r', encoding='utf-8'))
    pmi_data = "data/pmi_mrpc_gpt.txt"
    train_data = data[:int(0.1 * len(data))]
    test_data = data[int(0.9 * len(data)):]

    # 设置超参数
    learning_rate = 1e-2
    ngram_list = pmi()
    prompt_length = 10
    prompt_search_space = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    alphas = torch.FloatTensor([[1] * prompt_search_space] * prompt_length)
    alphas.requires_grad = True
    temperature = 1
    alpha_change_threshold = 1e-5

    train(epochs=50, sample_size=10)
    test_result, origin_result = test(10)

    print("test_result:", sum(test_result))
    print("origin_result:", sum(origin_result))
    print("Finished")