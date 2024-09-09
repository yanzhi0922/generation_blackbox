from numba.tests.complex_usecases import real_usecase
from openai import OpenAI
import json
import subprocess
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'      # AutoDL
#os.environ['HF_HOME'] = 'D:/tmp/cache'      # win11
#os.environ["HF_HOME"] = "/mnt/d/tmp/cache"  # wsl2
os.environ["HF_HOME"] = "/hy-tmp/cache"      # 恒源云
#os.environ["GPU_MEMORY_UTILIZATION"] = "0.5"
#os.environ["GPU_MEMORY_LIMIT"] = "0.5"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from tigerscore import TIGERScorer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from torch.optim import Adam, AdamW
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import transformers
from modelscope import snapshot_download
from transformers import AutoTokenizer
from datasets import load_dataset
from rouge import Rouge


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
                max_tokens=max_new_tokens
            )
            received = True
        except:
            time.sleep(1)
    return hypo_output.choices[0].message.content

def query_concurrently(prompts, instruction, input_context):
    if use_llama3:
        results = []
        for i in range(len(prompts)):
            input_ = prompts[i] + " " + instruction + input_context
            sequence = pipeline(
                input_,
                # do_sample=True,
                # top_k=10,
                num_return_sequences=1,
                eos_token_id=tokenizer1.eos_token_id,
                # truncation=True,
                max_new_tokens=64,
                return_full_text=False
            )
            tmp = sequence[0]['generated_text']
            while tmp is None or tmp[1]=='.' or len(tmp) < 10:
                tmp_ = pipeline(
                    prompts[i] + " " + instruction + input_context,
                    # do_sample=True,
                    # top_k=10,
                    num_return_sequences=1,
                    eos_token_id=tokenizer1.eos_token_id,
                    # truncation=True,
                    max_new_tokens=64,
                    return_full_text=False
                )
                tmp = tmp_[0]['generated_text']
            print("prompts[i]:", prompts[i])
            print("instruction+input_context:", instruction + input_context)
            print("output:", tmp)
            print("\n\n\n")
            results.append(tmp)
        return results
    else:
        with ThreadPoolExecutor(max_workers=len(prompts)) as executor:
            futures = [executor.submit(query, p + " " + instruction + input_context) for p in prompts]
            results = [future.result() for future in futures]
        return results

def query_concurrently_sample_size(sample_size, instruction, input_context):
    if use_llama3:
        inputs = [instruction + input_context] * sample_size
        sequences = pipeline(
            inputs,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer1.eos_token_id,
            max_new_tokens=64,
            truncation=True,
            return_full_text=False
        )
        results = []
        for s in sequences:
            result = s[0]['generated_text']
            results.append(result)
        return results
    else:
        with ThreadPoolExecutor(max_workers=sample_size) as executor:
            futures = [executor.submit(query, instruction + input_context) for _ in range(sample_size)]
            results = [future.result() for future in futures]
        return results

def loss_tiger(instruction, input_context, output):
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

def loss_rouge(data_, output):
    rouge = Rouge()
    rouge_dict = rouge.get_scores(output, data_['highlights'])[0]
    rouge_l = rouge_dict['rouge-1']['r']
    loss = 1 - rouge_l
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
    best_eval_result = 0.0
    best_probs = None
    best_epoch = 0
    eval_results = []
    test_results = []
    train_losses = []
    patience = 0
    prev_prompts_probs = prompts_probs.clone()

    for epoch in range(epochs):
        epoch_loss = 0
        #random.shuffle(train_data)
        for item in train_data:
            if (item is None) or (item['article'] is None) or (item['highlights'] is None):
                continue
            prompts_dist = torch.distributions.Categorical(prompts_probs)
            with torch.no_grad():
                loss_list = []
                prompts_discrete_indices_list = []
                prompts = []
                for k in range(sample_size):
                    prompts_discrete_indices = prompts_dist.sample()
                    prompts_discrete_indices_list.append(prompts_discrete_indices)
                    prompts_discrete_ngram_list = []
                    indices_list = prompts_discrete_indices.int().tolist()
                    for idx in indices_list:
                        prompts_discrete_ngram_list.append(ngram_list[idx])
                    prompts_discrete = ' '.join(prompts_discrete_ngram_list)
                    prompts.append(prompts_discrete)
                hypo_outputs = query_concurrently(prompts, "Summarize the news article below.", item['article'])
                for hypo_output in hypo_outputs:
                    loss = loss_rouge(item, hypo_output)
                    loss_list.append(loss)

                loss_avg = sum(loss_list) / sample_size
                epoch_loss += loss_avg
                print(loss_list)

                prompt_optimizer.zero_grad()

                derivative = (-1 / prompts_probs).repeat(sample_size, 1, 1)
                for k, prompts_discrete_indices in enumerate(prompts_discrete_indices_list):
                    for i in range(prompt_length):
                        derivative[k][i][prompts_discrete_indices[i]] *= -1

                prompts_probs.grad = torch.zeros_like(prompts_probs)
                for k in range(sample_size):
                    prompts_probs.grad += 1 / (sample_size - 1) * (loss_list[k] - loss_avg) * derivative[k]

                prompts_probs.grad += loss_avg
                torch.nn.utils.clip_grad_norm_(prompts_probs, 3)
                prompt_optimizer.step()
                constrainScoreByWholeExact(prompts_probs)
                print("epoch:", epoch, "loss_avg:", loss_avg)

        epoch_loss /= len(train_data)
        train_losses.append(epoch_loss)
        print("Epoch:", epoch, "Training Loss:", epoch_loss)

        probs_change = torch.norm(prompts_probs - prev_prompts_probs)
        prev_prompts_probs = prompts_probs.detach().clone()
        eval_result = evaluate()
        eval_results.append(eval_result)
        # 检查是否有性能提升或alphas参数变化是否小于阈值
        if eval_result < best_eval_result:
            best_eval_result = eval_result
            best_probs = prompts_probs.detach().clone()
            best_epoch = epoch
            print(f"New best eval result: {best_eval_result} at epoch {epoch} with probs_change {probs_change}")
            patience = 0
        else:
            patience += 1
            print(f"Patience counter: {patience}")

        if probs_change < probs_change_threshold or patience >= patience_threshold:
            print(f"Stopping training. probs_change: {probs_change}, patience: {patience}")
            break

        # 保存检查点
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = f"data/checkpoint/bdql_checkpoint_epoch_{epoch + 1}.pt"
            torch.save({
                'epoch': epoch + 1,
                'prompts_probs': prompts_probs,
                'optimizer': prompt_optimizer.state_dict(),
                'train_losses': train_losses,
                'eval_results': eval_results,
                'test_results': test_results,
                'best_eval_result': best_eval_result,
                'best_probs': best_probs,
                'best_epoch': best_epoch,
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch + 1}")

        # 如果cuda内存不够，清理一下
        if 'cuda' in str(device):
            torch.cuda.empty_cache()

    save_path = "data/prompt_probs_pair_summ.pt"
    torch.save(prompts_probs, save_path)
    print("Finished training")

    # 绘制训练和验证损失
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(eval_results, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    return best_probs

def evaluate(sample_size=5):
    eval_result = 0
    for item in eval_data:
        prompts_dist = torch.distributions.Categorical(prompts_probs)
        loss_list = []
        prompts = []
        for k in range(sample_size):
            prompts_discrete_indices = prompts_dist.sample()
            prompts_discrete_ngram_list = []
            indices_list = prompts_discrete_indices.int().tolist()
            for idx in indices_list:
                prompts_discrete_ngram_list.append(ngram_list[idx])
            prompts_discrete = ' '.join(prompts_discrete_ngram_list)
            prompts.append(prompts_discrete)
        hypo_outputs = query_concurrently(prompts, "Write a summary of the text below.", item['article'])
        for hypo_output in hypo_outputs:
            loss = loss_rouge(item, hypo_output)
            loss_list.append(loss)
        loss_avg = sum(loss_list) / sample_size
        print("eval_loss_avg:", loss_avg)
        eval_result += loss_avg
    eval_result /= len(test_data)
    return eval_result

def test(sample_size=5, probs=None):
    test_result = []
    origin_result = []
    for item in test_data:
        prompts_dist = torch.distributions.Categorical(probs)
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
        outputs = query_concurrently(prompts, "Summarize the news article below.", item['article'])
        outputs_origin = query_concurrently_sample_size(sample_size, "Summarize the news article below.", item['article'])
        for output, output_origin in zip(outputs, outputs_origin):
            loss = loss_rouge(item, output)
            origin_loss = loss_rouge(item, output_origin)
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

    # 读取数据
    data = load_dataset("abisee/cnn_dailymail", "3.0.0", cache_dir="/hy-tmp/cache")
    pmi_data = "data/vocabulary/vocab_cnn.txt"
    train_data = data['train'] # 取0.001的数据
    train_data = train_data.select(range(0, int(len(train_data) * 0.001)))
    print("train_data:", len(train_data))
    test_data = data['test']
    test_data = test_data.select(range(0, int(len(test_data) * 0.001)))
    print("test_data:", len(test_data))
    eval_data = data['validation']
    eval_data = eval_data.select(range(0, int(len(eval_data) * 0.001)))
    print("eval_data:", len(eval_data))

    # 设置超参数
    learning_rate = 1e-4
    ngram_list = pmi()
    prompt_length = 20
    prompt_search_space = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    prompts_probs = torch.FloatTensor([[1 / prompt_search_space] * prompt_search_space] * prompt_length).to(device)
    prompts_probs.requires_grad = True
    probs_change_threshold = 1e-4
    patience_threshold = 5
    checkpoint_interval = 5
    use_llama3 = True           # 是否使用llama3,当为False时使用api
    loss_type = "summarization" # "summarization" or "translation" or "TigerScore"
    max_new_tokens = 128



    # 设置优化器
    prompt_optimizer = AdamW([{
        "params": [prompts_probs],
        "weight_decay": 0.1,
    }], lr=learning_rate)


    # 设置大语言模型
    if use_llama3:
        model = "/hy-tmp/local"
        tokenizer1 = AutoTokenizer.from_pretrained(model)
        pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    else:
        client = OpenAI(api_key="sk-0c2e4c0ec7444bc7924a645788c4dd24", base_url="https://api.deepseek.com/v1")
        chatbot = "deepseek-chat"
        # client = OpenAI(api_key="sk-HPOmC99SEkbTxygFd28Nba6785yOocrSpDqzLu94FafdXqOW", base_url="https://api.moonshot.cn/v1")
        # chatbot = "moonshot-v1-8k"


    if loss_type == "TigerScore":
        # set up scorer
        scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B")  # on GPU
        # scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", quantized=True) # 4 bit quantization on GPU
        # scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", use_vllm=True) # VLLM on GPU, about 5 instances per seconds
        # scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B-GGUF", use_llamacpp=True) # 4 bit quantization on CPU

    best_probs = train(epochs=20, sample_size=10)
    test_result, origin_result = test(5, best_probs)
    print("test_result:", sum(test_result))
    print("origin_result:", sum(origin_result))
    print("Finished")