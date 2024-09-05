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
        random.shuffle(train_data)
        for item in train_data:
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
                hypo_outputs = query_concurrently(prompts, item['instruction'], item['input_context'])
                for hypo_output in hypo_outputs:
                    loss = tigerloss(item['instruction'], item['input_context'], hypo_output)
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

    save_path = "data/prompt_probs_pair.pt"
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
    for item in test_data:
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
        hypo_outputs = query_concurrently(prompts, item['instruction'], item['input_context'])
        for hypo_output in hypo_outputs:
            loss = tigerloss(item['instruction'], item['input_context'], hypo_output)
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
    learning_rate = 1e-4
    ngram_list = pmi()
    prompt_length = 35
    prompt_search_space = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    prompts_probs = torch.FloatTensor([[1 / prompt_search_space] * prompt_search_space] * prompt_length)
    prompts_probs.requires_grad = True
    probs_change_threshold = 1e-4
    patience_threshold = 5
    checkpoint_interval = 5


    # 设置优化器
    prompt_optimizer = AdamW([{
        "params": [prompts_probs],
        "weight_decay": 0.1,
    }], lr=learning_rate)

    best_probs = train(epochs=20, sample_size=10)
    test_result, origin_result = test(5, best_probs)

    print("test_result:", sum(test_result))
    print("origin_result:", sum(origin_result))
    print("Finished")
