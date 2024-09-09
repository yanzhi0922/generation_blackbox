from datasets import load_dataset
from rouge import Rouge
data = load_dataset("abisee/cnn_dailymail", "3.0.0", cache_dir="/hy-tmp/cache")
train_data = data["train"]
test_data = data["test"]
eval_data = data['validation']
print(train_data[0]["article"])
print(train_data[0]["highlights"])
print(len(train_data))
print(test_data[0]["article"])
print(test_data[0]["highlights"])
print(len(test_data))
print(eval_data[0]["article"])
print(eval_data[0]["highlights"])
print(len(eval_data))
