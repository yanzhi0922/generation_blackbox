import os
import json
import subprocess
import os
import time
import requests
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# set up scorer
from tigerscore import TIGERScorer

os.environ["CURL_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value

scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-13B") # on GPU
# 输出模型在本地保存的路径
# scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-13B", quantized=True) # 4 bit quantization on GPU
# scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", use_vllm=True) # VLLM on GPU, about 5 instances per seconds
# scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B-GGUF", use_llamacpp=True) # 4 bit quantization on CPU
# Instruction-following example
instruction = "Write an apology letter."
input_context = "Reason: You canceled a plan at the last minute due to illness."
hypo_output = "Hey [Recipient],\n\nI'm really glad for ditching our plan. I suddenly got an opportunity for a vacation so I took it. I know this might have messed up your plans and I love that.\n\nDespite being under the weather, I would rather go for an adventure. I hope you can understand my perspective and I hope this incident doesn't change anything between us.\n\nWe can reschedule our plan for another time. Sorry again for the trouble.\n\nPeace out,\n[Your Name]\n\n---"
results = scorer.score([instruction], [hypo_output], [input_context])
# results 输出为[{'num_errors': 3, 'score': -12.0, 'errors': {'error_0': {'location': '"I\'m really glad for ditching our plan. I suddenly got an opportunity for a vacation so I took it."', 'aspect': 'Misunderstanding context', 'explanation': 'The model seems to have misunderstood the context of the apology letter. Instead of apologizing for canceling the plan due to illness, it is apologizing for ditching the plan for a vacation. The correction would be to apologize for the inconvenience caused due to illness.', 'severity': 'Major', 'score_reduction': '4.0'}, 'error_1': {'location': '"I know this might have messed up your plans and I love that."', 'aspect': 'Inappropriate tone', 'explanation': 'The tone of this sentence is inappropriate and disrespectful. It does not show regret or remorse for the inconvenience caused. The correction would be to express regret for any inconvenience caused.', 'severity': 'Major', 'score_reduction': '4.0'}, 'error_2': {'location': '"Hey [Recipient], I\'m really glad for ditching our plan. I suddenly got an opportunity for a vacation so I took it."', 'aspect': 'Incorrect format', 'explanation': "The model's response does not follow the correct format of an apology letter. It does not express regret or remorse for the inconvenience caused. The correction would be to follow the correct format of an apology letter.", 'severity': 'Major', 'score_reduction': '4.0'}}, 'raw_output': 'You are evaluating errors in a model-generated output for a given instruction.\nInstruction: \nWrite an apology letter.\nReason: You canceled a plan at the last minute due to illness.\n\nModel-generated Output: \nHey [Recipient],\n\nI\'m really glad for ditching our plan. I suddenly got an opportunity for a vacation so I took it. I know this might have messed up your plans and I love that.\n\nDespite being under the weather, I would rather go for an adventure. I hope you can understand my perspective and I hope this incident doesn\'t change anything between us.\n\nWe can reschedule our plan for another time. Sorry again for the trouble.\n\nPeace out,\n[Your Name]\n\n---\n\nFor each error you give in the response, please also elaborate the following information:\n- error location (the words that are wrong in the output)\n- error aspect it belongs to.\n- explanation why it\'s an error, and the correction suggestions.\n- severity of the error ("Major" or "Minor"). \n- reduction of score (between 0.5 and 5 given the severity of the error)\n\nYour evaluation output: The model-generated output contains 3 errors, with a total score reduction of 12.0.\nError location 1:  "I\'m really glad for ditching our plan. I suddenly got an opportunity for a vacation so I took it."\nError aspect 1:  Misunderstanding context\nExplanation 1:  The model seems to have misunderstood the context of the apology letter. Instead of apologizing for canceling the plan due to illness, it is apologizing for ditching the plan for a vacation. The correction would be to apologize for the inconvenience caused due to illness.\nSeverity 1: Major\nScore reduction 1: 4.0\nError location 2:  "I know this might have messed up your plans and I love that."\nError aspect 2:  Inappropriate tone\nExplanation 2:  The tone of this sentence is inappropriate and disrespectful. It does not show regret or remorse for the inconvenience caused. The correction would be to express regret for any inconvenience caused.\nSeverity 2: Major\nScore reduction 2: 4.0\nError location 3:  "Hey [Recipient], I\'m really glad for ditching our plan. I suddenly got an opportunity for a vacation so I took it."\nError aspect 3:  Incorrect format\nExplanation 3:  The model\'s response does not follow the correct format of an apology letter. It does not express regret or remorse for the inconvenience caused. The correction would be to follow the correct format of an apology letter.\nSeverity 3: Major\nScore reduction 3: 4.0'}]
# 只输出'score': -12.0
TigerScore = results[0]['score']
print("TIGERScore: ", TigerScore)