from openai import OpenAI
client = OpenAI(
  api_key="sk-tFIIkEXB5icUKs6lWPTvbAwuG09hZxr3mx4xLJoq2unt38b5",
  base_url="https://api.moonshot.cn/v1",
)
prompt = [
    {"role": "system", "content":"Definition: What are a the How to you	input: sentence one: What are the best romantic songs in English? sentence two: What are the some of the best romantic songs in English? equivalent?\noutput: no"}]

response =  client.completions.create(
  model="moonshot-v1-8k",
  messages=prompt,
  max_tokens=1,
  temperature=0.5,
  logprobs=5,
  stop='\n'
  )
# 打印返回的结果的结构
#print(response.choices)

for a, ans in enumerate(response.choices):
    print(ans.logprobs.tokens)