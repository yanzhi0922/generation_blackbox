import torch
import torch.nn.functional as F
temperature = 10
prompt_search_space = 5
prompt_length = 5
alphas = torch.FloatTensor([[1, 2, 3, 4, 5],
                            [1, 3, 5, 7, 9],
                            [1, 4, 7, 10, 13],
                            [1, 5, 9, 13, 17],
                            [1, 6, 11, 16, 21]])
prompts_probs = F.gumbel_softmax(alphas, tau=temperature, hard=False)
print(alphas)
print(prompts_probs)
print(prompts_probs.shape)