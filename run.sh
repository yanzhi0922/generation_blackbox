#!/bin/bash

seed=42
api_limit=8000
prompt_length=20
prompt_search_space=50
gpu_no=3
tn=sst2 # mrpc
cmd="CUDA_VISIBLE_DEVICES=$gpu_no python /tmp/pycharm_project_492/tmp_test.py"
echo "$cmd"
bash -c "$cmd"