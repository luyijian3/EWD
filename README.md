# An Entropy-based Text Watermarking Detection Method
## This is the official github repo for paper https://arxiv.org/abs/2403.13485
## News
💡 **_<span style="color: #0366d6;">Our proposed EWD has been integrated into [MarkLLM](https://github.com/THU-BPM/MarkLLM), an open-source toolkit for LLM watermarking. You can try EWD and many other watermarking algorithms / evaluations / piplines in the MarkLLM repo!</span>_**

💡 **_<span style="color: #0366d6;">The EWD paper has been accepted to ACL 2024 Main Conference</span>_**

### 1. Generating watermarked machine-generated code
**Important**: Note that we used different `hash_key` for MBPP&humanevl which is `15485917`, not `15485863`(default), using 15485863 will get similar results

```
accelerate launch main.py \
    --model bigcode/starcoder \
    --use_auth_token \
    --task mbpp \
    --temperature 0.2 \
    --precision bf16 \
    --batch_size 3 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 3 \
    --max_length_generation 2048 \
    --save_generations \
    --outputs_dir ./mbpp_outputs \
    --save_generations_path machine_code.json \
    --gamma 0.5 \
    --delta 2 \
    --generation_only
```

### 2. save entropy and evaluate pass@1 of machine-generated code

```
accelerate launch main.py \
    --model bigcode/starcoder \
    --use_auth_token \
    --task mbpp \
    --temperature 0.2 \
    --precision bf16 \
    --batch_size 3 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 3 \
    --max_length_generation 2048 \
    --load_generations_path ./mbpp_outputs/machine_code.json \
    --outputs_dir ./mbpp_outputs \
    --gamma 0.5 \
    --delta 2
```

### 3. save spike entropy and evaluate pass@1 of human-written code
```
accelerate launch main.py \
    --model bigcode/starcoder \
    --use_auth_token \
    --task mbpp \
    --temperature 0.2 \
    --precision bf16 \
    --batch_size 3 \
    --allow_code_execution \
    --do_sample \
    --top_p 0.95 \
    --n_samples 3 \
    --max_length_generation 2048 \
    --detect_human_code \
    --outputs_dir ./mbpp_outputs \
    --metric_output_path human_results.json \
    --gamma 0.5 \
    --delta 2
```

### 4. calculate detection performance
```
python calculate_auroc_tpr.py \
    --human_fname ./mbpp_outputs/human_results.json \
    --machine_fname ./mbpp_outputs/evaluation_results.json \
    --min_length 15 \
    --max_length 999999
```