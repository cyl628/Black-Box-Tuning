python bbt.py \
  --task_name "sst2" \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 16 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "hinge" \
  --cat_or_add "add" \
  --budget 5000 \
  --print_every 50 \
  --eval_every 100 \
  --inference_framework "ort"