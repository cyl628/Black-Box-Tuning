nohup python -u bbt.py \
  --task_name "sst2" \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 16 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "ce" \
  --cat_or_add "add" \
  --budget 8000 \
  --print_every 50 \
  --eval_every 100 > log_sst2.txt

nohup python -u bbt.py \
  --task_name "yelpp" \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 16 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "ce" \
  --cat_or_add "add" \
  --budget 8000 \
  --print_every 50 \
  --eval_every 100 > log_yelpp.txt

nohup python -u bbt.py \
  --task_name "agnews" \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 16 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "ce" \
  --cat_or_add "add" \
  --budget 8000 \
  --print_every 50 \
  --eval_every 100 > log_agnews.txt

nohup python -u bbt.py \
  --task_name "dbpedia" \
  --n_prompt_tokens 50 \
  --intrinsic_dim 500 \
  --k_shot 16 \
  --device "cuda:0" \
  --seed 42 \
  --loss_type "ce" \
  --cat_or_add "add" \
  --budget 8000 \
  --print_every 50 \
  --eval_every 100 > log_dbpedia.txt


