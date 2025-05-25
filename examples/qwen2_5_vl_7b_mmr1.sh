set -x

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct

FORMAT_PROMPT="\nPlease reason step by step, and put your final answer within \boxed{}."

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=MMR1/MMR1-Math-RL-Data-v0@train \
    data.val_files=hiyouga/geometry3k@test \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_mmr1 \
    trainer.n_gpus_per_node=8 \
    data.val_batch_size=500 \
    data.max_pixels=1204224 \
    trainer.total_episodes=15 \
    trainer.save_limit=7 \
    worker.reward.score_function=./examples/score_function/math.py:compute_score \

