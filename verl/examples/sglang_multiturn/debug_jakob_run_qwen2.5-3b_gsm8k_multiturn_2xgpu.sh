# run on 4xH100
# make sure your current working directory is the root of the project

set -x
export HYDRA_FULL_ERROR=1
ulimit -n 65535

PROJECT_DIR="$(pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"

# actor_rollout_ref.rollout.n=1 JAKOB 16 before or something slow. doing just rl baselines no variance reduction.
# added trainer.log_val_generations=10 and trainer.val_before_train=True.
RAY_DEBUG=1 python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name='gsm8k_multiturn_grpo' \
    algorithm.adv_estimator=grpo \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=1536 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang_async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl-tests' \
    trainer.experiment_name='debug_qwen2.5-7b_function_rm-combo_lock-async-sgl-multi-w-tool-8' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.log_val_generations=10 \
    trainer.val_before_train=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    critic.ppo_max_token_len_per_gpu=8192 \
    critic.forward_max_token_len_per_gpu=8192 \
    data.train_files=/nas/ucb/jbjorner3/data/multi_turn_combo_lock/train.parquet \
    data.val_files=/nas/ucb/jbjorner3/data/multi_turn_combo_lock/test.parquet \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/combo_lock_tool_config.yaml" \
    reward_model.reward_manager=naivetool \
    +custom_reward_function.path=/nas/ucb/jbjorner3/dev/optimal-explorer-dev/verl/verl/utils/reward_score/multi_turn_combo_lock.py \
    $@