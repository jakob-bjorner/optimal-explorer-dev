# now with base model (not sure how to do this yet, but very confident the codebase can be adapted whenever I'd like.)
    # data.return_raw_chat=True \ # this is used for multi step tool calling. not for using base models.
# try to see where the chat template is applied, and where the 
# grpo
    # don't know what these are necessary for?
    # actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    # actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    # actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    # critic.ppo_max_token_len_per_gpu=8192 \
    # critic.forward_max_token_len_per_gpu=8192 \
# moving to larger model 3 B model? try see how long training takes, and compare to performance I can get for 
# GSM8k in the zero shot setting
# 7B model.
# srun --pty --qos scavenger --partition scavenger --mem=125GB --cpus-per-task=16 --gpus-per-node=A100-PCI-80GB:4 --time 300 -J rdev bash 
WANDB_API_KEY=d20252b9059d2790c321604fd8d850aa70c2e2d4 RAY_DEBUG=1 SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 data.train_files=/nas/ucb/jbjorner3/data/single_turn_combo_lock_history_posterior/train.parquet \
 data.val_files=/nas/ucb/jbjorner3/data/single_turn_combo_lock_history_posterior/test.parquet \
 algorithm.adv_estimator=grpo \
 data.train_batch_size=256 \
 data.max_prompt_length=2048 \
 data.max_response_length=256 \
 actor_rollout_ref.rollout.name=sglang \
 actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 algorithm.use_kl_in_reward=False \
 trainer.critic_warmup=0 \
 trainer.logger=['console','wandb'] \
 trainer.project_name='verl-tests' \
 trainer.experiment_name='grpo-3b-combo-history-posterior-0' \
 trainer.log_val_generations=10 \
 trainer.validation_data_dir=/nas/ucb/jbjorner3/dev/optimal-explorer-dev/verl/checkpoints/verl-tests/grpo-3b-combo-history-posterior-0/ \
 trainer.val_before_train=True \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=4 \
 trainer.nnodes=1 \
 trainer.save_freq=10 \
 trainer.test_freq=10 \
 +custom_reward_function.path=/nas/ucb/jbjorner3/dev/optimal-explorer-dev/verl/verl/utils/reward_score/jakob_single_turn_combo_lock.py \
 trainer.total_epochs=15 2>&1 | tee verl_demo2.log
# increasing the n_gpus_per_node from 2 to 4 and increasing tensor parallelism from 1 to 2 and increment name from -0 to -1. 
# Enabled val before train.
# export NCCL_P2P_LEVEL=SYS
# WANDB_API_KEY=d20252b9059d2790c321604fd8d850aa70c2e2d4 RAY_DEBUG=1 SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
#     data.train_files=/nas/ucb/jbjorner3/data/gsm8k/train.parquet \
#     data.val_files=/nas/ucb/jbjorner3/data/gsm8k/test.parquet \
#     data.train_batch_size=1024 \
#     data.max_prompt_length=1024 \
#     data.max_response_length=1024 \
#     actor_rollout_ref.rollout.name=sglang \
#     actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=64 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=True \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
#     critic.optim.lr=1e-5 \
#     critic.model.path=Qwen/Qwen2.5-3B-Instruct \
#     critic.ppo_micro_batch_size_per_gpu=1 \
#     critic.model.fsdp_config.param_offload=True \
#     critic.model.fsdp_config.optimizer_offload=True \
#     algorithm.kl_ctrl.kl_coef=0.001 \
#     trainer.logger=['console','wandb'] \
#     trainer.project_name='verl-tests' \
#     trainer.experiment_name='ppo3b-0' \
#     trainer.val_before_train=False \
#     trainer.default_hdfs_dir=null \
#     trainer.n_gpus_per_node=8 \
#     trainer.nnodes=1 \
#     trainer.save_freq=-1 \
#     trainer.test_freq=10 \
#     trainer.total_epochs=15 2>&1 | tee verl_demo.log
# GRPO 0.5 instruct.
# WANDB_API_KEY=d20252b9059d2790c321604fd8d850aa70c2e2d4 RAY_DEBUG=1 SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True CUDA_VISIBLE_DEVICES="2,3,5,6" PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
#  data.train_files=/nas/ucb/jbjorner3/data/gsm8k/train.parquet \
#  data.val_files=/nas/ucb/jbjorner3/data/gsm8k/test.parquet \
#  algorithm.adv_estimator=grpo \
#  data.train_batch_size=256 \
#  data.max_prompt_length=512 \
#  data.max_response_length=256 \
#  actor_rollout_ref.rollout.name=sglang \
#  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
#  actor_rollout_ref.actor.optim.lr=1e-6 \
#  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
#  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
#  actor_rollout_ref.actor.use_kl_loss=True \
#  actor_rollout_ref.actor.kl_loss_coef=0.001 \
#  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#  actor_rollout_ref.actor.entropy_coeff=0 \
#  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
#  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
#  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#  algorithm.kl_ctrl.kl_coef=0.001 \
#  algorithm.use_kl_in_reward=False \
#  trainer.critic_warmup=0 \
#  trainer.logger=['console','wandb'] \
#  trainer.project_name='verl-tests' \
#  trainer.experiment_name='grpo-3b-0' \
#  trainer.log_val_generations=10 \
#  trainer.validation_data_dir=/nas/ucb/jbjorner3/dev/dev_verl/checkpoints/verl-tests/grpo-3b-0/ \
#  trainer.val_before_train=False \
#  trainer.default_hdfs_dir=null \
#  trainer.n_gpus_per_node=2 \
#  trainer.nnodes=1 \
#  trainer.save_freq=10 \
#  trainer.test_freq=10 \
#  trainer.total_epochs=15 2>&1 | tee verl_demo.log
# for ppo
# SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True CUDA_VISIBLE_DEVICES="2,3,5,6" PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
#  data.train_files=$HOME/data/gsm8k/train.parquet \
#  data.val_files=$HOME/data/gsm8k/test.parquet \
#  data.train_batch_size=256 \
#  data.max_prompt_length=512 \
#  data.max_response_length=256 \
#  actor_rollout_ref.rollout.name=sglang \
#  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
#  actor_rollout_ref.actor.optim.lr=1e-6 \
#  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
#  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
#  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
#  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
#  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#  critic.optim.lr=1e-5 \
#  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
#  critic.ppo_micro_batch_size_per_gpu=4 \
#  algorithm.kl_ctrl.kl_coef=0.001 \
#  trainer.logger=['console','wandb'] \
#  trainer.project_name='verl-tests' \
#  trainer.experiment_name='test-5-new-reward.' \
#  trainer.val_before_train=False \
#  trainer.default_hdfs_dir=null \
#  trainer.n_gpus_per_node=2 \
#  trainer.nnodes=1 \
#  trainer.save_freq=10 \
#  trainer.test_freq=10 \
#  trainer.total_epochs=15 2>&1 | tee verl_demo.log
#  +custom_reward_function.path=/nas/ucb/jbjorner3/dev/dev_verl/verl/verl/utils/reward_score/jakob_gsm8k.py \


# 25 s/it for 4 gpus and 30.41s for 2 gpus and 55 secs for 1 gpu.