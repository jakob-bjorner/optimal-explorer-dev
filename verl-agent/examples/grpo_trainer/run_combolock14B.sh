set -x
ENGINE=${ENGINE:-vllm}
# ENGINE=vllm
ASYNC=${ASYNC:-False}
# DSET=${DSET:-interaction_base_base}
SEED=${SEED:-1}
DEBUG=${DEBUG:-}
SINGLE_CTX=${SINGLE_CTX:-False}
MULTI_MSG=${MULTI_MSG:-True}
if [ $SINGLE_CTX == True ]; then
    MAX_PROMPT_LEN=8096
else
    MAX_PROMPT_LEN=1024 #2048
fi

train_data_size=${train_data_size:-18}
val_data_size=6
group_size=2
B=${B:-14}
GRADE_BELIEF=${GRADE_BELIEF:-0.0}
FULL_HIST_BELIEF=${FULL_HIST_BELIEF:-False}
GRADE_BELIEF_TYPE=${GRADE_BELIEF_TYPE:-0}
MODEL_NAME=${MODEL_NAME:-"qwen/qwen2.5-${B}b-instruct"}

# invalid_action_penalty_coef=0.0
# also their kl is 0.01 lol.
# Need to change the max_prompt_len hyper param for real run. ahh its probably ok actually.
# max steps is low for testing

# We only use data preparation to indicate the modality and the data size.
# python3 -m examples.data_preprocess.prepare \
#     --mode 'text' \
#     --train_data_size $train_data_size \
#     --val_data_size $val_data_size

# jakob: this along with rollout.mode=async below are changes for testing vllm in async mode. Also I commented the XFORMERS because its incompatible with V1
#     actor_rollout_ref.rollout.mode=async \
# export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# this doesn't seem useful other than specifying the number of workers.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# data.train_files=/nas/ucb/jbjorner3/dev/optimal-explorer-dev/data/multi_turn_combo_lock_$DSET/train.parquet \
# data.val_files=/nas/ucb/jbjorner3/dev/optimal-explorer-dev/data/multi_turn_combo_lock_$DSET/train.parquet \
# CUDA_VISIBLE_DEVICES="0,1,2,3" SINGLE_CTX=True MULTI_MSG=False bash examples/grpo_trainer/run_combolock.sh; CUDA_VISIBLE_DEVICES="0,1,2,3" SINGLE_CTX=True MULTI_MSG=True bash examples/grpo_trainer/run_combolock.sh
# ENGINE=sglang SEED=3 DEBUG=_debugVLLMSGLANG bash examples/grpo_trainer/run_combolock.sh 
# ENGINE=sglang ASYNC=True SEED=3 DEBUG=_debugVLLMSGLANG_ASYNC bash examples/grpo_trainer/run_combolock.sh 

### ppo setup, and change adv_estimator to ppo or remove.
# critic.optim.lr=1e-5 \
# critic.model.use_remove_padding=True \
# critic.optim.lr_warmup_steps_ratio=0.05 \
# critic.model.path=qwen/qwen2.5-7b-instruct \
# critic.model.enable_gradient_checkpointing=True \
# critic.ppo_micro_batch_size_per_gpu=2 \
# critic.model.fsdp_config.param_offload=True \
# critic.model.fsdp_config.optimizer_offload=True \

# going to interactive debug some run with belief state grading. Need to get something training before tonight to see how it does against a baseline.
# have we even trained something in this environment? What do we have going in this setting?
# lets do a 3 b model for testing, and have belief states generated. 
# B=3 train_data_size=4 GRADE_BELIEF=1.0 DEBUG=combolock_vs_r1_test bash examples/grpo_trainer/run_combolock.sh
# B=7 train_data_size=16 SEED=3 GRADE_BELIEF=2.0 DEBUG=combolock_vs_r1_new_parse5 bash examples/grpo_trainer/run_combolock.sh
# B=7 train_data_size=16 SEED=3 SINGLE_CTX=True MULTI_MSG=False DEBUG=combolock_vanilla bash examples/grpo_trainer/run_combolock.sh; B=7 train_data_size=16 SEED=3 SINGLE_CTX=True MULTI_MSG=True DEBUG=combolock_belief_prompting bash examples/grpo_trainer/run_combolock.sh
# B=7 train_data_size=16 SEED=1 FULL_HIST_BELIEF=True MAX_PROMPT_LEN=4096 DEBUG=combolock_full_history bash examples/grpo_trainer/run_combolock.sh;


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_NAME \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$train_data_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.optim.foreach=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.rollout.jakob_async=$ASYNC \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(($MAX_PROMPT_LEN + 1024)) \
    actor_rollout_ref.rollout.single_context=$SINGLE_CTX \
    actor_rollout_ref.rollout.belief_multiple_messages=$MULTI_MSG \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
    actor_rollout_ref.actor.single_batch=True \
    algorithm.use_kl_in_reward=False \
    env.env_name=combolock \
    env.seed=$SEED \
    env.max_steps=24 \
    env.non_terminal_penalty=1.0 \
    +env.vocab="0123456789" \
    +env.max_attempts=12 \
    +env.full_history_belief=$FULL_HIST_BELIEF \
    env.rollout.n=$group_size \
    +trainer.ceiling_belief_grading_reward=$BG_CEIL_RWD \
    +trainer.belief_state_grading_type=$GRADE_BELIEF_TYPE \
    trainer.belief_state_grading=$GRADE_BELIEF \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=grpo_qwen2.5_7b_16sfr_seed${SEED}_sc_${SINGLE_CTX}_belief_prompting_${MULTI_MSG}_BG_${GRADE_BELIEF}${DEBUG} \
    trainer.n_gpus_per_node=6 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10000 \
    trainer.total_epochs=140 \
    trainer.val_before_train=False $@
