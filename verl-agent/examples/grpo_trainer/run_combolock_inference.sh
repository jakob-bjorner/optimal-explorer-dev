# set -x
ENGINE=${ENGINE:-vllm}
# ENGINE=vllm
TEMPERATURE=${TEMPERATURE:-1.0}
VOCAB=${VOCAB:-0123456789}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-12}
MAX_STEPS=$(($MAX_ATTEMPTS * 2))
CKPT=${CKPT:-}
SINGLE_CTX=${SINGLE_CTX:-False}
MULTI_MSG=${MULTI_MSG:-True}

if [ $SINGLE_CTX == True ]; then
    echo "True single context"
    if [ $MULTI_MSG == False ]; then
        MAX_STEPS=$MAX_ATTEMPTS
    fi
    MAX_PROMPT_LEN=16384
    max_num_batched_tokens=$(($MAX_PROMPT_LEN * 2))
else
    MAX_PROMPT_LEN=2048
    max_num_batched_tokens=8196
fi
# DSET=${DSET:-interaction_base_base}
export VLLM_ATTENTION_BACKEND=XFORMERS

train_data_size=${train_data_size:-256}
# 16 for training
# minbatch should be 8 on 4 gpus. and 16 on 8 gpus to match the micro per gpu assert.
val_data_size=8
group_size=2
# invalid_action_penalty_coef=0.0
# also their kl is 0.01 lol.
# Need to change the max_prompt_len hyper param for real run. ahh its probably ok actually.
# max steps is low for testing

# We only use data preparation to indicate the modality and the data size.
# python3 -m examples.data_preprocess.prepare \
#     --mode 'text' \
#     --train_data_size $train_data_size \
#     --val_data_size $val_data_size

# this doesn't seem useful other than specifying the number of workers.
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# data.train_files=/nas/ucb/jbjorner3/dev/optimal-explorer-dev/data/multi_turn_combo_lock_$DSET/train.parquet \
# data.val_files=/nas/ucb/jbjorner3/dev/optimal-explorer-dev/data/multi_turn_combo_lock_$DSET/train.parquet \

# use this when you have some run you want to evaluate at a particular saved step.
#   trainer.resume_from_path=checkpoints/verl_agent_alfworld/grpo_qwen2.5_7b_16sfr${CKPT} \ 
# CKPT="_seed1/global_step_20" VOCAB='abcdefghij' bash examples/grpo_trainer/run_combolock_inference.sh
# CUDA_VISIBLE_DEVICES="4,5,6,7" CKPT="_seed1/global_step_40" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CUDA_VISIBLE_DEVICES="4,5,6,7" CKPT="_seed1/global_step_60" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh;CUDA_VISIBLE_DEVICES="4,5,6,7" CKPT="_seed1/global_step_80" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CUDA_VISIBLE_DEVICES="4,5,6,7" CKPT="_seed1/global_step_100" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh
# CUDA_VISIBLE_DEVICES="4,5,6,7" CKPT="_seed1/global_step_120" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CUDA_VISIBLE_DEVICES="4,5,6,7" CKPT="_seed1/global_step_140" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CUDA_VISIBLE_DEVICES="4,5,6,7" CKPT="_seed2/global_step_20" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CUDA_VISIBLE_DEVICES="4,5,6,7" CKPT="_seed2/global_step_40" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CUDA_VISIBLE_DEVICES="4,5,6,7" CKPT="_seed2/global_step_60" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh;CUDA_VISIBLE_DEVICES="4,5,6,7" CKPT="_seed2/global_step_80" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CUDA_VISIBLE_DEVICES="4,5,6,7" CKPT="_seed2/global_step_100" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CUDA_VISIBLE_DEVICES="4,5,6,7" CKPT="_seed2/global_step_120" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CUDA_VISIBLE_DEVICES="4,5,6,7" CKPT="_seed2/global_step_140" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh
# CKPT="/global_step_20" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="/global_step_40" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="/global_step_60" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="/global_step_80" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="/global_step_100" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="/global_step_120" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="/global_step_140" MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh
# CKPT="_seed1_sc_True_belief_prompting_False/global_step_20" SINGLE_CTX=True MULTI_MSG=False MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="_seed1_sc_True_belief_prompting_False/global_step_40" SINGLE_CTX=True MULTI_MSG=False MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="_seed1_sc_True_belief_prompting_False/global_step_60" SINGLE_CTX=True MULTI_MSG=False MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="_seed1_sc_True_belief_prompting_False/global_step_80" SINGLE_CTX=True MULTI_MSG=False MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="_seed1_sc_True_belief_prompting_False/global_step_100" SINGLE_CTX=True MULTI_MSG=False MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="_seed1_sc_True_belief_prompting_False/global_step_120" SINGLE_CTX=True MULTI_MSG=False MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="_seed1_sc_True_belief_prompting_False/global_step_140" SINGLE_CTX=True MULTI_MSG=False MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; 

# CUDA_VISIBLE_DEVICES="0,1,2,3" MAX_ATTEMPTS=16 train_data_size=256 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CUDA_VISIBLE_DEVICES="0,1,2,3" SINGLE_CTX=True MULTI_MSG=False MAX_ATTEMPTS=16 train_data_size=256 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CUDA_VISIBLE_DEVICES="0,1,2,3" SINGLE_CTX=True MULTI_MSG=True MAX_ATTEMPTS=16 train_data_size=256 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh

# CKPT="_seed1_sc_True_belief_prompting_False/global_step_140" SINGLE_CTX=True MULTI_MSG=False MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh

# CKPT="_seed1_sc_True_belief_prompting_True/global_step_20" SINGLE_CTX=True MULTI_MSG=True MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="_seed1_sc_True_belief_prompting_True/global_step_40" SINGLE_CTX=True MULTI_MSG=True MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="_seed1_sc_True_belief_prompting_True/global_step_60" SINGLE_CTX=True MULTI_MSG=True MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="_seed1_sc_True_belief_prompting_True/global_step_80" SINGLE_CTX=True MULTI_MSG=True MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="_seed1_sc_True_belief_prompting_True/global_step_100" SINGLE_CTX=True MULTI_MSG=True MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="_seed1_sc_True_belief_prompting_True/global_step_120" SINGLE_CTX=True MULTI_MSG=True MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; CKPT="_seed1_sc_True_belief_prompting_True/global_step_140" SINGLE_CTX=True MULTI_MSG=True MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh

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
    actor_rollout_ref.model.path=qwen/qwen2.5-7b-instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.single_context=$SINGLE_CTX \
    actor_rollout_ref.rollout.belief_multiple_messages=$MULTI_MSG \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
    actor_rollout_ref.actor.single_batch=True \
    algorithm.use_kl_in_reward=False \
    env.env_name=combolock \
    env.seed=0 \
    env.max_steps=$MAX_STEPS \
    env.non_terminal_penalty=1.0 \
    +env.vocab=$VOCAB \
    +env.max_attempts=$MAX_ATTEMPTS \
    env.rollout.n=$group_size \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.resume_from_path=checkpoints/verl_agent_alfworld/grpo_qwen2.5_7b_16sfr${CKPT} \
    trainer.experiment_name=grpo_qwen2.5_7b_b${train_data_size}_t_${TEMPERATURE}_v_${VOCAB}_m_${MAX_ATTEMPTS}_ckpt_${CKPT}_sc_${SINGLE_CTX}_belief_prompting_${MULTI_MSG}_inference \
    trainer.only_gen_once=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=400 \
    trainer.val_before_train=False $@
