set -x
ENGINE=${ENGINE:-vllm}
# ENGINE=vllm
ASYNC=${ASYNC:-False}
# DSET=${DSET:-interaction_base_base}
SEED=${SEED:-1}
DEBUG=${DEBUG:-}
SINGLE_CTX=${SINGLE_CTX:-False}
MULTI_MSG=${MULTI_MSG:-True}
MAX_STEPS=${MAX_STEPS:-6}
MAX_ATTEMPTS=$(($MAX_STEPS*2))
if [ $SINGLE_CTX ]; then
    MAX_PROMPT_LEN=16384
else
    MAX_PROMPT_LEN=4096
fi



train_data_size=${train_data_size:-16}
val_data_size=8
group_size=2
IS_MEM1=${IS_MEM1:-False}
INSTRUCT=${INSTRUCT:-True}

if [ $IS_MEM1 == True ]; then
    MAX_ATTEMPTS=$MAX_STEPS
fi
if [ $INSTRUCT == True ]; then
    MODEL_DESC=instruct
    MODEL_PATH=qwen/qwen2.5-7b-instruct
else
    MODEL_DESC=""
    MODEL_PATH=qwen/qwen2.5-7b
fi



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

### ppo setup, also need to cahnge adv_estimator to grpo.
# critic.optim.lr=1e-5 \
# critic.model.use_remove_padding=True \
# critic.optim.lr_warmup_steps_ratio=0.05 \
# critic.model.path=$MODEL_PATH \
# critic.model.enable_gradient_checkpointing=True \
# critic.ppo_micro_batch_size_per_gpu=2 \
# critic.model.fsdp_config.param_offload=True \
# critic.model.fsdp_config.optimizer_offload=True \

### launch training loop in the retrieval environment
# before launching need to run 
# conda activate retriever
# bash /nas/ucb/jbjorner3/dev/optimal-explorer-dev/MEM1/Mem1/train/retrieval_launch.sh
# this will use gpus 4, and 5 for hosting the RAG endpoint to conduct search on.
# Then use GPU 0,1,2,3 for the training and rollout.
# normal model:
# DEBUG=GRPO_INSTRUCT bash examples/grpo_trainer/run_nqhotpotqa.sh
# with base model
# DEBUG=PPO2 INSTRUCT=False bash examples/grpo_trainer/run_nqhotpotqa.sh
# with base model and mem1 enables with ppo (REMEMBER TO CHANGE TO PPO CRITIC WHEN LAUNCHING!!!)
# DEBUG=MEM1GRPO32 IS_MEM1=True INSTRUCT=False bash examples/grpo_trainer/run_nqhotpotqa.sh
# DEBUG=MEM1GRPO64 train_data_size=64 IS_MEM1=True INSTRUCT=False bash examples/grpo_trainer/run_nqhotpotqa.sh

# peak token lengths experiment: training for 260
# DEBUG=GRPO_INSTRUCT bash examples/grpo_trainer/run_nqhotpotqa.sh
# DEBUG=GRPO_INSTRUCT IS_MEM1=True bash examples/grpo_trainer/run_nqhotpotqa.sh 
# the instruct version of this model should be tuned at least sensibly, because there could be minor things which cause it's performance to be worse than we expect. And we want the sensible thing to happen that it performs better than MEM1.

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=1000 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
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
    actor_rollout_ref.rollout.jakob_async=$ASYNC \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$(($MAX_PROMPT_LEN * 2)) \
    actor_rollout_ref.rollout.single_context=$SINGLE_CTX \
    actor_rollout_ref.rollout.belief_multiple_messages=$MULTI_MSG \
    actor_rollout_ref.rollout.instruct=$INSTRUCT \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
    actor_rollout_ref.actor.single_batch=True \
    algorithm.use_kl_in_reward=False \
    env.env_name=nqhotpotqa \
    env.seed=$SEED \
    env.max_steps=$MAX_STEPS \
    env.non_terminal_penalty=0.0 \
    env.rollout.n=$group_size \
    +env.split=train \
    +env.max_attempts=$MAX_ATTEMPTS \
    +env.num_objectives=2 \
    +env.max_obs_length=1000 \
    +env.topk=3 \
    +env.is_mem1=$IS_MEM1 \
    +env.search_url="http://127.0.0.1:8013/retrieve" \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=nqhotpotqa_grpo_qwen2.5-7b-${MODEL_DESC}_16sfr_seed${SEED}_sc_${SINGLE_CTX}_belief_prompting_${MULTI_MSG}_is_mem1_${IS_MEM1}${DEBUG} \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=10000 \
    trainer.total_epochs=400 \
    trainer.val_before_train=False $@
