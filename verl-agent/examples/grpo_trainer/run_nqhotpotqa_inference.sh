set -x
ENGINE=${ENGINE:-vllm}
# ENGINE=vllm
ASYNC=${ASYNC:-False}
# DSET=${DSET:-interaction_base_base}
TEMPERATURE=${TEMPERATURE:-1.0}
SEED=${SEED:-1}
DEBUG=${DEBUG:-}
SINGLE_CTX=${SINGLE_CTX:-False}
MULTI_MSG=${MULTI_MSG:-True}
CKPT=${CKPT:-} # one command per checkpoint.
MAX_ATTEMPTS=${MAX_ATTEMPTS:-6}
NUM_OBJECTIVES=${NUM_OBJECTIVES:-2}
if (( $NUM_OBJECTIVES >= 8 )); then
    MAX_ATTEMPTS=20
fi

MAX_STEPS=$(($MAX_ATTEMPTS * 2 - 1))
if [ $SINGLE_CTX == True ]; then
    echo "True single context"
    if [ $MULTI_MSG == False ]; then
        MAX_STEPS=$MAX_ATTEMPTS
    fi
    MAX_PROMPT_LEN=16384
    max_num_batched_tokens=$(($MAX_PROMPT_LEN * 2))
else
    MAX_PROMPT_LEN=4096
    max_num_batched_tokens=8196
fi

train_data_size=${train_data_size:-16}
val_data_size=8
group_size=1
INSTRUCT=${INSTRUCT:-True}
IS_MEM1=${IS_MEM1:-False}

if [ $IS_MEM1 == True ]; then
    MAX_STEPS=$MAX_ATTEMPTS
fi
if [ $INSTRUCT == True ]; then
    MODEL_DESC=instruct
    MODEL_PATH=qwen/qwen2.5-7b-instruct
else
    MODEL_DESC=""
    MODEL_PATH=qwen/qwen2.5-7b
fi

FORCE_FULL=${FORCE_FULL:-False}

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
# bash /nas/ucb/dayan/optimal-explorer-dev/MEM1/Mem1/train/retrieval_launch.sh
#   run in the background: ctrl + z, then bg, then fg to bring it back.
# this will use gpus 4, and 5 for hosting the RAG endpoint to conduct search on.
# Then use GPU 0,1,2,3 for the training and rollout.
# with base model
# DEBUG=PPO2 INSTRUCT=False bash examples/grpo_trainer/run_nqhotpotqa.sh

# Running evaluations on the instruct checkpoints. change MAX_ATTEMPTS based on the number of objectives.
# CKPT="_seed1_sc_False_belief_prompting_True/global_step_100" NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh


# evals:

# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;  CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh


# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=16 TEMPERATURE=1.0 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_TrueGRPO_INSTRUCT/global_step_260" train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh

# /home/jbjorner3/storage/dev/optimal-explorer-dev/verl-agent/checkpoints/verl_agent_alfworld/nqhotpotqa_grpo_qwen2.5-7b-_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueMEM1GRPO32/global_step_260
# CKPT="qwen2.5-7b-_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueMEM1GRPO32/global_step_260" IS_MEM1=True INSTRUCT=False train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueMEM1GRPO32/global_step_260" IS_MEM1=True INSTRUCT=False train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueMEM1GRPO32/global_step_260" IS_MEM1=True INSTRUCT=False train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueMEM1GRPO32/global_step_260" IS_MEM1=True INSTRUCT=False train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueMEM1GRPO32/global_step_260" IS_MEM1=True INSTRUCT=False train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;CKPT="qwen2.5-7b-_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueMEM1GRPO32/global_step_260" IS_MEM1=True INSTRUCT=False train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueMEM1GRPO32/global_step_260" IS_MEM1=True INSTRUCT=False train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueMEM1GRPO32/global_step_260" IS_MEM1=True INSTRUCT=False train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueMEM1GRPO32/global_step_260" IS_MEM1=True INSTRUCT=False train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueMEM1GRPO32/global_step_260" IS_MEM1=True INSTRUCT=False train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh


#CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT/global_step_140" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh
# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT/global_step_140" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh
# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT/global_step_140" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh
# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT/global_step_140" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh
# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT/global_step_140" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh
# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT/global_step_140" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh
# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT/global_step_140" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh
# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT/global_step_140" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh
# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT/global_step_140" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh
# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT/global_step_140" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh
# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT/global_step_140" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh

# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT/global_step_140" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh
# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT/global_step_140" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh


# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=1.0 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh
# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=1.0 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=1.0 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=1.0 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=1.0 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=0.01 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=0.01 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=0.01 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=0.01 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=0.01 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh

# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_200" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=1.0 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_200" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=1.0 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_200" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=1.0 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_200" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=1.0 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_200" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=1.0 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_200" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=0.01 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_200" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=0.01 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_200" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=0.01 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_200" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=0.01 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_3/global_step_200" IS_MEM1=False INSTRUCT=True train_data_size=128 TEMPERATURE=0.01 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh

# need to do full test set split for evaluation of the checkpoints on 1, 2, 4, 8, and 16 objectives
# something something something... What if I really don't want to? Yeah it could just be fine. Should make the case based on amount of time it would take tho.
# It might just not make sense to pay that price of like 200$ while we still might need the money for esential experiments.
# I'm pretty sure full test set eval shouldn't cost that much tho, its just inference. like why would it be so expensive???
# I could try to just test all
# 

# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.0GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_4/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_4/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_4/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.1GRPO_INSTRUCT_4/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh

# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.0GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.0GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.0GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.0GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.0GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh

# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.05GRPO_INSTRUCT_4/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.05GRPO_INSTRUCT_4/global_step_60" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.05GRPO_INSTRUCT_4/global_step_200" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.05GRPO_INSTRUCT_4/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.05GRPO_INSTRUCT_4/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.05GRPO_INSTRUCT_4/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.05GRPO_INSTRUCT_4/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh

# ABBEL len pen 0.0 (temp 0.01)
# 1 objective for MEM1
# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_TrueGRPO_INSTRUCT/global_step_260" IS_MEM1=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.0GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.0GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.0GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.0GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.0GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh


# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.002GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.002GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh;
# fix the lenpen code, and relaunch. Now 
# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.01GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.01GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.01GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.01GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.01GRPO_INSTRUCT/global_step_260" IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh

# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_True_belief_prompting_False_is_mem1_False_belief_len_pen_0GRPO_INSTRUCT/global_step_260" SINGLE_CTX=True MULTI_MSG=False IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=16 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_True_belief_prompting_False_is_mem1_False_belief_len_pen_0GRPO_INSTRUCT/global_step_260" SINGLE_CTX=True MULTI_MSG=False IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=8 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_True_belief_prompting_False_is_mem1_False_belief_len_pen_0GRPO_INSTRUCT/global_step_260" SINGLE_CTX=True MULTI_MSG=False IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=4 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_True_belief_prompting_False_is_mem1_False_belief_len_pen_0GRPO_INSTRUCT/global_step_260" SINGLE_CTX=True MULTI_MSG=False IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=2 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh; CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_True_belief_prompting_False_is_mem1_False_belief_len_pen_0GRPO_INSTRUCT/global_step_260" SINGLE_CTX=True MULTI_MSG=False IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh

# CKPT="qwen2.5-7b-instruct_16sfr_seed1_sc_True_belief_prompting_False_is_mem1_False_belief_len_pen_0GRPO_INSTRUCT/global_step_260" SINGLE_CTX=True MULTI_MSG=False IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=1.0 NUM_OBJECTIVES=1 bash examples/grpo_trainer/run_nqhotpotqa_inference.sh




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
    actor_rollout_ref.rollout.val_kwargs.temperature=0.01 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens \
    actor_rollout_ref.rollout.single_context=$SINGLE_CTX \
    actor_rollout_ref.rollout.belief_multiple_messages=$MULTI_MSG \
    actor_rollout_ref.rollout.instruct=$INSTRUCT \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.actor.single_batch=True \
    algorithm.use_kl_in_reward=False \
    env.env_name=nqhotpotqa \
    env.seed=$SEED \
    env.max_steps=$MAX_STEPS \
    env.non_terminal_penalty=0.0 \
    env.rollout.n=$group_size \
    +env.force_full_step_len=$FORCE_FULL \
    +env.split=test \
    +env.num_objectives=$NUM_OBJECTIVES \
    +env.max_obs_length=1000 \
    +env.topk=3 \
    +env.search_url="http://127.0.0.1:8013/retrieve" \
    +env.is_mem1=$IS_MEM1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=nqhotpotqa_grpo_qwen2.5-7b-${MODEL_DESC}_t_${TEMPERATURE}_${train_data_size}sfr_seed${SEED}_sc_${SINGLE_CTX}_belief_prompting_${MULTI_MSG}${DEBUG}_ckpt_${CKPT}_objectives_${NUM_OBJECTIVES}_inference3 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=500 \
    trainer.only_gen_once=True \
    trainer.resume_from_path=/nas/ucb/jbjorner3/dev/optimal-explorer-dev/verl-agent/checkpoints/verl_agent_alfworld/nqhotpotqa_grpo_${CKPT} \
    trainer.val_before_train=False $@
# /nas/ucb/jbjorner3/dev/optimal-explorer-dev/verl-agent/checkpoints/verl_agent_alfworld/nqhotpotqa_grpo_
# /nas/ucb/dayan/optimal-explorer-dev/verl-agent/checkpoints/verl_agent_alfworld/nqhotpotqa_grpo_qwen2.5-7b-instruct_16sfr_seed1_sc_False_belief_prompting_True_is_mem1_False_belief_len_pen_0.0GRPO_INSTRUCT/global_step_260