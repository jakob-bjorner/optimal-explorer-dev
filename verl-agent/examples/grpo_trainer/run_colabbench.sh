set -x
ENGINE=${ENGINE:-vllm}
# ENGINE=vllm
ASYNC=${ASYNC:-False}
# DSET=${DSET:-interaction_base_base}
SEED=${SEED:-1}
DEBUG=${DEBUG:-}
SINGLE_CTX=${SINGLE_CTX:-False}
MULTI_MSG=${MULTI_MSG:-True}
MAX_ATTEMPTS=${MAX_ATTEMPTS:-10}
MAX_RES_LENGTH=${MAX_RES_LENGTH:-1000}
MAX_STEPS=$(($MAX_ATTEMPTS * 2 - 1))
if [ $SINGLE_CTX == True ]; then
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
    MAX_STEPS=$MAX_ATTEMPTS
fi
if [ $INSTRUCT == True ]; then
    MODEL_DESC=instruct
    MODEL_PATH=qwen/qwen2.5-7b-instruct
else
    MODEL_DESC=""
    MODEL_PATH=qwen/qwen2.5-7b
fi

LENPEN=${LENPEN:-0}
FORCE_FULL=${FORCE_FULL:-False}
GRADE_BELIEF=${GRADE_BELIEF:-0.0}
GRADE_BELIEF_TYPE=${GRADE_BELIEF_TYPE:-0}
BG_CEIL_RWD=${BG_CEIL_RWD:--0.8}
FULL_HIST_BELIEF=${FULL_HIST_BELIEF:-False}
if [ -z "${CKPT+x}" ]; then
    RESUME_PATH=null
else
    RESUME_PATH=checkpoints/verl_agent_alfworld/colabbench_grpo_${CKPT}
fi
# 0 corresponds to belief grading with full reconstruction, and 1 with reconstructing only observations.


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

# SINGLE_CTX=True MULTI_MSG=False bash examples/grpo_trainer/run_combolock.sh; SINGLE_CTX=True MULTI_MSG=True bash examples/grpo_trainer/run_combolock.sh

### ppo setup, also need to cahnge adv_estimator to grpo.
# critic.optim.lr=1e-5 \
# critic.model.use_remove_padding=True \
# critic.optim.lr_warmup_steps_ratio=0.05 \
# critic.model.path=$MODEL_PATH \
# critic.model.enable_gradient_checkpointing=True \
# critic.ppo_micro_batch_size_per_gpu=2 \
# critic.model.fsdp_config.param_offload=True \
# critic.model.fsdp_config.optimizer_offload=True \


# before launching need to run 
# source ../.venv/bin/activate
# meta-llama/Llama-3.1-8B-Instruct
# openai/gpt-oss-20b
# google/gemma-3-27b-it
# CUDA_VISIBLE_DEVICES=4,5 vllm serve google/gemma-3-27b-it --max-model-len 16384 --tensor-parallel-size 2 --gpu-memory-utilization=0.85 --max-num-seqs 16 --port 8000 --enforce-eager --trust-remote-code
# this will use gpus 4, and 5 for hosting the RAG endpoint to conduct search on.
# Then use GPU 0,1,2,3 for the training and rollout.

# with base model and mem1 enables with ppo (REMEMBER TO CHANGE TO PPO CRITIC WHEN LAUNCHING!!!)
# DEBUG=MEM1GRPO32 IS_MEM1=True INSTRUCT=False bash examples/grpo_trainer/run_colabbench.sh
# DEBUG=MEM1GRPO64 train_data_size=64 IS_MEM1=True INSTRUCT=False bash examples/grpo_trainer/run_nqhotpotqa.sh

# DEBUG=GRPO_INSTRUCT_G_ABBEL bash examples/grpo_trainer/run_colabbench.sh
# DEBUG=GRPO_INSTRUCT_G_VANILLA SINGLE_CTX=True MULTI_MSG=False bash examples/grpo_trainer/run_colabbench.sh
# DEBUG=GRPO_INSTRUCT IS_MEM1=True bash examples/grpo_trainer/run_colabbench.sh 
# DEBUG=GRPO_INSTRUCT LENPEN=0.0002 bash examples/grpo_trainer/run_colabbench.sh
# DEBUG=GRPO_INSTRUCT_G_BG GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh
# DEBUG=GRPO_INSTRUCT_G_BG_2048 MAX_RES_LENGTH=2048 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh
# DEBUG=GRPO_INSTRUCT_G_BG GRADE_BELIEF_TYPE=1 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh
# I launched the below changing the ceiling to -0.8.
# DEBUG=GRPO_INSTRUCT_G_BG_LP_CEILn0_8 GRADE_BELIEF=1.0 LENPEN=0.01 bash examples/grpo_trainer/run_colabbench.sh
# DEBUG=GRPO_INSTRUCT_G_ABBEL_RESUME_FROM_PROB_OBS_50 CKPT="qwen2.5-7b-instruct_16_seed1_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_1.0_1GRPO_INSTRUCT_G_BG_NOCRASHPLS/global_step_50" bash examples/grpo_trainer/run_colabbench.sh
# DEBUG=GRPO_INSTRUCT_G_BG_LOG_PROB_OBS_2 SEED=2 GRADE_BELIEF_TYPE=1 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh

# DEBUG=GRPO_INSTRUCT_G_BG_2048_MAX_ATT_3 MAX_ATTEMPTS=3 MAX_PROMPT_LEN=5186 MAX_RES_LENGTH=2048 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh
# DEBUG=GRPO_INSTRUCT_G_VANILLA_MAX_ATT_3 SINGLE_CTX=True MULTI_MSG=False MAX_ATTEMPTS=3 bash examples/grpo_trainer/run_colabbench.sh
# DEBUG=GRPO_INSTRUCT_G_BG_2048_MAX_ATT_3_LP LENPEN=0.01 MAX_ATTEMPTS=3 MAX_PROMPT_LEN=5186 MAX_RES_LENGTH=2048 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh; DEBUG=GRPO_INSTRUCT_G_BG_MAX_ATT_3 MAX_ATTEMPTS=3 bash examples/grpo_trainer/run_colabbench.sh
# doubling batch size, and adding two to number of steps because vanilla isn't learning to ask more questions.
# DEBUG=GRPO_INSTRUCT_G_VANILLA_MAX_ATT_5 SINGLE_CTX=True MULTI_MSG=False MAX_ATTEMPTS=5 train_data_size=32 bash examples/grpo_trainer/run_colabbench.sh; DEBUG=GRPO_INSTRUCT_G_BG_2048_MAX_ATT_5 MAX_ATTEMPTS=5 train_data_size=32 MAX_PROMPT_LEN=5186 MAX_RES_LENGTH=2048 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh; DEBUG=GRPO_INSTRUCT_G_BG_2048_MAX_ATT_5_LP LENPEN=0.01 MAX_ATTEMPTS=5 train_data_size=32 MAX_PROMPT_LEN=5186 MAX_RES_LENGTH=2048 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh; DEBUG=GRPO_INSTRUCT_G_BG_MAX_ATT_5 MAX_ATTEMPTS=5 train_data_size=32 bash examples/grpo_trainer/run_colabbench.sh
# DEBUG=GRPO_INSTRUCT_G_BG_200_LP LENPEN=0.01 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh; 
# CKPT="qwen2.5-7b-instruct_16_seed1_sc_False_belief_promp_True_is_mem1_False_belief_lp_0.01_bg_1.0_0GRPO_INSTRUCT_G_BG_200_LP/global_step_50" DEBUG=GRPO_INSTRUCT_G_BG_200_LP STEP_RESUME=50 INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed1_sc_False_belief_promp_True_is_mem1_False_belief_lp_0.01_bg_1.0_0GRPO_INSTRUCT_G_BG_200_LP/global_step_100" DEBUG=GRPO_INSTRUCT_G_BG_200_LP STEP_RESUME=100 INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh
# DEBUG=GRPO_INSTRUCT_G_VANILLA_2 SINGLE_CTX=True MULTI_MSG=False SEED=2 bash examples/grpo_trainer/run_colabbench.sh
# CKPT="qwen2.5-7b-instruct_16_seed2_sc_True_belief_promp_False_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_VANILLA_2/global_step_50" DEBUG=REDO STEP_RESUME=50 SINGLE_CTX=True MULTI_MSG=False IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed2_sc_True_belief_promp_False_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_VANILLA_2/global_step_100" DEBUG=REDO STEP_RESUME=100 SINGLE_CTX=True MULTI_MSG=False IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; 
# DEBUG=GRPO_INSTRUCT_G_BG_400_LP LENPEN=0.01 GRADE_BELIEF=0.1 bash examples/grpo_trainer/run_colabbench.sh
# CKPT="qwen2.5-7b-instruct_16_seed1_sc_False_belief_promp_True_is_mem1_False_belief_lp_0.01_bg_0.1_0GRPO_INSTRUCT_G_BG_400_LP/global_step_50" DEBUG=GRPO_INSTRUCT_G_BG_400_LP STEP_RESUME=50 INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed1_sc_False_belief_promp_True_is_mem1_False_belief_lp_0.01_bg_0.1_0GRPO_INSTRUCT_G_BG_400_LP/global_step_100" DEBUG=GRPO_INSTRUCT_G_BG_400_LP STEP_RESUME=100 INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh
# DEBUG=GRPO_INSTRUCT_G_BG_400_LP_2 SEED=2 LENPEN=0.01 GRADE_BELIEF=0.5 bash examples/grpo_trainer/run_colabbench.sh; 
# colabbench_grpo_qwen2.5-7b-instruct_16_seed2_sc_False_belief_promp_True_is_mem1_False_belief_lp_0.01_bg_0.5_0GRPO_INSTRUCT_G_BG_400_LP_2
# CKPT="qwen2.5-7b-instruct_16_seed2_sc_False_belief_promp_True_is_mem1_False_belief_lp_0.01_bg_0.5_0GRPO_INSTRUCT_G_BG_400_LP_2/global_step_50" DEBUG=GRPO_INSTRUCT_G_BG_400_LP_2 STEP_RESUME=50 INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed2_sc_False_belief_promp_True_is_mem1_False_belief_lp_0.01_bg_0.5_0GRPO_INSTRUCT_G_BG_400_LP_2/global_step_100" DEBUG=GRPO_INSTRUCT_G_BG_400_LP_2 STEP_RESUME=100 INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh
# DEBUG=GRPO_INSTRUCT_G_VANILLA_1 SINGLE_CTX=True MULTI_MSG=False SEED=1 bash examples/grpo_trainer/run_colabbench.sh
# DEBUG=GRPO_INSTRUCT_G_VANILLA_3 SINGLE_CTX=True MULTI_MSG=False SEED=3 bash examples/grpo_trainer/run_colabbench.sh; CKPT="qwen2.5-7b-instruct_16_seed3_sc_True_belief_promp_False_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_VANILLA_3/global_step_50" STEP_RESUME=50 SINGLE_CTX=True MULTI_MSG=False IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed3_sc_True_belief_promp_False_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_VANILLA_3/global_step_100" STEP_RESUME=100 SINGLE_CTX=True MULTI_MSG=False IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh
# we want to launch new seeds of the runs.
# DEBUG=GRPO_INSTRUCT_G_ABBEL_1 SEED=1 bash examples/grpo_trainer/run_colabbench.sh; DEBUG=GRPO_INSTRUCT_G_ABBEL_1 CKPT="qwen2.5-7b-instruct_16_seed1_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_ABBEL_1/global_step_50" STEP_RESUME=50 IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; DEBUG=GRPO_INSTRUCT_G_ABBEL_1 CKPT="qwen2.5-7b-instruct_16_seed1_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_ABBEL_1/global_step_100" STEP_RESUME=100 IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh
# FULL_HIST_BELIEF=True DEBUG=GRPO_INSTRUCT_G_FC_BG_PROXY bash examples/grpo_trainer/run_colabbench.sh
# CKPT="qwen2.5-7b-instruct_16_seed1_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_FC_BG_PROXY_THINK/global_step_100" MAX_PROMPT_LEN=5186 MAX_RES_LENGTH=2048 STEP_RESUME=100 FULL_HIST_BELIEF=True DEBUG=GRPO_INSTRUCT_G_FC_BG_PROXY_THINK INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; FULL_HIST_BELIEF=True DEBUG=GRPO_INSTRUCT_G_FC_BG_PROXY_THINK SEED=2 MAX_PROMPT_LEN=5186 MAX_RES_LENGTH=2048 bash examples/grpo_trainer/run_colabbench.sh; CKPT="qwen2.5-7b-instruct_16_seed2_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_FC_BG_PROXY_THINK/global_step_50" MAX_PROMPT_LEN=5186 MAX_RES_LENGTH=2048 STEP_RESUME=50 FULL_HIST_BELIEF=True DEBUG=GRPO_INSTRUCT_G_FC_BG_PROXY_THINK INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed2_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_FC_BG_PROXY_THINK/global_step_100" MAX_PROMPT_LEN=5186 MAX_RES_LENGTH=2048 STEP_RESUME=100 FULL_HIST_BELIEF=True DEBUG=GRPO_INSTRUCT_G_FC_BG_PROXY_THINK INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh
# DEBUG=GRPO_INSTRUCT_G_BG_SSR_debug_3 GRADE_BELIEF_TYPE=-1 MAX_ATTEMPTS=3 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh
# SSR = single step reward
# qwen2.5-7b-instruct_16_seed1_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_1.0_-1GRPO_INSTRUCT_G_BG_SSR
# DEBUG=GRPO_INSTRUCT_G_BG_SSR GRADE_BELIEF_TYPE=-1 SEED=2 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh; CKPT="qwen2.5-7b-instruct_16_seed2_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_1.0_-1GRPO_INSTRUCT_G_BG_SSR/global_step_50" STEP_RESUME=50 DEBUG=GRPO_INSTRUCT_G_BG_SSR INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed2_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_1.0_-1GRPO_INSTRUCT_G_BG_SSR/global_step_100" STEP_RESUME=100 DEBUG=GRPO_INSTRUCT_G_BG_SSR INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh
# DEBUG=GRPO_INSTRUCT_G_ABBEL_FR_1 SEED=1 bash examples/grpo_trainer/run_colabbench.sh; CKPT="qwen2.5-7b-instruct_16_seed1_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_ABBEL_FR_1/global_step_50" STEP_RESUME=50 DEBUG=GRPO_INSTRUCT_G_ABBEL_FR_1 INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed1_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_ABBEL_FR_1/global_step_100" STEP_RESUME=100 DEBUG=GRPO_INSTRUCT_G_ABBEL_FR_1 INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh

# DEBUG=GRPO_INSTRUCT_G_ABBEL_FR_2 SEED=2 bash examples/grpo_trainer/run_colabbench.sh; CKPT="qwen2.5-7b-instruct_16_seed2_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_ABBEL_FR_2/global_step_50" STEP_RESUME=50 DEBUG=GRPO_INSTRUCT_G_ABBEL_FR_2 INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed2_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_ABBEL_FR_2/global_step_100" STEP_RESUME=100 DEBUG=GRPO_INSTRUCT_G_ABBEL_FR_2 INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh
# DEBUG=GRPO_INSTRUCT_G_ABBEL_FR_4 SEED=4 bash examples/grpo_trainer/run_colabbench.sh; CKPT="qwen2.5-7b-instruct_16_seed4_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_ABBEL_FR_4/global_step_50" STEP_RESUME=50 DEBUG=GRPO_INSTRUCT_G_ABBEL_FR_4 INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed4_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_ABBEL_FR_4/global_step_100" STEP_RESUME=100 DEBUG=GRPO_INSTRUCT_G_ABBEL_FR_4 INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; DEBUG=GRPO_INSTRUCT_G_VANILLA_4 SINGLE_CTX=True MULTI_MSG=False SEED=4 bash examples/grpo_trainer/run_colabbench.sh; CKPT="qwen2.5-7b-instruct_16_seed4_sc_True_belief_promp_False_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_VANILLA_4/global_step_50" STEP_RESUME=50 SINGLE_CTX=True MULTI_MSG=False IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed4_sc_True_belief_promp_False_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_VANILLA_4/global_step_100" STEP_RESUME=100 SINGLE_CTX=True MULTI_MSG=False IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh
# DEBUG=GRPO_INSTRUCT_G_VANILLA_5 SINGLE_CTX=True MULTI_MSG=False SEED=5 bash examples/grpo_trainer/run_colabbench.sh; CKPT="qwen2.5-7b-instruct_16_seed5_sc_True_belief_promp_False_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_VANILLA_5/global_step_50" STEP_RESUME=50 SINGLE_CTX=True MULTI_MSG=False IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed5_sc_True_belief_promp_False_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_VANILLA_5/global_step_100" STEP_RESUME=100 SINGLE_CTX=True MULTI_MSG=False IS_MEM1=False INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh
# DEBUG=GRPO_INSTRUCT_G_BG_REF_RECONSTRUCT BG_CEIL_RWD=-0.2 GRADE_BELIEF_TYPE=3 SEED=1 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh; CKPT="qwen2.5-7b-instruct_16_seed1_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_1.0_3GRPO_INSTRUCT_G_BG_REF_RECONSTRUCT/global_step_50" STEP_RESUME=50 DEBUG=GRPO_INSTRUCT_G_BG_REF_RECONSTRUCT INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed1_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_1.0_3GRPO_INSTRUCT_G_BG_REF_RECONSTRUCT/global_step_100" STEP_RESUME=100 DEBUG=GRPO_INSTRUCT_G_BG_REF_RECONSTRUCT INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; DEBUG=GRPO_INSTRUCT_G_BG_REF_RECONSTRUCT BG_CEIL_RWD=-0.2 GRADE_BELIEF_TYPE=3 SEED=2 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh; CKPT="qwen2.5-7b-instruct_16_seed2_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_1.0_3GRPO_INSTRUCT_G_BG_REF_RECONSTRUCT/global_step_50" STEP_RESUME=50 DEBUG=GRPO_INSTRUCT_G_BG_REF_RECONSTRUCT INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed2_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_1.0_3GRPO_INSTRUCT_G_BG_REF_RECONSTRUCT/global_step_100" STEP_RESUME=100 DEBUG=GRPO_INSTRUCT_G_BG_REF_RECONSTRUCT INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh

# DEBUG=GRPO_INSTRUCT_G_ABBEL_FR_5 SEED=5 bash examples/grpo_trainer/run_colabbench.sh; CKPT="qwen2.5-7b-instruct_16_seed5_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_ABBEL_FR_5/global_step_50" STEP_RESUME=50 DEBUG=GRPO_INSTRUCT_G_ABBEL_FR_5 INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed5_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_0.0_0GRPO_INSTRUCT_G_ABBEL_FR_5/global_step_100" STEP_RESUME=100 DEBUG=GRPO_INSTRUCT_G_ABBEL_FR_5 INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh ; DEBUG=GRPO_INSTRUCT_G_BG_SSR GRADE_BELIEF_TYPE=-1 SEED=3 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh; CKPT="qwen2.5-7b-instruct_16_seed3_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_1.0_-1GRPO_INSTRUCT_G_BG_SSR/global_step_50" STEP_RESUME=50 DEBUG=GRPO_INSTRUCT_G_BG_SSR INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed3_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_1.0_-1GRPO_INSTRUCT_G_BG_SSR/global_step_100" STEP_RESUME=100 DEBUG=GRPO_INSTRUCT_G_BG_SSR INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh

# DEBUG=GRPO_INSTRUCT_G_BG_SSR GRADE_BELIEF_TYPE=-1 SEED=4 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh; CKPT="qwen2.5-7b-instruct_16_seed4_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_1.0_-1GRPO_INSTRUCT_G_BG_SSR/global_step_50" STEP_RESUME=50 DEBUG=GRPO_INSTRUCT_G_BG_SSR INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed4_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_1.0_-1GRPO_INSTRUCT_G_BG_SSR/global_step_100" STEP_RESUME=100 DEBUG=GRPO_INSTRUCT_G_BG_SSR INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh ; DEBUG=GRPO_INSTRUCT_G_BG_SSR GRADE_BELIEF_TYPE=-1 SEED=5 GRADE_BELIEF=1.0 bash examples/grpo_trainer/run_colabbench.sh; CKPT="qwen2.5-7b-instruct_16_seed5_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_1.0_-1GRPO_INSTRUCT_G_BG_SSR/global_step_50" STEP_RESUME=50 DEBUG=GRPO_INSTRUCT_G_BG_SSR INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh; CKPT="qwen2.5-7b-instruct_16_seed5_sc_False_belief_promp_True_is_mem1_False_belief_lp_0_bg_1.0_-1GRPO_INSTRUCT_G_BG_SSR/global_step_100" STEP_RESUME=100 DEBUG=GRPO_INSTRUCT_G_BG_SSR INSTRUCT=True train_data_size=256 TEMPERATURE=0.01 bash examples/grpo_trainer/run_colabbench_inference.sh

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RES_LENGTH \
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
    env.env_name=colabbench \
    env.seed=$SEED \
    env.max_steps=$MAX_STEPS \
    trainer.post_normalization_length_penalty=$LENPEN \
    env.non_terminal_penalty=0.0 \
    env.rollout.n=$group_size \
    env.belief_length_penalty=0.0 \
    +env.split=train \
    +env.full_history_belief=$FULL_HIST_BELIEF \
    +env.is_mem1=$IS_MEM1 \
    +env.hostname=$HOSTNAME \
    +env.port=8000 \
    +env.model_id=google/gemma-3-27b-it \
    +env.task_type=backend \
    +env.max_attempts=$MAX_ATTEMPTS \
    trainer.belief_state_grading=$GRADE_BELIEF \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.belief_state_grading_type=$GRADE_BELIEF_TYPE \
    +trainer.ceiling_belief_grading_reward=$BG_CEIL_RWD \
    trainer.project_name='verl_agent_alfworld' \
    trainer.experiment_name=colabbench_grpo_qwen2.5-7b-${MODEL_DESC}_16_seed${SEED}_sc_${SINGLE_CTX}_belief_promp_${MULTI_MSG}_is_mem1_${IS_MEM1}_belief_lp_${LENPEN}_bg_${GRADE_BELIEF}_${GRADE_BELIEF_TYPE}${DEBUG} \
    trainer.n_gpus_per_node=4 \
    trainer.resume_from_path=$RESUME_PATH \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.total_epochs=100 \
    trainer.val_before_train=False $@
