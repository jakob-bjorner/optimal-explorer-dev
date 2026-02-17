# optimal-explorer-dev
Developing code for ABBEL. During development we used the code name optimal-explorer hence the repository name.

This repository is a research artifact meant to be used for replicating or inspecting the details of our results in ABBEL.

We utilized submodules in cases where we thought repositories might benefit from future upstream changes
```bash
git clone --recurse-submodules https://github.com/jakob-bjorner/optimal-explorer-dev.git
```

For replicating results with frontier models see notebooks/neurips_workshops_frontier_models.ipynb

For results with RL see the verl-agent repo and follow setup instructions in the readme there.
For replicating results in the combination lock setting
See verl-agent/examples/grpo_trainer/run_combolock.sh and verl-agent/examples/grpo_trainer/run_combolock_inference.sh

For replicating results in multi-objective question answering, and follow the dataset creation setup instructions in the MEM1 submodule.
See verl-agent/examples/grpo_trainer/run_nqhotpotqa.sh and verl-agent/examples/grpo_trainer/run_nqhotpotqa_inference.sh


( if the next line doesn't work citing some CUDA_HOME not set error, switch to conda and run everything again with this line inserted. conda install -c conda-forge cudatoolkit-dev)

cd optimal-explorer-dev/
uv venv --python 3.12.0
source .venv/bin/activate
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install packaging wheel
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
cd verl-agent/
uv pip install -e .
uv pip install vllm==0.8.5

cd .. # (optimal-explorer-dev)
uv pip install -e .
uv pip install debugpy
uv pip install peft==0.17.0 # this is a sad dependancy
wandb login


cd /root
uv venv retriever --python 3.10
source retriever/bin/activate
uv pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
uv pip install transformers datasets pyserini
uv pip install faiss-gpu-cu12==1.8.0.2
uv pip install uvicorn fastapi

cd optimal-explorer-dev/MEM1/Mem1
bash train/retrieval_launch.sh

`control+z` 
`bg`

source .venv/bin/activate
cd optimal-explorer-dev/verl-agent
DEBUG=GRPO_INSTRUCT_3 LENPEN=0.1 bash examples/grpo_trainer/run_nqhotpotqa.sh





uv venv --python 3.12.0
source .venv/bin/activate
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install packaging wheel
uv pip install flash-attn==2.7.4.post1 --no-build-isolation
cd optimal-explorer-dev/verl-agent/
uv pip install -e .
uv pip install vllm==0.8.5
cd optimal-explorer-dev
uv pip install -e .
uv pip install debugpy
wandb login

export OPENROUTER_API_KEY=YOUR_KEY
B=7 train_data_size=16 GRADE_BELIEF=2.0 DEBUG=combolock_vs_r1_new_parse4 bash examples/grpo_trainer/run_combolock.sh


For replicating results from ColabBench, 

Ensure you have an up to date environment for running the VLLM server. which should be seperate from the environment used for running our code.
uv venv vllm_runner --python 3.12 --seed
source vllm_runner/bin/activate
uv pip install vllm --torch-backend=auto

Ensure all the backend data from colabbench is initialized.
see readme for sweet_RL for installing geckodriver, and getting things up.
wget https://github.com/mozilla/geckodriver/releases/download/v0.35.0/geckodriver-v0.35.0-linux64.tar.gz
tar -xvzf geckodriver-v0.35.0-linux64.tar.gz
mkdir ~/bin/
mv geckodriver ~/bin/
echo "export PATH=$PATH:~/bin" >> ~/.bashrc
source ~/.bashrc
geckodriver --version

activate verl-agent environment
cd sweet_rl
uv pip install -e .
// may need to use hf download instead.
huggingface-cli download --repo-type dataset facebook/collaborative_agent_bench backend_tasks/train.jsonl backend_tasks/test.jsonl colbench_code_offline_15k_llama8b.jsonl --local-dir data

```bash
git submodule update --init --recursive
```


```bash
pip install -e .
```

## Running combolock environment experiment
Note: the mess is caused because we save all checkpoints and eval on them afterwards. This was convenient when programming the loop, and when changing up the eval settings for combolock's environment, and has remained the way we eval, to the detriment of running experiments, and the files this method produces as biproducts as a default.
```sh
B=7 train_data_size=16 SEED=3 DEBUG=combolock_abbel_3 bash examples/grpo_trainer/run_combolock.sh; STEP_RESUME=20 DEBUG=combolock_abbel_3 CKPT="_seed3_sc_False_belief_prompting_True_BG_0.0combolock_abbel_3/global_step_20" train_data_size=128 GRADE_BELIEF=0.0 MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh;  DEBUG=combolock_abbel_3 STEP_RESUME=40 CKPT="_seed3_sc_False_belief_prompting_True_BG_0.0combolock_abbel_3/global_step_40" train_data_size=128 GRADE_BELIEF=0.0 MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh;  DEBUG=combolock_abbel_3 STEP_RESUME=60 CKPT="_seed3_sc_False_belief_prompting_True_BG_0.0combolock_abbel_3/global_step_60" train_data_size=128 GRADE_BELIEF=0.0 MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh;  DEBUG=combolock_abbel_3 STEP_RESUME=80 CKPT="_seed3_sc_False_belief_prompting_True_BG_0.0combolock_abbel_3/global_step_80" train_data_size=128 GRADE_BELIEF=0.0 MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh;  DEBUG=combolock_abbel_3 STEP_RESUME=100 CKPT="_seed3_sc_False_belief_prompting_True_BG_0.0combolock_abbel_3/global_step_100" train_data_size=128 GRADE_BELIEF=0.0 MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh;  DEBUG=combolock_abbel_3 STEP_RESUME=120 CKPT="_seed3_sc_False_belief_prompting_True_BG_0.0combolock_abbel_3/global_step_120" train_data_size=128 GRADE_BELIEF=0.0 MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh; DEBUG=combolock_abbel_3 STEP_RESUME=140 CKPT="_seed3_sc_False_belief_prompting_True_BG_0.0combolock_abbel_3/global_step_140" train_data_size=128 GRADE_BELIEF=0.0 MAX_ATTEMPTS=16 VOCAB='qawsedrftgyhujik' bash examples/grpo_trainer/run_combolock_inference.sh
```
