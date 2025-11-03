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



```bash
git submodule update --init --recursive
```


```bash
pip install -e .
```