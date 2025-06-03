conda create -n env-v2 python==3.10
conda activate env-v2
cd verl
pip install -e ".[sglang]"
pip install flash-attn --no-build-isolation
pip install wandb IPython matplotlib debugpy