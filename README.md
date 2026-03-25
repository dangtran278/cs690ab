# KVBench

## Usage
1. Clone this repo
```
git clone https://github.com/dangtran278/KVBench.git
```
2.  Create environment
```
cd KVBench
conda create -n kvbench python=3.12 -y
conda activate kvbench
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## Model Download

## Evaluation
### How to run
Perplexity (4K context):
```
python scripts/run_ppl.py \
    --model "/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b" \
    --method kivi2 \
    --max_tokens 4096
```
Passkey (e.g., 8192 tokens):
```
python scripts/run_passkey.py \
    --model "/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b" \
    --method kvquant_nuq4_1p \
    --context_tokens 8192
```

### How to evaluate
Both tasks:
```
python scripts/run_matrix.py
```
Only PPL:
```
python scripts/run_matrix.py \
    --run_ppl \
    --methods fp16 kivi2 kvquant_nuq4_1p
```
Only passkey:
```
python scripts/run_matrix.py \
    --run_passkey \
    --passkey_contexts 2048 4096 8192 16384 32768
```
Custom decode length/seed/output:
```
python scripts/run_matrix.py \
    --decode_tokens 10 \
    --seed 0 \
    --output_dir logs
```