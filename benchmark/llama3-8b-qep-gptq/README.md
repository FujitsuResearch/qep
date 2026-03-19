# Llama-3-8B QEP+GPTQ Benchmark

QEP (Quantization Error Propagation) + GPTQ benchmark for [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) using OneComp v0.3.7.

QEP only supports a single quantizer per run, so each `bits × group_size` combination is launched as a separate SLURM array task.

## Benchmark Configuration

| Parameter | Values |
|---|---|
| bits | 4, 3 |
| group_size | 128, per-channel |
| symmetric | true |
| num_calibration_samples | 1024 |
| calibration_strategy | drop_rand |
| max_length | 2048 |

QEP parameters:

| Parameter | Values |
|---|---|
| percdamp | 0.01 |
| perccorr | 0.5 |

This produces **4 array tasks** (2 bits × 2 group sizes), each with QEP enabled.

| task_id | bits | group_size |
|---|---|---|
| 0 | 4 | 128 |
| 1 | 4 | per-channel |
| 2 | 3 | 128 |
| 3 | 3 | per-channel |

### Evaluation

- Perplexity (WikiText-2)
- Accuracy (lm-eval-harness)

Each task evaluates its own quantized model. Original model metrics are computed only in task 0.

## Usage

Requires [Hydra](https://hydra.cc/) (see [benchmark/README.md](../README.md) for installation).

Specify the path to the model via `model_path` and the task via `task_id`:

```bash
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B task_id=0
```

### Hydra Overrides

You can override any parameter from the command line:

```bash
# Change QEP correction percentage
python quant_benchmark.py model_path=/path/to/model task_id=0 qep.perccorr=0.3

# Change calibration samples
python quant_benchmark.py model_path=/path/to/model task_id=0 num_calibration_samples=512
```

## Results

### Perplexity (WikiText-2, ↓ lower is better)

| Model | bits | group_size | PPL |
|---|---|---|---|
| Original | — | — | 6.14 |
| QEP+GPTQ | 4 | 128 | 6.66 |
| QEP+GPTQ | 4 | per-channel | 7.67 |
| QEP+GPTQ | 3 | 128 | 8.95 |
| QEP+GPTQ | 3 | per-channel | 17.93 |

### Accuracy (0-shot, ↑ higher is better)

Values are `acc_norm` where available, `acc` otherwise (winogrande).

| Model | bits | group_size | ARC-c | ARC-e | PIQA | WinoGrande |
|---|---|---|---|---|---|---|
| Original | — | — | 0.5401 | 0.7761 | 0.8063 | 0.7380 |
| QEP+GPTQ | 4 | 128 | 0.5265 | 0.7942 | 0.7916 | 0.7293 |
| QEP+GPTQ | 4 | per-channel | 0.4957 | 0.7542 | 0.7758 | 0.7269 |
| QEP+GPTQ | 3 | 128 | 0.4352 | 0.6498 | 0.7546 | 0.6946 |
| QEP+GPTQ | 3 | per-channel | 0.2688 | 0.4184 | 0.6806 | 0.6156 |

### Quantization Time

| Model | bits | group_size | Time (s) |
|---|---|---|---|
| QEP+GPTQ | 4 | 128 | 300.7 |
| QEP+GPTQ | 4 | per-channel | 297.7 |
| QEP+GPTQ | 3 | 128 | 272.7 |
| QEP+GPTQ | 3 | per-channel | 292.7 |

Each task's total elapsed time (including calibration data preparation and QEP error propagation) was approximately 2900–3300 s (~49–55 min).

## Environment

- GPU: NVIDIA B200 × 1
