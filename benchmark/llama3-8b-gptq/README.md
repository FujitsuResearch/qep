# Llama-3-8B GPTQ Benchmark

GPTQ benchmark for [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) using OneComp v0.3.7.

All combinations of `bits × group_size` are run in a single pass, sharing calibration data accumulation across quantizers for efficiency.

## Benchmark Configuration

| Parameter | Values |
|---|---|
| bits | 4, 3 |
| group_size | 128, per-channel |
| symmetric | true |
| num_calibration_samples | 1024 |
| calibration_strategy | drop_rand |
| max_length | 2048 |

This produces **4 quantizers** (2 bits × 2 group sizes) per run.

### Evaluation

- Perplexity (WikiText-2)
- Accuracy (lm-eval-harness)

Both are computed for the original (unquantized) model and all quantized models.

## Usage

Requires [Hydra](https://hydra.cc/) (see [benchmark/README.md](../README.md) for installation).

Specify the path to the model via `model_path`:

```bash
python quant_benchmark.py model_path=/path/to/Meta-Llama-3-8B
```

### Hydra Overrides

You can override any parameter from the command line:

```bash
# Run only 4-bit
python quant_benchmark.py model_path=/path/to/model 'gptq.bits=[4]'

# Change calibration samples
python quant_benchmark.py model_path=/path/to/model num_calibration_samples=512
```

## Results

### Perplexity (WikiText-2, ↓ lower is better)

| Model | bits | group_size | PPL |
|---|---|---|---|
| Original | — | — | 6.14 |
| GPTQ | 4 | 128 | 12.66 |
| GPTQ | 4 | per-channel | 665.94 |
| GPTQ | 3 | 128 | 45.22 |
| GPTQ | 3 | per-channel | 1721.06 |

### Accuracy (0-shot, ↑ higher is better)

Values are `acc_norm` where available, `acc` otherwise (winogrande).

| Model | bits | group_size | ARC-c | ARC-e | PIQA | WinoGrande |
|---|---|---|---|---|---|---|
| Original | — | — | 0.5401 | 0.7761 | 0.8063 | 0.7380 |
| GPTQ | 4 | 128 | 0.5026 | 0.7710 | 0.7922 | 0.7206 |
| GPTQ | 4 | per-channel | 0.3089 | 0.5076 | 0.6861 | 0.6298 |
| GPTQ | 3 | 128 | 0.3097 | 0.4886 | 0.6610 | 0.6259 |
| GPTQ | 3 | per-channel | 0.2167 | 0.2862 | 0.5419 | 0.5004 |

### Quantization Time

| Model | bits | group_size | Time (s) |
|---|---|---|---|
| GPTQ | 4 | 128 | 276.8 |
| GPTQ | 4 | per-channel | 268.9 |
| GPTQ | 3 | 128 | 273.8 |
| GPTQ | 3 | per-channel | 268.4 |

Total elapsed time (including calibration data preparation): 3697.5 s (~62 min).

## Environment

- GPU: NVIDIA B200 × 1

## Notes

This benchmark internally uses `Runner.quantize_with_calibration_chunked`, which can run multiple quantizers simultaneously without QEP. However, it requires the entire model to fit on the GPU and involves redundant forward passes. Addressing these limitations is future work.
