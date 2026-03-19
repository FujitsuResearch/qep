# benchmark

Benchmark scripts for OneComp.
Configuration is managed with [Hydra](https://hydra.cc/).

> **Note:** Hydra is not a dependency of OneComp and must be installed separately.

## Installing Hydra

```bash
pip install hydra-core
```

Verify the installation:

```bash
python -c "import hydra; print(hydra.__version__)"
```

## Benchmarks

| Directory | Description |
|---|---|
| [llama3-8b-gptq/](llama3-8b-gptq/) | GPTQ (4bit/3bit × gs128/per-channel) |
| [llama3-8b-qep-gptq/](llama3-8b-qep-gptq/) | QEP+GPTQ (4bit/3bit × gs128/per-channel) |
| [llama3-8b-various/](llama3-8b-various/) | Various quantizers with default parameters (no QEP) |

## Llama-3-8B: GPTQ vs QEP+GPTQ

Comparison of GPTQ with and without QEP (Quantization Error Propagation) on [Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) (OneComp v0.3.7).

### Perplexity (WikiText-2, ↓ lower is better)

| bits | group_size | GPTQ | QEP+GPTQ |
|---|---|---|---|
| — | — | 6.14 (original) | 6.14 (original) |
| 4 | 128 | 12.66 | **6.66** |
| 4 | per-channel | 665.94 | **7.67** |
| 3 | 128 | 45.22 | **8.95** |
| 3 | per-channel | 1721.06 | **17.93** |

### Accuracy (0-shot, ↑ higher is better)

Values are `acc_norm` where available, `acc` otherwise (winogrande).

| bits | group_size | Method | ARC-c | ARC-e | PIQA | WinoGrande |
|---|---|---|---|---|---|---|
| — | — | Original | 0.5401 | 0.7761 | 0.8063 | 0.7380 |
| 4 | 128 | GPTQ | 0.5026 | 0.7710 | 0.7922 | 0.7206 |
| 4 | 128 | **QEP+GPTQ** | **0.5265** | **0.7942** | **0.7916** | **0.7293** |
| 4 | per-channel | GPTQ | 0.3089 | 0.5076 | 0.6861 | 0.6298 |
| 4 | per-channel | **QEP+GPTQ** | **0.4957** | **0.7542** | **0.7758** | **0.7269** |
| 3 | 128 | GPTQ | 0.3097 | 0.4886 | 0.6610 | 0.6259 |
| 3 | 128 | **QEP+GPTQ** | **0.4352** | **0.6498** | **0.7546** | **0.6946** |
| 3 | per-channel | GPTQ | 0.2167 | 0.2862 | 0.5419 | 0.5004 |
| 3 | per-channel | **QEP+GPTQ** | **0.2688** | **0.4184** | **0.6806** | **0.6156** |

See [llama3-8b-gptq/](llama3-8b-gptq/) and [llama3-8b-qep-gptq/](llama3-8b-qep-gptq/) for full details.
