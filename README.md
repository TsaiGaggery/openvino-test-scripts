# OpenVINO Multi-Model Benchmark Script

## Overview

This revised benchmark script allows you to compare the performance of multiple LLM models across different hardware devices (CPU, GPU, NPU) using OpenVINO. It reads model configurations from `benchmark.json` and generates comprehensive comparison reports.

## 🚀 Quick Start - Single Model Test

**NEW!** Test any model quickly without editing JSON files:

```bash
python3 quick_benchmark.py OpenVINO/TinyLlama-1.1B-Chat-v1.0-int8-ov
```

This will:
- ✅ Test the model on CPU, GPU, and NPU
- ✅ Run **5 diverse test prompts**:
  1. "What is the capital of France?"
  2. "Explain quantum computing in simple terms."
  3. "Write a short poem about artificial intelligence."
  4. "What are the main differences between Python and JavaScript?"
  5. "Describe the process of photosynthesis."
- ✅ Generate `benchmark_report.html` with results
- ✅ **Does NOT modify your `benchmark.json`** - uses temporary config file

**More examples:**
```bash
python3 quick_benchmark.py OpenVINO/Phi-3.5-vision-instruct-int8-ov
python3 quick_benchmark.py OpenVINO/Mistral-7B-Instruct-v0.3-int8-ov
python3 quick_benchmark.py OpenVINO/Qwen2.5-1.5B-Instruct-int8-ov
```

### Custom Config File

You can also use a custom config file with the main benchmark script:

```bash
# Use custom config instead of benchmark.json
python3 benchmark_devices.py --config my_custom_config.json

# Or short form
python3 benchmark_devices.py -c my_custom_config.json
```

**No configuration needed!** Just provide the model ID and go! 🎯

---

## ❓ FAQ - Do I Need to Convert Models?

**Short answer: NO! ✅**

All models from the `OpenVINO/` namespace on HuggingFace are **already pre-converted** to OpenVINO IR format with INT4/INT8/FP16 quantization applied.

**What you get:**
- ✅ Pre-converted OpenVINO IR models (`.xml` + `.bin` files)
- ✅ Already quantized (INT4/INT8/FP16 - ready for NPU/GPU/CPU)
- ✅ Optimized with NNCF weight compression
- ✅ Compatible with OpenVINO 2024.2.0+

**You DO NOT need to:**
- ❌ Run `optimum-cli export openvino`
- ❌ Convert from PyTorch/ONNX
- ❌ Apply quantization yourself
- ❌ Use `nncf.compress_weights`

**Just download and use!** The scripts automatically use `snapshot_download()` to fetch ready-to-use models.

### If You Want to Convert Your Own Models

Only needed if you're using models **NOT** from the OpenVINO hub:

```bash
# Install optimum-cli
pip install optimum[openvino,nncf]

# Convert and quantize
optimum-cli export openvino \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --weight-format int8 \
  --output ./my_custom_model
```

But for this project, **stick to `OpenVINO/` models** - they're already optimized! 🎯

---

## Key Features

### ✨ New Capabilities

1. **Multi-Model Support**: Test multiple models in a single run
2. **JSON Configuration**: Easy model management via `benchmark.json`
3. **Cross-Model Comparisons**: Compare performance across different model sizes and architectures
4. **Device Analysis**: Detailed statistics for each device across all models
5. **Results Export**: Saves results to `benchmark_results.json` for further analysis
6. **Smart Recommendations**: Automatic suggestions for optimal model/device combinations

### 📊 Comparison Types

- **Per-Model Summary**: Performance breakdown for each model across devices
- **Cross-Model Comparison**: All models and devices ranked by performance
- **Device Analysis**: Average performance and statistics per device
- **Top Performers**: Best model/device combinations

## Configuration

### benchmark.json Structure

```json
{
  "models": [
    {
      "name": "Model display name",
      "model_id": "HuggingFace model ID",
      "size": "Model size (e.g., '7B')",
      "quantization": "INT4, INT4-CW, etc.",
      "recommended_devices": ["GPU", "NPU", "CPU"],
      "description": "Brief description",
      "enabled": true/false  ← Set to true to benchmark
    }
  ],
  "benchmark_config": {
    "test_prompts": [...],
    "generation_config": {...},
    "devices_to_test": ["CPU", "GPU", "NPU"],
    "cache_dir": "./ov_cache"
  }
}
```

### Enabling Models

To benchmark a model, set `"enabled": true` in benchmark.json:

```json
{
  "name": "Mistral-7B-Instruct-v0.3",
  "enabled": true  ← This model will be tested
}
```

**Recommendation**: Start with 1-2 small models (1-4B parameters) to test quickly.

## Usage

### Basic Usage

```bash
# Run benchmark with default configuration
python benchmark_devices.py
```

The script will:
1. Load `benchmark.json`
2. Download enabled models from HuggingFace
3. Test each model on available devices
4. Generate comprehensive reports
5. Save results to `benchmark_results.json`

### Quick Start Examples

**Test a single small model:**
```json
"Qwen3-1.7B": { "enabled": true }
// All others: "enabled": false
```

**Test multiple models of different sizes:**
```json
"TinyLlama-1.1B": { "enabled": true },
"Qwen3-4B": { "enabled": true },
"Mistral-7B-Instruct-v0.3": { "enabled": true }
```

**Test specific devices only:**
```json
"benchmark_config": {
  "devices_to_test": ["GPU", "NPU"]  // Skip CPU
}
```

## Output

### Console Output

The script provides detailed output in 6 stages:

1. **Configuration Loading**: Shows enabled models
2. **Device Detection**: Lists available hardware
3. **Model Benchmarking**: Real-time progress for each model/device
4. **Comprehensive Results**: Cross-model comparisons
5. **Recommendations**: Optimal configurations
6. **Results Export**: Saves to JSON file

### Sample Output

```
🚀 OpenVINO Multi-Model Device Benchmark
================================================================================

[1/6] Loading configuration...
✅ Found 2 enabled model(s):
   • TinyLlama-1.1B (1.1B) - INT4
   • Mistral-7B-Instruct-v0.3 (7B) - INT4-CW

[2/6] Checking available devices...
Available devices: ['CPU', 'GPU.0', 'NPU']
✅ Will test devices: CPU, GPU, NPU

[3/6] Downloading and benchmarking models...
================================================================================

📦 MODEL 1/2: TinyLlama-1.1B
   Size: 1.1B | Quantization: INT4
================================================================================

🔧 Testing TinyLlama-1.1B on CPU
  Loading model to CPU...
  ✅ Loaded in 2.3s
  Running 5 test prompts on CPU...
    [1/5] What is artificial intelligence?...
         3.45s | 98 tokens | 28.4 tok/s
    ...

================================================================================
📊 TinyLlama-1.1B - Summary
================================================================================

Device     Load (s)     Avg Time (s)    Avg Speed        Total Time (s)   
--------------------------------------------------------------------------------
NPU            3.1            2.87         35.2 tok/s           14.3
GPU            2.5            3.12         32.1 tok/s           15.6
CPU            2.3            3.54         28.4 tok/s           17.7

🏆 Fastest: NPU at 35.2 tok/s

[4/6] COMPREHENSIVE RESULTS
====================================================================================================
🔍 CROSS-MODEL COMPARISON
====================================================================================================

Model                          Size     Device     Speed               Load (s)     Avg Time (s)
----------------------------------------------------------------------------------------------------
TinyLlama-1.1B                1.1B     NPU        35.2 tok/s             3.1          2.87
TinyLlama-1.1B                1.1B     GPU        32.1 tok/s             2.5          3.12
Mistral-7B-Instruct-v0.3      7B       GPU        28.6 tok/s             5.2          3.49
TinyLlama-1.1B                1.1B     CPU        28.4 tok/s             2.3          3.54
Mistral-7B-Instruct-v0.3      7B       CPU        18.7 tok/s             4.8          5.35
Mistral-7B-Instruct-v0.3      7B       NPU        16.2 tok/s             6.1          6.17

====================================================================================================
🏆 TOP PERFORMERS
====================================================================================================

1️⃣  Overall Fastest: TinyLlama-1.1B on NPU
    Speed: 35.2 tok/s

📊 Best Model per Device:
   CPU: TinyLlama-1.1B (1.1B) - 28.4 tok/s
   GPU: TinyLlama-1.1B (1.1B) - 32.1 tok/s
   NPU: TinyLlama-1.1B (1.1B) - 35.2 tok/s

📊 Best Device per Model:
   Mistral-7B-Instruct-v0.3: GPU - 28.6 tok/s
   TinyLlama-1.1B: NPU - 35.2 tok/s

[5/6] RECOMMENDATIONS
====================================================================================================

✨ Optimal Configuration:
   Model: TinyLlama-1.1B
   Device: NPU
   Speed: 35.2 tok/s

   Code:
   model_id = "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"
   device = "NPU"
   pipe = ov_genai.LLMPipeline(model_dir, device)

💡 General Guidelines:
   • Smaller models (1-4B): Faster, good for NPU
   • Larger models (7-8B): Better quality, prefer GPU
   • NPU: Best power efficiency for mobile/battery use
   • GPU: Best raw performance for desktop/workstation
   • CPU: Most compatible, good for servers/cloud

[6/6] Saving results...
✅ Results saved to benchmark_results.json
```

### Results JSON

The script saves detailed results to `benchmark_results.json`:

```json
{
  "timestamp": "2025-10-30 14:23:45",
  "devices_tested": ["CPU", "GPU", "NPU"],
  "models_tested": ["TinyLlama-1.1B", "Mistral-7B-Instruct-v0.3"],
  "results": {
    "TinyLlama-1.1B": {
      "NPU": {
        "load_time": 3.1,
        "avg_time": 2.87,
        "avg_speed": 35.2,
        "avg_tokens": 100.2,
        "total_time": 14.3,
        "total_tokens": 501
      },
      ...
    }
  }
}
```

## Model Selection Guide

### By Use Case

**Power Efficiency (Battery/Mobile)**
- Enable: TinyLlama-1.1B, Qwen3-1.7B
- Test on: NPU first

**Balanced Performance**
- Enable: Qwen3-4B, Phi-3-mini-128k
- Test on: NPU, GPU

**Maximum Quality**
- Enable: Mistral-7B, Qwen3-8B
- Test on: GPU first

**Quick Testing**
- Enable: 1-2 models (TinyLlama + one other)
- Limit devices if needed

### By Model Size

| Size | Models | Best For | Typical Speed (NPU) |
|------|--------|----------|---------------------|
| 1-2B | TinyLlama, Qwen3-1.7B | Speed, efficiency | 30-40 tok/s |
| 3-5B | Phi-3-mini, Qwen3-4B | Balance | 20-30 tok/s |
| 7-8B | Mistral-7B, Qwen3-8B | Quality | 15-25 tok/s |

## Key Changes from Original Script

### Added Features

1. **JSON Configuration System**
   - Load models from `benchmark.json`
   - Easy enable/disable per model
   - Centralized test configuration

2. **Multi-Model Support**
   - Automatically download and test multiple models
   - Compare models side-by-side
   - Track results per model/device combination

3. **Enhanced Reporting**
   - Cross-model comparison tables
   - Device performance analysis
   - Top performers identification
   - Optimal configuration recommendations

4. **Better Organization**
   - Modular functions for each task
   - Separate summary per model
   - Comprehensive final analysis

5. **Results Export**
   - Save to `benchmark_results.json`
   - Timestamp tracking
   - Structured data for further analysis

6. **Error Handling**
   - Graceful handling of failed downloads
   - Continue testing if one model/device fails
   - Clear error messages

### Modified Behavior

- Models downloaded to `models/{model_name}/` 
- Results saved automatically
- More detailed progress indicators
- Better visual formatting with emojis and tables

## Troubleshooting

### No Models Enabled

```
❌ No models enabled in benchmark.json!
```

**Solution**: Set at least one model's `"enabled": true`

### Model Download Failed

```
❌ Error downloading Model-Name: ...
⚠️  Skipping Model-Name due to download error
```

**Solution**: Check internet connection and HuggingFace model ID

### Device Not Available

If a device isn't tested, it means it's not available on your system. The script will automatically skip unavailable devices.

### Out of Memory

If larger models (7-8B) fail on NPU:
1. Try smaller models first
2. Use GPU instead for large models
3. Enable fewer models at once

## Performance Tips

1. **Start Small**: Test 1-2 small models first to validate setup
2. **Use Cache**: The script caches compiled models in `./ov_cache`
3. **Batch Testing**: Enable multiple models for comprehensive comparison
4. **Device Priority**: Test on recommended_devices listed in model config

## Dependencies

- `openvino` and `openvino_genai`
- `huggingface_hub`
- Python 3.8+

## Example Workflows

### Workflow 1: Quick NPU Test
```json
// Enable only small models
"TinyLlama-1.1B": { "enabled": true },
"Qwen3-1.7B": { "enabled": true },

// Test NPU only
"devices_to_test": ["NPU"]
```

### Workflow 2: Complete Comparison
```json
// Enable multiple sizes
"TinyLlama-1.1B": { "enabled": true },
"Qwen3-4B": { "enabled": true },
"Mistral-7B-Instruct-v0.3": { "enabled": true },

// Test all devices
"devices_to_test": ["CPU", "GPU", "NPU"]
```

### Workflow 3: GPU Performance Test
```json
// Enable larger models
"Mistral-7B-Instruct-v0.3": { "enabled": true },
"Qwen3-8B": { "enabled": true },

// GPU only
"devices_to_test": ["GPU"]
```

## Next Steps

After benchmarking:

1. Review `benchmark_results.json` for detailed metrics
2. Use optimal model/device combination in your application
3. Adjust `generation_config` parameters if needed
4. Add custom test prompts relevant to your use case

## Support

For issues with:
- **OpenVINO**: Check OpenVINO documentation
- **Models**: Verify model IDs on HuggingFace
- **Hardware**: Ensure drivers are installed (GPU/NPU)
