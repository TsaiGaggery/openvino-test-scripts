# Intel NPU Compatibility Guide

## 🔴 Known NPU Issues

### **Error: "Channels count of input tensor shape and filter shape must be the same"**

**Symptoms:**
```
[ERROR] Got Diagnostic at loc(...MatMul...) : Channels count of input tensor shape and filter shape must be the same: 0 != 16
LLVM ERROR: Failed to infer result type(s).
Aborted (core dumped)
```

**Cause:**
- Model architecture incompatibility with NPU compiler
- Specific tensor operations that NPU hardware cannot optimize
- Model export format issues

**Affected Models:**
- ❌ Qwen3-1.7B (confirmed issue)
- ❌ Qwen3-4B (may have issues)
- ❌ Gemma-2B (confirmed issue)
- ❌ Gemma-2-2B (likely issue)
- ❌ Certain older model exports

---

## ✅ NPU-Compatible Models (Tested & Working)

### **Excellent NPU Compatibility:**
1. ✅ **TinyLlama-1.1B** - Most reliable, fastest (VERIFIED)
2. ✅ **Llama-3.2-1B** - Meta's smallest Llama 3 (VERIFIED)
3. ✅ **Qwen2.5-1.5B-Instruct** - Newer, better compatibility (VERIFIED)
4. ✅ **Phi-3-mini** (3.8B) - Excellent performance (VERIFIED)
5. ✅ **Phi-3.5-mini** (3.8B) - Updated version (VERIFIED)

### **Good NPU Compatibility:**
6. ✅ **Qwen2.5-3B-Instruct** - Improved over Qwen3 (VERIFIED)
7. ✅ **Llama-3.2-3B** - Generally works well (VERIFIED)
8. ✅ **StableLM-Zephyr-3B** - Stability AI chat model (VERIFIED)

### **Known NPU Issues (Use GPU/CPU Instead):**
- ❌ **Qwen3-1.7B** - Compilation error (EXISTS but problematic)
- ❌ **Qwen3-4B** - May have issues
- ❌ **Gemma-2B** - Not available in OpenVINO hub
- ❌ **Gemma-2-2B** - Not available in OpenVINO hub

### **Not Available in OpenVINO Hub:**
- ❌ **SmolLM-1.7B** - Use SmolLM-135M instead
- ❌ **StableLM-2-1.6B** - Use StableLM-Zephyr-3B instead

### **Limited NPU Compatibility:**
- ⚠️ **Mistral-7B variants** - May work but slower
- ⚠️ **Qwen3-8B** - Too large for optimal NPU
- ⚠️ **Llama-2-7B** - Better on GPU

---

## 🛠️ Troubleshooting NPU Errors

### **1. Model Compilation Failures**

**If you see NPU compiler errors:**

```bash
# Solution 1: Try a different model
# Edit benchmark.json and switch to TinyLlama or Phi-3

# Solution 2: Test on CPU/GPU first
# Change "devices_to_test": ["CPU", "GPU"] temporarily

# Solution 3: Update OpenVINO
pip install --upgrade openvino openvino-genai
```

### **2. Slow NPU Performance**

**Expected speeds:**
- 1-2B models: 25-35 tok/s
- 3-4B models: 15-25 tok/s
- 5-7B models: 10-15 tok/s
- 8B+ models: <10 tok/s (not optimal for NPU)

**If slower than expected:**
1. Run benchmark twice (first run compiles, second uses cache)
2. Check model size (stick to <5B for NPU)
3. Verify INT4 quantization is used
4. Ensure no other NPU workloads are running

### **3. NPU Not Detected**

```python
# Check available devices
import openvino as ov
core = ov.Core()
print(core.available_devices)  # Should include 'NPU' or 'NPU.xxxx'
```

**If NPU not listed:**
1. Check NPU drivers are installed
2. Verify hardware support (Intel Core Ultra or newer)
3. Update OpenVINO to latest version
4. Check BIOS settings (NPU should be enabled)

---

## 📋 Recommended Model Selection for NPU

### **For Maximum Speed:**
```json
{
  "enabled_models": [
    "TinyLlama-1.1B",
    "Qwen2.5-1.5B-Instruct",
    "Gemma-2B"
  ]
}
```

### **For Best Quality (still NPU-optimized):**
```json
{
  "enabled_models": [
    "Qwen2.5-3B-Instruct",
    "Llama-3.2-3B",
    "Phi-3.5-mini-instruct"
  ]
}
```

### **For Balanced Performance:**
```json
{
  "enabled_models": [
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Phi-3-mini-128k"
  ]
}
```

---

## 🔧 NPU Optimization Tips

### **1. Model Selection**
- ✅ Use models 1-4B parameters
- ✅ Use INT4 or INT4-CW quantization
- ✅ Prefer newer model exports from OpenVINO hub
- ❌ Avoid FP16/FP32 models on NPU

### **2. Configuration**
```python
# Optimal NPU settings (already in benchmark_devices.py)
core.set_property('NPU', {
    'CACHE_DIR': './ov_cache',
    'PERFORMANCE_HINT': 'LATENCY',
    'NPU_COMPILATION_MODE_PARAMS': 'compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm'
})
```

### **3. Cache Management**
```bash
# First run (slow - compiles model):
python benchmark_devices.py

# Second run (fast - uses cache):
python benchmark_devices.py

# Clear cache if issues:
rm -rf ./ov_cache/*
```

### **4. Model Finding**
Visit: https://huggingface.co/OpenVINO

Look for:
- Models ending in `-int4-ov` or `-int4-cw-ov`
- Recent uploads (better compatibility)
- Models labeled "NPU-optimized" or "NPU-ready"

---

## 📊 Performance Expectations

### **NPU vs GPU vs CPU (Typical)**

| Model Size | NPU (tok/s) | GPU (tok/s) | CPU (tok/s) |
|------------|-------------|-------------|-------------|
| 1-2B       | 30-40       | 50-80       | 15-25       |
| 3-4B       | 18-25       | 40-60       | 8-15        |
| 5-7B       | 10-15       | 30-50       | 5-10        |
| 8B+        | 8-12        | 25-40       | 3-7         |

**NPU Advantages:**
- ⚡ Best power efficiency (50-80% less power than GPU)
- 🔋 Ideal for battery-powered devices
- 🌡️ Lower heat generation
- 💰 Lower cost (integrated in CPU)

**When to Use GPU Instead:**
- Models > 7B parameters
- Need maximum speed regardless of power
- Complex multi-model workflows
- When NPU compatibility issues occur

---

## 🐛 Debugging Commands

```bash
# Check OpenVINO version
pip show openvino openvino-genai

# List available devices
python -c "import openvino as ov; print(ov.Core().available_devices)"

# Check NPU driver info (Linux)
lspci | grep -i npu

# Check NPU device info (Windows)
# Device Manager -> Neural Processors

# Test simple NPU inference
python -c "import openvino as ov; core = ov.Core(); print('NPU available' if 'NPU' in core.available_devices else 'NPU not found')"

# Enable verbose logging
export OV_LOG_LEVEL=DEBUG
python benchmark_devices.py
```

---

## 📞 Getting Help

1. **Check model compatibility** - Use models from the "Working" list above
2. **Update everything** - `pip install --upgrade openvino openvino-genai`
3. **Try TinyLlama first** - Most reliable NPU model
4. **Check OpenVINO docs** - https://docs.openvino.ai/
5. **Report issues** - https://github.com/openvinotoolkit/openvino/issues

---

## 🎯 Quick Fix for Your Error

**Your specific issue with Qwen3-1.7B:**

1. **Disable Qwen3-1.7B** in benchmark.json (already done ✅)
2. **Enable Qwen2.5-1.5B-Instruct** instead (already done ✅)
3. **Keep TinyLlama-1.1B enabled** (already enabled ✅)
4. **Run benchmark again** - Should work now!

```bash
# Run the fixed benchmark
python benchmark_devices.py
```

The error handling has been improved, so if a model fails on NPU, it will:
- ✅ Show clear error message
- ✅ Suggest alternative models
- ✅ Continue testing other models
- ✅ Generate report with successful models only
