# Auto-Skip Feature for Problematic Models

## ✅ **New Feature Added!**

The benchmark script now **automatically skips** models that are known to cause NPU compilation crashes, preventing the entire benchmark from failing.

---

## 🚀 **How It Works**

### **1. Pre-emptive Detection**
The script checks model names against a list of known problematic patterns:
- `qwen3` models (e.g., Qwen3-1.7B, Qwen3-4B)
- `gemma-2b` models (e.g., Gemma-2B, Gemma-2-2B)

### **2. Automatic Skip**
When testing on NPU, if a model matches a problematic pattern:
```
⚠️  Known NPU compatibility issue with this model
💡 Skipping NPU test for this model to prevent crash
💡 This model will be tested on other devices (GPU/CPU)
```

### **3. Continue Testing**
- The script continues with remaining devices (GPU, CPU)
- Other models are still tested on NPU
- No crash, no interruption!

### **4. Clear Reporting**
In the summary, you'll see:
```
⚠️  Skipped:
   • NPU: Known compatibility issue
```

---

## 📋 **Currently Auto-Skipped on NPU**

These models will automatically skip NPU testing:
- ❌ **Qwen3-1.7B** - MatMul tensor mismatch
- ❌ **Qwen3-4B** - Similar architecture issues
- ❌ **Gemma-2B** - Channel count error
- ❌ **Gemma-2-2B** - Related architecture

**They will still be tested on:**
- ✅ GPU (usually works well)
- ✅ CPU (always compatible)

---

## 🛠️ **Additional Safety Features Added**

### **1. Timeout Protection (Linux)**
Models that hang during NPU compilation will timeout after 5 minutes:
```python
# Prevents indefinite hangs
timeout_seconds = 300  # 5 minutes max
```

### **2. Enhanced Error Messages**
When NPU errors occur:
```
❌ NPU Compilation Error: Model incompatible with NPU
   This model architecture is not supported by the NPU compiler.
   💡 Solutions:
      1. Try a different model (TinyLlama, Phi-3, or StableLM work well)
      2. Use GPU or CPU for this model instead
      3. Check for updated model version at huggingface.co/OpenVINO
   ℹ️  Continuing with remaining tests...
```

### **3. Graceful Continuation**
- Script doesn't crash on NPU errors
- Continues testing other models
- Continues testing other devices
- Generates complete report with successful tests

---

## 📊 **Example Output**

### **Before (Would Crash):**
```
Testing Gemma-2B on NPU...
[ERROR] Channels count mismatch...
Aborted (core dumped)
❌ ENTIRE BENCHMARK STOPPED
```

### **After (Graceful Skip):**
```
Testing Gemma-2B on NPU...
⚠️  Known NPU compatibility issue with this model
💡 Skipping NPU test for this model to prevent crash
💡 This model will be tested on other devices (GPU/CPU)

Testing Gemma-2B on GPU...
✅ GPU completed: 45.2 tok/s

Testing Gemma-2B on CPU...
✅ CPU completed: 18.7 tok/s

Testing TinyLlama-1.1B on NPU...
✅ NPU completed: 32.4 tok/s (good performance)
```

---

## 🎯 **Recommended Models for NPU**

Since some models are now auto-skipped, here are the **best NPU models** to enable:

### **Ultra-Fast (1-2B):**
```json
{
  "TinyLlama-1.1B": "enabled: true",
  "Qwen2.5-1.5B-Instruct": "enabled: true",
  "StableLM-2-1.6B": "enabled: true",
  "SmolLM-1.7B-Instruct": "enabled: true"
}
```

### **Balanced (3-4B):**
```json
{
  "Phi-3-mini-128k": "enabled: true",
  "Phi-3.5-mini-instruct": "enabled: true",
  "Qwen2.5-3B-Instruct": "enabled: true",
  "Llama-3.2-3B-Instruct": "enabled: true"
}
```

---

## 🔧 **Customize Skip List**

You can modify the skip patterns in `benchmark_devices.py`:

```python
# Line ~110 in benchmark_devices.py
npu_problematic_patterns = ['qwen3', 'gemma-2b']

# Add more patterns if you discover other problematic models:
npu_problematic_patterns = ['qwen3', 'gemma-2b', 'your-pattern']
```

---

## ✨ **Benefits**

1. ✅ **No More Crashes** - Script always completes
2. ✅ **Complete Reports** - Get results for all working models
3. ✅ **Time Saving** - Don't waste time on incompatible models
4. ✅ **Clear Guidance** - Know which models work on which devices
5. ✅ **HTML Reports** - Beautiful reports even with some skips

---

## 📝 **Summary**

**What happens now when running benchmark:**
1. Script checks each model against skip list
2. Auto-skips problematic NPU tests (with warning)
3. Tests model on other devices (GPU/CPU)
4. Continues to next model
5. Generates complete report with all successful tests
6. Shows clear summary of what was skipped and why

**Result:** Robust, crash-free benchmarking with maximum coverage! 🎉
