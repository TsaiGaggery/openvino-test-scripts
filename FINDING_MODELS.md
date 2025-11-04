# Finding Available OpenVINO Models

## 🔍 How to Find Models on HuggingFace

### **Official OpenVINO Organization:**
https://huggingface.co/OpenVINO

Browse all available pre-optimized models here!

---

## ✅ Verified Available Models (as of Nov 2025)

### **Tiny Models (< 1B) - Ultra Fast on NPU:**
- ✅ `OpenVINO/SmolLM-135M-Instruct-int4-ov` (135M)
- ✅ `OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov` (1.1B)

### **Small Models (1-2B) - Excellent for NPU:**
- ✅ `OpenVINO/Llama-3.2-1B-Instruct-int4-ov` (1B)
- ✅ `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov` (1.5B)
- ✅ `OpenVINO/Qwen3-1.7B-int4-ov` (1.7B) ⚠️ NPU issues

### **Medium Models (3-4B) - Good for NPU:**
- ✅ `OpenVINO/Llama-3.2-3B-Instruct-int4-ov` (3B)
- ✅ `OpenVINO/Qwen2.5-3B-Instruct-int4-ov` (3B)
- ✅ `OpenVINO/stablelm-zephyr-3b-int4-ov` (3B)
- ✅ `OpenVINO/Phi-3-mini-128k-instruct-int4-ov` (3.8B)
- ✅ `OpenVINO/Phi-3.5-mini-instruct-int4-ov` (3.8B)
- ✅ `OpenVINO/Qwen3-4B-int4-ov` (4B)

### **Large Models (7-8B) - Better on GPU:**
- ✅ `OpenVINO/Llama-2-7b-chat-int4-ov` (7B)
- ✅ `OpenVINO/Mistral-7B-Instruct-v0.2-int4-cw-ov` (7B)
- ✅ `OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov` (7B)
- ✅ `OpenVINO/Qwen3-8B-int4-cw-ov` (8B)

---

## ❌ Models NOT Available (Common Mistakes)

### **Don't Exist in OpenVINO Hub:**
- ❌ `OpenVINO/SmolLM-1.7B-Instruct-int4-ov`
  - **Use instead:** `OpenVINO/SmolLM-135M-Instruct-int4-ov`

- ❌ `OpenVINO/stablelm-2-1_6b-chat-int4-ov`
  - **Use instead:** `OpenVINO/stablelm-zephyr-3b-int4-ov`

- ❌ `OpenVINO/gemma-2b-it-int4-ov`
  - **Use instead:** Try Qwen or Llama models

- ❌ `OpenVINO/gemma-2-2b-it-int4-ov`
  - **Use instead:** Try Qwen or Llama models

### **Common Typos:**
- ❌ `OpenVINO/qwen2.5-1.5b-instruct-int4-ov` (wrong case)
  - ✅ `OpenVINO/Qwen2.5-1.5B-Instruct-int4-ov` (correct)

- ❌ `OpenVINO/llama-3.2-3b-instruct-int4-ov` (wrong case)
  - ✅ `OpenVINO/Llama-3.2-3B-Instruct-int4-ov` (correct)

---

## 🔍 How to Verify a Model Exists

### **Method 1: Browser**
Go to: `https://huggingface.co/OpenVINO/[MODEL-NAME]`

Example:
- https://huggingface.co/OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov ✅
- https://huggingface.co/OpenVINO/SmolLM-1.7B-Instruct-int4-ov ❌

### **Method 2: Python**
```python
from huggingface_hub import list_models

# List all OpenVINO models
models = list_models(author="OpenVINO")
for model in models:
    if 'int4' in model.modelId.lower():
        print(model.modelId)
```

### **Method 3: Search Page**
Visit: https://huggingface.co/models?other=openvino&sort=trending

Filter by:
- Organization: OpenVINO
- Task: Text Generation
- Library: OpenVINO

---

## 📋 Recommended NPU Model Set (All Verified)

### **Quick Test Set (3 models):**
```json
{
  "enabled_models": [
    "TinyLlama-1.1B",           // Fastest, most reliable
    "Llama-3.2-1B-Instruct",    // Meta's latest small model
    "Qwen2.5-3B-Instruct"       // Best quality for size
  ]
}
```

### **Comprehensive Test Set (6 models):**
```json
{
  "enabled_models": [
    "SmolLM-135M",              // Ultra fast
    "TinyLlama-1.1B",           // Very fast
    "Llama-3.2-1B-Instruct",    // Fast + good quality
    "Qwen2.5-1.5B-Instruct",    // Balanced
    "Qwen2.5-3B-Instruct",      // Good quality
    "Phi-3.5-mini-instruct"     // Best quality (3.8B)
  ]
}
```

### **Multi-Size Comparison (5 models):**
```json
{
  "enabled_models": [
    "Llama-3.2-1B-Instruct",    // 1B baseline
    "Qwen2.5-3B-Instruct",      // 3B mid-range
    "Phi-3.5-mini-instruct",    // 3.8B balanced
    "Mistral-7B-Instruct-v0.3", // 7B large
    "Qwen3-8B"                  // 8B largest
  ]
}
```

---

## 🛠️ Troubleshooting Model Download Errors

### **Error: "Repository Not Found" (401 Error)**

**Causes:**
1. Model doesn't exist at that path
2. Typo in model name
3. Model was renamed or removed

**Solutions:**
```bash
# 1. Check the OpenVINO hub
# Visit: https://huggingface.co/OpenVINO

# 2. Search for similar models
# Visit: https://huggingface.co/models?search=openvino%20int4

# 3. Use verified models from the list above
```

### **Error: "Invalid username or password"**

This usually means the model doesn't exist (confusing error message).
- **Not** an authentication issue
- Model path is incorrect
- Check spelling and capitalization

### **Error: "Connection timeout"**

Network or HuggingFace API issue:
```bash
# Retry the download
python benchmark_devices.py

# Or set longer timeout
export HF_HUB_DOWNLOAD_TIMEOUT=600
python benchmark_devices.py
```

---

## 🎯 Best Practice: Test Before Adding

Before adding a model to `benchmark.json`, verify it exists:

```python
from huggingface_hub import model_info

model_id = "OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov"

try:
    info = model_info(model_id)
    print(f"✅ Model exists: {info.modelId}")
    print(f"   Last modified: {info.lastModified}")
    print(f"   Downloads: {info.downloads}")
except Exception as e:
    print(f"❌ Model not found: {e}")
```

---

## 📊 Model Naming Convention

OpenVINO models follow this pattern:
```
OpenVINO/[ModelFamily]-[Size]-[Variant]-[Quantization]-ov

Examples:
- OpenVINO/Llama-3.2-1B-Instruct-int4-ov
- OpenVINO/Qwen2.5-3B-Instruct-int4-ov
- OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov

Where:
- [ModelFamily] = Llama, Qwen, Mistral, Phi, etc.
- [Size] = 1B, 3B, 7B, etc.
- [Variant] = Instruct, Chat, Base, etc.
- [Quantization] = int4, int4-cw, int8, etc.
- -ov = OpenVINO format suffix
```

---

## 🔄 Keeping Models Updated

OpenVINO releases new optimized models regularly:

1. **Check monthly:** https://huggingface.co/OpenVINO
2. **Look for:** "Updated [recently]" tag
3. **Prefer:** Newer quantizations and model versions
4. **Test:** New models before production use

---

## 📞 Getting Help

**If model not found:**
1. Search OpenVINO org: https://huggingface.co/OpenVINO
2. Check model discussions: Click "Community" tab on model page
3. Report issue: https://github.com/openvinotoolkit/openvino/issues
4. Use verified models: From the list above

**Model exists but download fails:**
1. Check network connection
2. Increase timeout: `export HF_HUB_DOWNLOAD_TIMEOUT=600`
3. Try different mirror/region
4. Check disk space (models are large!)
