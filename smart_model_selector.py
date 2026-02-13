#!/usr/bin/env python3
"""
Smart Model Selector - Automatically select optimal models based on device capabilities

This script queries device capabilities via the OpenVINO Python API, determines the best
common quantization format, searches HuggingFace for matching models, and generates
an optimized benchmark configuration file.
"""
import re
import json
from pathlib import Path
from huggingface_hub import HfApi
import sys
import openvino as ov


def query_device_capabilities():
    """Use OpenVINO Python API to query device capabilities"""
    print("=" * 80)
    print("🔍 Querying Device Capabilities via OpenVINO Python API")
    print("=" * 80)

    core = ov.Core()
    available_devices = core.available_devices

    device_caps = {}

    for device in available_devices:
        try:
            # Get device optimization capabilities
            caps = core.get_property(device, 'OPTIMIZATION_CAPABILITIES')
            full_name = core.get_property(device, 'FULL_DEVICE_NAME')

            device_caps[device] = {
                'capabilities': list(caps),
                'full_name': full_name
            }

            print(f"\n📱 {device} - {full_name}")
            print(f"   Capabilities: {', '.join(caps)}")

        except Exception as e:
            print(f"   ⚠️  Cannot query {device}: {e}")

    return device_caps

def find_common_capabilities(device_caps, devices_to_test=None):
    """Find common optimization capabilities across all devices"""
    if devices_to_test is None:
        devices_to_test = list(device_caps.keys())

    # Filter to keep only devices to test
    test_devices = {dev: caps for dev, caps in device_caps.items() if dev in devices_to_test}

    if not test_devices:
        print("❌ No available devices")
        return set()

    # Find common capabilities
    common_caps = set(list(test_devices.values())[0]['capabilities'])

    for device, info in test_devices.items():
        common_caps &= set(info['capabilities'])

    print(f"\n🎯 Common Capabilities Across All Devices:")
    print(f"   {', '.join(sorted(common_caps))}")

    return common_caps

def determine_best_quantization(common_caps):
    """
    Determine best quantization format based on common capabilities
    Priority: INT4 > INT8 > FP16 > FP32
    """
    # Quantization format priority (from best to worst)
    quantization_priority = [
        ('INT4', 'int4'),      # INT4 - smallest, fastest
        ('INT8', 'int8'),      # INT8 - small, fast
        ('FP16', 'fp16'),      # FP16 - medium
        ('BF16', 'bf16'),      # BF16 - medium
        ('FP32', 'fp32'),      # FP32 - large, slow
    ]

    print(f"\n🎯 Determining Best Quantization Format:")

    for cap_name, quant_suffix in quantization_priority:
        if cap_name in common_caps:
            print(f"   ✅ Selected {cap_name} (supported by all devices)")
            return cap_name, quant_suffix

    print(f"   ⚠️  No standard quantization format found, using default FP16")
    return 'FP16', 'fp16'

def search_models_on_huggingface(quantization='int4', max_results=20):
    """Search for models with specified quantization in HuggingFace OpenVINO organization"""
    print(f"\n🔍 Searching for {quantization} models on HuggingFace...")
    print(f"   Query: OpenVINO organization, {quantization} models")

    api = HfApi()

    try:
        # Search models in OpenVINO organization
        models = list(api.list_models(
            author="OpenVINO",
            sort="downloads",
            direction=-1,
            limit=100
        ))

        # Filter models with specified quantization format
        quantization_models = []
        search_patterns = [f'-{quantization}-', f'_{quantization}_', f'-{quantization.upper()}-']

        for model in models:
            model_id = model.modelId.lower()
            # Check if model contains quantization format identifier
            if any(pattern.lower() in model_id for pattern in search_patterns):
                quantization_models.append({
                    'model_id': model.modelId,
                    'downloads': model.downloads if hasattr(model, 'downloads') else 0,
                    'likes': model.likes if hasattr(model, 'likes') else 0,
                })

        # Sort by downloads
        quantization_models.sort(key=lambda x: x['downloads'], reverse=True)

        return quantization_models[:max_results]

    except Exception as e:
        print(f"   ❌ Search failed: {e}")
        return []

def categorize_models_by_size(models):
    """Categorize models by parameter size"""
    categories = {
        'tiny': [],      # < 1B
        'small': [],     # 1-2B
        'medium': [],    # 3-5B
        'large': [],     # 7-9B
        'xlarge': []     # > 10B
    }

    size_patterns = {
        'tiny': ['135m', '150m', '0.5b'],
        'small': ['1b', '1.1b', '1.5b', '1.7b', '2b'],
        'medium': ['3b', '3.8b', '4b', '5b'],
        'large': ['7b', '8b', '9b'],
        'xlarge': ['11b', '13b', '14b', '70b']
    }

    for model in models:
        model_id = model['model_id'].lower()
        categorized = False

        # Skip non-LLM models (whisper, bge, etc.)
        skip_keywords = ['whisper', 'bge', 'reranker', 'embed', 'vision']
        if any(keyword in model_id for keyword in skip_keywords):
            continue

        for category, patterns in size_patterns.items():
            for pattern in patterns:
                if pattern in model_id:
                    categories[category].append(model)
                    categorized = True
                    break
            if categorized:
                break

        # If not categorized, assume medium
        if not categorized:
            categories['medium'].append(model)

    return categories

def extract_model_size(model_id):
    """Extract model size from model ID"""
    model_lower = model_id.lower()

    # Find XB or X.YB pattern
    size_match = re.search(r'(\d+\.?\d*)b', model_lower)
    if size_match:
        return size_match.group(1) + 'B'

    # Find XM pattern
    size_match = re.search(r'(\d+)m', model_lower)
    if size_match:
        return size_match.group(1) + 'M'

    return 'Unknown'

def generate_recommended_config(device_caps, common_caps, quantization_type, models):
    """Generate recommended configuration based on device capabilities"""
    print("\n" + "=" * 80)
    print("📋 Generating Recommended Configuration")
    print("=" * 80)

    # Categorize models by size
    categorized = categorize_models_by_size(models)

    # Select recommended models
    recommended_models = []

    # NPU: Prioritize small models
    npu_exists = 'NPU' in device_caps
    if npu_exists:
        print("\n💻 NPU detected - Recommending small models (1-4B)")
        # Select from tiny and small categories
        npu_models = (categorized['tiny'][:2] + categorized['small'][:2] + categorized['medium'][:1])[:3]
        recommended_models.extend(npu_models)

    # GPU/CPU: Can handle larger models
    gpu_exists = any('GPU' in dev for dev in device_caps)
    if gpu_exists:
        print("🎮 GPU detected - Can test medium to large models (3-8B)")
        gpu_models = (categorized['medium'][:2] + categorized['large'][:1])[:2]
        recommended_models.extend(gpu_models)

    # CPU only mode or no recommendations yet - add comprehensive selection
    if not recommended_models:
        print("\n🖥️  CPU mode - Recommending various sizes for comprehensive testing")
        cpu_models = (
            categorized['tiny'][:2] +      # 2 tiny models
            categorized['small'][:2] +     # 2 small models
            categorized['medium'][:2] +    # 2 medium models
            categorized['large'][:1]       # 1 large model
        )
        recommended_models.extend(cpu_models)

    # Remove duplicates
    seen = set()
    unique_models = []
    for model in recommended_models:
        if model['model_id'] not in seen:
            seen.add(model['model_id'])
            unique_models.append(model)

    # Generate configuration
    config_models = []

    for idx, model in enumerate(unique_models[:8]):  # Max 8 models
        model_id = model['model_id']
        model_name = model_id.split('/')[-1]
        size = extract_model_size(model_id)

        # Recommend devices based on size
        size_num = float(size.replace('B', '').replace('M', '0.001')) if size != 'Unknown' else 3.0

        if size_num < 2.0:
            recommended_devices = ["NPU", "GPU", "CPU"]
        elif size_num < 5.0:
            recommended_devices = ["NPU", "GPU", "CPU"] if npu_exists else ["GPU", "CPU"]
        else:
            recommended_devices = ["GPU", "CPU"]

        config_models.append({
            "name": model_name.replace('-int4-ov', '').replace('-int8-ov', ''),
            "model_id": model_id,
            "size": size,
            "quantization": quantization_type.upper(),
            "recommended_devices": recommended_devices,
            "description": f"Auto-selected {quantization_type.upper()} model - {size} parameters",
            "enabled": idx < 5,  # Enable first 5 by default
            "downloads": model['downloads'],
            "notes": f"Auto-detected based on device capabilities. Downloads: {model['downloads']:,}"
        })

    return config_models

def save_config(device_caps, common_caps, quantization_type, models, output_file="benchmark_auto.json"):
    """Save configuration to file"""

    recommended_models = generate_recommended_config(device_caps, common_caps, quantization_type, models)

    config = {
        "auto_generated": True,
        "generation_info": {
            "detected_devices": list(device_caps.keys()),
            "common_capabilities": list(common_caps),
            "selected_quantization": quantization_type,
            "total_models_found": len(models),
            "recommended_models": len(recommended_models)
        },
        "models": recommended_models,
        "benchmark_config": {
            "test_prompts": [
                "What is artificial intelligence?",
                "Explain machine learning in simple terms.",
                "What are neural networks?",
                "Describe deep learning.",
                "What is the difference between AI and ML?"
            ],
            "generation_config": {
                "max_new_tokens": 100,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True
            },
            "devices_to_test": list(device_caps.keys()),
            "run_warmup": True,
            "cache_dir": "./ov_cache"
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Configuration saved to: {output_file}")

    return config

def print_summary(config):
    """Print configuration summary"""
    print("\n" + "=" * 80)
    print("📊 Configuration Summary")
    print("=" * 80)

    info = config['generation_info']

    print(f"\n🖥️  Detected Devices: {', '.join(info['detected_devices'])}")
    print(f"🎯 Common Capabilities: {', '.join(info['common_capabilities'])}")
    print(f"⚡ Selected Quantization: {info['selected_quantization']}")
    print(f"📦 Total Models Found: {info['total_models_found']}")
    print(f"✅ Recommended Models: {info['recommended_models']}")

    print(f"\n📋 Recommended Model List:")
    print(f"{'Status':<6} {'Model Name':<40} {'Size':<8} {'Downloads':<12} {'Devices'}")
    print("-" * 100)

    for model in config['models']:
        status = "🟢 ON" if model['enabled'] else "⚪ OFF"
        name = model['name'][:38]
        size = model['size']
        downloads = f"{model['downloads']:,}"
        devices = ', '.join(model['recommended_devices'])
        print(f"{status:<8} {name:<40} {size:<8} {downloads:<12} {devices}")

    enabled_count = sum(1 for m in config['models'] if m['enabled'])
    print(f"\n💡 {enabled_count} models enabled by default, ready to benchmark!")

def main():
    """Main function"""
    print("=" * 80)
    print("🤖 OpenVINO Smart Model Selector")
    print("   Automatically select optimal models based on device capabilities")
    print("=" * 80)

    # 1. Query device capabilities via OpenVINO Python API
    try:
        device_caps = query_device_capabilities()
    except Exception as e:
        print(f"❌ Failed to query device capabilities: {e}")
        print(f"\n💡 Solution: Install OpenVINO Python: pip install openvino>=2025.0.0")
        return 1

    if not device_caps:
        print("❌ No available devices detected")
        return 1

    # 2. Find common capabilities
    common_caps = find_common_capabilities(device_caps)

    if not common_caps:
        print("❌ No common optimization capabilities across devices")
        return 1

    # 3. Determine best quantization format
    quantization_name, quantization_suffix = determine_best_quantization(common_caps)

    # 4. Search models on HuggingFace
    models = search_models_on_huggingface(quantization_suffix, max_results=30)

    if not models:
        print(f"❌ No {quantization_name} models found")
        return 1

    print(f"\n✅ Found {len(models)} {quantization_name} models")

    # Display top 10 models
    print(f"\n📦 Top 10 Popular Models:")
    for idx, model in enumerate(models[:10], 1):
        print(f"   {idx:2d}. {model['model_id']:<60} (⬇️  {model['downloads']:,})")

    # 5. Generate and save configuration
    config = save_config(device_caps, common_caps, quantization_name, models)

    # 6. Print summary
    print_summary(config)

    # 7. Usage instructions
    print("\n" + "=" * 80)
    print("🚀 Next Steps")
    print("=" * 80)
    if sys.platform == 'win32':
        print("""
1. View generated configuration:
   type benchmark_auto.json

2. Edit if needed:
   notepad benchmark_auto.json

3. Run benchmark:
   copy benchmark_auto.json benchmark.json
   python benchmark_devices.py
""")
    else:
        print("""
1. View generated configuration:
   cat benchmark_auto.json

2. Edit if needed:
   nano benchmark_auto.json

3. Run benchmark:
   cp benchmark_auto.json benchmark.json
   python benchmark_devices.py
""")
    print("💡 Tip: Models are automatically selected based on your device capabilities!")

    return 0

if __name__ == "__main__":
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
