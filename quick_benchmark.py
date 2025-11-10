#!/usr/bin/env python3
"""
Quick Benchmark Script - Test a Single Model on All Devices

This script tests pre-converted OpenVINO models from HuggingFace.
Models from OpenVINO/* namespace are already converted to OpenVINO IR format
with INT4/INT8/FP16 quantization - NO conversion needed!

Usage:
    python3 quick_benchmark.py <model_id>
    
Examples:
    python3 quick_benchmark.py OpenVINO/Phi-3.5-vision-instruct-int8-ov
    python3 quick_benchmark.py OpenVINO/TinyLlama-1.1B-Chat-v1.0-int8-ov
    python3 quick_benchmark.py OpenVINO/Mistral-7B-Instruct-v0.3-int8-ov

Features:
    - Tests on CPU, GPU, and NPU automatically
    - Runs 5 diverse test prompts
    - Generates HTML report (benchmark_report.html)
    - Uses temporary config file (no modification to benchmark.json)
    - No manual configuration needed!

Note:
    All OpenVINO/* models are pre-converted and quantized.
    You do NOT need to run optimum-cli or any conversion tools.
"""

import sys
import json
import tempfile
import subprocess
import os
from pathlib import Path


def extract_model_info(model_id):
    """
    Extract model name, size, and quantization from HuggingFace model ID
    
    Args:
        model_id: HuggingFace model ID (e.g., OpenVINO/Phi-3.5-vision-instruct-int8-ov)
    
    Returns:
        tuple: (model_name, model_size, quantization)
    """
    # Extract the model name from the ID
    # e.g., "OpenVINO/Phi-3.5-vision-instruct-int8-ov" -> "Phi-3.5-vision-instruct"
    parts = model_id.split('/')
    if len(parts) >= 2:
        model_name = parts[1]
        
        # Detect quantization from suffix
        quantization = "int8"  # default
        if '-int4-ov' in model_name:
            quantization = "int4"
        elif '-int8-ov' in model_name:
            quantization = "int8"
        elif '-fp16-ov' in model_name:
            quantization = "fp16"
        elif '-fp32-ov' in model_name:
            quantization = "fp32"
        
        # Remove -int8-ov, -int4-ov, -fp16-ov suffixes
        for suffix in ['-int8-ov', '-int4-ov', '-fp16-ov', '-fp32-ov']:
            if model_name.endswith(suffix):
                model_name = model_name[:-len(suffix)]
                break
    else:
        model_name = model_id
        quantization = "int8"
    
    # Estimate model size from name
    # Common patterns: TinyLlama-1.1B, Phi-3.5, Mistral-7B, etc.
    import re
    
    # Try to find size pattern like "1.1B", "7B", "3.8B"
    size_match = re.search(r'(\d+\.?\d*)[Bb]', model_name)
    if size_match:
        size_str = size_match.group(1)
        model_size = f"{size_str}B"
    else:
        # Try to estimate from known model names
        lower_name = model_name.lower()
        if 'tiny' in lower_name:
            model_size = "1.1B"
        elif 'phi-3.5' in lower_name or 'phi-3' in lower_name:
            model_size = "3.8B"
        elif 'mistral' in lower_name:
            model_size = "7B"
        elif 'qwen2.5-1.5b' in lower_name:
            model_size = "1.5B"
        elif 'qwen' in lower_name:
            model_size = "1.8B"
        else:
            model_size = "Unknown"
    
    return model_name, model_size, quantization


def create_temp_config(model_id):
    """
    Create a temporary JSON config file for benchmark_devices.py
    
    Args:
        model_id: HuggingFace model ID
    
    Returns:
        str: Path to temporary JSON file
    """
    model_name, model_size, quantization = extract_model_info(model_id)
    
    # Five diverse test questions
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short poem about artificial intelligence.",
        "What are the main differences between Python and JavaScript?",
        "Describe the process of photosynthesis."
    ]
    
    config = {
        "models": [
            {
                "name": model_name,
                "model_id": model_id,
                "size": model_size,
                "quantization": quantization,
                "recommended_devices": ["CPU", "GPU", "NPU"],
                "description": f"Quick benchmark test for {model_name}",
                "enabled": True
            }
        ],
        "benchmark_config": {
            "test_prompts": prompts,  # 5 diverse test prompts
            "generation_config": {
                "max_new_tokens": 50,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "do_sample": True
            },
            "devices_to_test": ["CPU", "GPU", "NPU"],
            "run_warmup": True,
            "cache_dir": "./ov_cache"
        }
    }
    
    # Create temporary file
    temp_fd, temp_path = tempfile.mkstemp(suffix='.json', prefix='quick_benchmark_')
    
    # Write config to temp file
    with os.fdopen(temp_fd, 'w') as f:
        json.dump(config, f, indent=2)
    
    return temp_path


def run_benchmark(config_path):
    """
    Run benchmark_devices.py with the temporary config
    
    Args:
        config_path: Path to temporary config JSON file
    
    Returns:
        int: Return code from benchmark script
    """
    print(f"\n{'='*80}")
    print(f"🚀 Running Benchmark")
    print(f"{'='*80}\n")
    
    # Run benchmark_devices.py with the temp config
    benchmark_script = Path(__file__).parent / "benchmark_devices.py"
    
    if not benchmark_script.exists():
        print(f"❌ Error: benchmark_devices.py not found at {benchmark_script}")
        return 1
    
    print(f"📝 Using config: {config_path}")
    
    # Run the benchmark with --config parameter
    result = subprocess.run(
        [sys.executable, str(benchmark_script), '--config', config_path],
        cwd=benchmark_script.parent
    )
    
    return result.returncode


def main():
    """Main entry point"""
    
    # Print header
    print(f"\n{'='*80}")
    print(f"🎯 Quick Benchmark - Single Model Testing")
    print(f"{'='*80}\n")
    
    # Check arguments
    if len(sys.argv) < 2:
        print("❌ Error: Model ID required")
        print("\nUsage:")
        print(f"  python3 {sys.argv[0]} <model_id>")
        print("\nExamples:")
        print(f"  python3 {sys.argv[0]} OpenVINO/Phi-3.5-vision-instruct-int8-ov")
        print(f"  python3 {sys.argv[0]} OpenVINO/TinyLlama-1.1B-Chat-v1.0-int8-ov")
        print(f"  python3 {sys.argv[0]} OpenVINO/Mistral-7B-Instruct-v0.3-int8-ov")
        return 1
    
    model_id = sys.argv[1]
    print(f"📦 Model: {model_id}")
    
    # Extract model info
    model_name, model_size, quantization = extract_model_info(model_id)
    print(f"📝 Name: {model_name}")
    print(f"📏 Size: {model_size}")
    print(f"� Quantization: {quantization}")
    print(f"�🖥️  Devices: CPU, GPU, NPU")
    print(f"❓ Questions: 5 diverse prompts")
    
    # Create temporary config
    print(f"\n{'='*80}")
    print(f"📝 Creating Temporary Configuration")
    print(f"{'='*80}\n")
    
    temp_config = create_temp_config(model_id)
    print(f"✅ Config created: {temp_config}")
    
    # Show config contents
    with open(temp_config, 'r') as f:
        config_data = json.load(f)
        print(f"\nConfiguration:")
        print(json.dumps(config_data, indent=2))
    
    try:
        # Run benchmark
        return_code = run_benchmark(temp_config)
        
        if return_code == 0:
            print(f"\n{'='*80}")
            print(f"✅ Benchmark Completed Successfully!")
            print(f"{'='*80}\n")
            print(f"📊 Report: benchmark_report.html")
            print(f"📄 Results: benchmark_results.json")
            print(f"\nOpen benchmark_report.html in your browser to view results.")
        else:
            print(f"\n{'='*80}")
            print(f"⚠️  Benchmark completed with errors (exit code: {return_code})")
            print(f"{'='*80}\n")
        
        return return_code
    
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_config)
            print(f"\n🗑️  Cleaned up temporary config: {temp_config}")
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
