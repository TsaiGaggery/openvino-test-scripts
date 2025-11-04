#!/usr/bin/env python3
# benchmark_devices_improved.py
"""
Multi-Model Device Performance Benchmark - NPU Optimized Version
Compare inference performance across CPU, GPU, and NPU for multiple LLM models
"""
import os
import sys
import time
import json
import statistics
import signal
from pathlib import Path
from datetime import datetime
from huggingface_hub import snapshot_download
import openvino as ov
import openvino_genai as ov_genai

# IMPORTANT: DO NOT clear NPU environment variables!
# Commented out to preserve NPU optimizations
# for key in list(os.environ.keys()):
#     if 'NPU' in key or 'OV_NPU' in key:
#         del os.environ[key]

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Operation timed out")

def safe_load_pipeline(model_dir, device, timeout_seconds=300):
    """Safely load pipeline with timeout protection (Linux only)"""
    if sys.platform != 'linux':
        # On non-Linux, just try direct load
        return ov_genai.LLMPipeline(model_dir, device)
    
    # Set up timeout signal (Linux only)
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        pipe = ov_genai.LLMPipeline(model_dir, device)
        signal.alarm(0)  # Cancel alarm
        return pipe
    except TimeoutError:
        signal.alarm(0)  # Cancel alarm
        raise Exception(f"Model loading timed out after {timeout_seconds}s - likely a compilation error")
    finally:
        signal.signal(signal.SIGALRM, old_handler)

def load_benchmark_config(config_path="benchmark.json"):
    """Load benchmark configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"❌ Error: {config_path} not found!")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing {config_path}: {e}")
        sys.exit(1)

def download_model(model_id, model_name):
    """Download model from HuggingFace"""
    print(f"\n📥 Downloading {model_name}...")
    safe_name = model_name.replace(" ", "_").replace("-", "_").lower()
    model_dir = Path(f"models/{safe_name}")
    
    try:
        local_dir = snapshot_download(repo_id=model_id, local_dir=str(model_dir))
        print(f"✅ Model downloaded to: {local_dir}")
        return local_dir
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return None

def configure_device_optimizations(core, device, cache_dir):
    """Apply device-specific optimizations"""
    try:
        if device == 'NPU':
            # NPU-specific optimizations
            print(f"  Applying NPU optimizations...")
            core.set_property('NPU', {
                'CACHE_DIR': str(cache_dir),
                'PERFORMANCE_HINT': 'LATENCY',
                'NPU_COMPILATION_MODE_PARAMS': 'compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm'
            })
        elif device == 'GPU':
            # GPU optimizations
            core.set_property('GPU', {
                'CACHE_DIR': str(cache_dir),
                'PERFORMANCE_HINT': 'LATENCY'
            })
        elif device == 'CPU':
            # CPU optimizations
            core.set_property('CPU', {
                'PERFORMANCE_HINT': 'LATENCY'
            })
        print(f"  ✅ Device optimizations applied")
    except Exception as e:
        print(f"  ⚠️  Could not apply all optimizations: {e}")

def benchmark_model_device(model_dir, device, test_prompts, gen_config, cache_dir, core, model_size):
    """Benchmark a single model on a single device"""
    result = {
        'success': False,
        'load_time': 0,
        'generation_times': [],
        'token_counts': [],
        'avg_time': 0,
        'avg_tokens': 0,
        'avg_speed': 0,
        'total_time': 0,
        'total_tokens': 0,
        'error': None,
        'cache_status': 'unknown',
        'skipped': False
    }
    
    # Known problematic model patterns for NPU
    model_name = Path(model_dir).name.lower()
    npu_problematic_patterns = ['qwen3', 'gemma-2b']
    
    if device == 'NPU' and any(pattern in model_name for pattern in npu_problematic_patterns):
        print(f"  ⚠️  Known NPU compatibility issue with this model")
        print(f"  💡 Skipping NPU test for this model to prevent crash")
        print(f"  💡 This model will be tested on other devices (GPU/CPU)")
        result['error'] = 'Known NPU compatibility issue - skipped to prevent crash'
        result['skipped'] = True
        return result
    
    try:
        # Check for existing cache
        cache_exists = any(cache_dir.glob(f"*{device.lower()}*"))
        result['cache_status'] = 'hit' if cache_exists else 'miss'
        
        if cache_exists:
            print(f"  ℹ️  Cache found for {device} - load should be fast")
        else:
            print(f"  ℹ️  No cache for {device} - first load will compile model")
        
        # Configure device
        configure_device_optimizations(core, device, cache_dir)
        
        # Load model with timeout protection
        print(f"  Loading model to {device}...")
        load_start = time.time()
        
        try:
            pipe = safe_load_pipeline(model_dir, device, timeout_seconds=300)
            result['load_time'] = time.time() - load_start
        except Exception as load_error:
            load_error_msg = str(load_error)
            if 'timed out' in load_error_msg.lower():
                raise Exception(f"Model loading timed out - likely NPU compilation error. Try a different model.")
            else:
                raise  # Re-raise other exceptions
        
        if result['cache_status'] == 'miss' and result['load_time'] > 10:
            print(f"  ⚠️  Long load time ({result['load_time']:.1f}s) - model compiled and cached")
            print(f"     Next run should be much faster!")
        else:
            print(f"  ✅ Loaded in {result['load_time']:.1f}s")
        
        # Warmup
        print(f"  Warming up {device}...")
        _ = pipe.generate("Hello", gen_config)
        print(f"  ✅ Warmup complete")
        
        # Run benchmark
        print(f"  Running {len(test_prompts)} test prompts on {device}...")
        
        generation_times = []
        token_counts = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"    [{i}/{len(test_prompts)}] {prompt[:45]}...")
            
            start = time.time()
            response = pipe.generate(prompt, gen_config)
            elapsed = time.time() - start
            
            tokens = len(response.split())
            generation_times.append(elapsed)
            token_counts.append(tokens)
            
            speed = tokens / elapsed if elapsed > 0 else 0
            print(f"         {elapsed:.2f}s | {tokens} tokens | {speed:.1f} tok/s")
        
        # Calculate statistics
        result['generation_times'] = generation_times
        result['token_counts'] = token_counts
        result['avg_time'] = statistics.mean(generation_times)
        result['avg_tokens'] = statistics.mean(token_counts)
        result['total_tokens'] = sum(token_counts)
        result['total_time'] = sum(generation_times)
        result['avg_speed'] = result['total_tokens'] / result['total_time'] if result['total_time'] > 0 else 0
        result['success'] = True
        
        # Performance assessment for NPU
        if device == 'NPU':
            expected_speed = get_expected_npu_speed(model_size)
            if result['avg_speed'] < expected_speed * 0.7:
                print(f"  ⚠️  NPU speed ({result['avg_speed']:.1f} tok/s) is below expected ({expected_speed:.1f} tok/s)")
                print(f"     Consider running again for cache benefits or try smaller models")
            else:
                print(f"  ✅ {device} completed: {result['avg_speed']:.1f} tok/s (good performance)")
        else:
            print(f"  ✅ {device} completed: {result['avg_speed']:.1f} tok/s")
        
    except Exception as e:
        result['error'] = str(e)
        error_msg = str(e)
        
        # Provide specific guidance for NPU compilation errors
        if device == 'NPU':
            if 'Channels count' in error_msg or 'LLVM ERROR' in error_msg or 'vpux-compiler' in error_msg or 'timed out' in error_msg.lower():
                print(f"  ❌ NPU Compilation Error: Model incompatible with NPU")
                print(f"     This model architecture is not supported by the NPU compiler.")
                print(f"     💡 Solutions:")
                print(f"        1. Try a different model (TinyLlama, Phi-3, or StableLM work well)")
                print(f"        2. Use GPU or CPU for this model instead")
                print(f"        3. Check for updated model version at huggingface.co/OpenVINO")
                print(f"     ℹ️  Continuing with remaining tests...")
            else:
                print(f"  ❌ Error on {device}: {error_msg[:200]}")
        else:
            print(f"  ❌ Error on {device}: {error_msg[:200]}")
    
    return result

def get_expected_npu_speed(model_size):
    """Get expected NPU speed based on model size"""
    size_map = {
        '1.1B': 30,
        '1.7B': 25,
        '3.8B': 18,
        '4B': 17,
        '7B': 12,
        '8B': 11
    }
    return size_map.get(model_size, 15)

def print_model_summary(model_name, device_results, model_size):
    """Print summary for a single model across devices"""
    print(f"\n{'=' * 80}")
    print(f"📊 {model_name} ({model_size}) - Summary")
    print('=' * 80)
    
    successful_results = {dev: res for dev, res in device_results.items() if res['success']}
    skipped_results = {dev: res for dev, res in device_results.items() if res.get('skipped', False)}
    
    if not successful_results:
        print("❌ No successful benchmarks for this model")
        if skipped_results:
            print("\n⚠️  Skipped tests:")
            for dev, res in skipped_results.items():
                print(f"   • {dev}: {res['error']}")
        return
    
    print(f"\n{'Device':<10} {'Load (s)':<12} {'Cache':<10} {'Avg Time (s)':<15} {'Avg Speed':<15} {'Total Time (s)':<15}")
    print("-" * 90)
    
    sorted_devices = sorted(successful_results.items(), 
                          key=lambda x: x[1]['avg_speed'], 
                          reverse=True)
    
    for device, data in sorted_devices:
        cache_status = data.get('cache_status', 'unknown')
        print(f"{device:<10} {data['load_time']:>10.1f}  {cache_status:<10} {data['avg_time']:>13.2f}  "
              f"{data['avg_speed']:>13.1f} tok/s  {data['total_time']:>13.1f}")
    
    # Show skipped tests
    if skipped_results:
        print("\n⚠️  Skipped:")
        for dev in skipped_results:
            print(f"   • {dev}: Known compatibility issue")
    
    # Winner and analysis
    fastest_device = sorted_devices[0][0]
    fastest_speed = sorted_devices[0][1]['avg_speed']
    print(f"\n🏆 Fastest: {fastest_device} at {fastest_speed:.1f} tok/s")
    
    # NPU-specific analysis
    if 'NPU' in successful_results:
        npu_result = successful_results['NPU']
        expected_speed = get_expected_npu_speed(model_size)
        
        print(f"\n💡 NPU Analysis:")
        print(f"   Actual speed: {npu_result['avg_speed']:.1f} tok/s")
        print(f"   Expected speed: {expected_speed:.1f} tok/s")
        
        if npu_result['avg_speed'] >= expected_speed * 0.9:
            print(f"   ✅ NPU performing well!")
        elif npu_result['avg_speed'] >= expected_speed * 0.7:
            print(f"   ⚠️  NPU performing moderately - check cache and run again")
        else:
            print(f"   ⚠️  NPU underperforming - model may be too large for optimal NPU performance")
            
        if npu_result.get('cache_status') == 'miss' and npu_result['load_time'] > 10:
            print(f"   ℹ️  First run detected - run benchmark again for better performance")

def print_cross_model_comparison(all_results, models_info):
    """Print comparison across all models and devices"""
    print("\n" + "=" * 100)
    print("🔍 CROSS-MODEL COMPARISON")
    print("=" * 100)
    
    # Collect all successful results
    comparison_data = []
    for model_name, device_results in all_results.items():
        model_info = next((m for m in models_info if m['name'] == model_name), {})
        model_size = model_info.get('size', 'Unknown')
        
        for device, result in device_results.items():
            if result['success']:
                comparison_data.append({
                    'model': model_name,
                    'size': model_size,
                    'device': device,
                    'speed': result['avg_speed'],
                    'load_time': result['load_time'],
                    'avg_time': result['avg_time'],
                    'cache': result.get('cache_status', 'unknown')
                })
    
    if not comparison_data:
        print("❌ No successful benchmarks to compare")
        return
    
    # Sort by speed
    comparison_data.sort(key=lambda x: x['speed'], reverse=True)
    
    # Print table
    print(f"\n{'Model':<30} {'Size':<8} {'Device':<10} {'Speed':<18} {'Load (s)':<12} {'Cache':<10}")
    print("-" * 100)
    
    for item in comparison_data:
        print(f"{item['model']:<30} {item['size']:<8} {item['device']:<10} "
              f"{item['speed']:>15.1f} tok/s  {item['load_time']:>10.1f}  {item['cache']:<10}")
    
    # Top performers
    print("\n" + "=" * 100)
    print("🏆 TOP PERFORMERS")
    print("=" * 100)
    
    print(f"\n1️⃣  Overall Fastest: {comparison_data[0]['model']} on {comparison_data[0]['device']}")
    print(f"    Speed: {comparison_data[0]['speed']:.1f} tok/s")
    
    # Best per device
    print("\n📊 Best Model per Device:")
    devices = set(item['device'] for item in comparison_data)
    for device in sorted(devices):
        device_data = [item for item in comparison_data if item['device'] == device]
        if device_data:
            best = device_data[0]
            print(f"   {device}: {best['model']} ({best['size']}) - {best['speed']:.1f} tok/s")
    
    # Best per model
    print("\n📊 Best Device per Model:")
    models = set(item['model'] for item in comparison_data)
    for model in sorted(models):
        model_data = [item for item in comparison_data if item['model'] == model]
        if model_data:
            best = model_data[0]
            print(f"   {model}: {best['device']} - {best['speed']:.1f} tok/s")
    
    # NPU-specific insights
    npu_data = [item for item in comparison_data if item['device'] == 'NPU']
    if npu_data:
        print("\n💻 NPU-Specific Insights:")
        avg_npu_speed = statistics.mean([item['speed'] for item in npu_data])
        print(f"   Average NPU speed: {avg_npu_speed:.1f} tok/s")
        
        # Compare to GPU
        gpu_data = [item for item in comparison_data if item['device'] == 'GPU']
        if gpu_data:
            avg_gpu_speed = statistics.mean([item['speed'] for item in gpu_data])
            ratio = (avg_gpu_speed / avg_npu_speed) if avg_npu_speed > 0 else 0
            print(f"   Average GPU speed: {avg_gpu_speed:.1f} tok/s")
            print(f"   GPU/NPU ratio: {ratio:.2f}x")
            
            if ratio > 2:
                print(f"   ⚠️  GPU significantly faster - your models may be too large for NPU")
                print(f"   💡 Try smaller models (1-2B) for better NPU performance")

def print_device_analysis(all_results):
    """Analyze device performance across all models"""
    print("\n" + "=" * 100)
    print("💻 DEVICE PERFORMANCE ANALYSIS")
    print("=" * 100)
    
    device_stats = {}
    
    for model_name, device_results in all_results.items():
        for device, result in device_results.items():
            if result['success']:
                if device not in device_stats:
                    device_stats[device] = {
                        'speeds': [],
                        'load_times': [],
                        'models': [],
                        'cache_hits': 0,
                        'cache_misses': 0
                    }
                device_stats[device]['speeds'].append(result['avg_speed'])
                device_stats[device]['load_times'].append(result['load_time'])
                device_stats[device]['models'].append(model_name)
                
                if result.get('cache_status') == 'hit':
                    device_stats[device]['cache_hits'] += 1
                elif result.get('cache_status') == 'miss':
                    device_stats[device]['cache_misses'] += 1
    
    if not device_stats:
        print("No successful benchmarks")
        return
    
    for device in sorted(device_stats.keys()):
        stats = device_stats[device]
        print(f"\n{device}:")
        print(f"  Models tested: {len(stats['models'])}")
        print(f"  Avg speed: {statistics.mean(stats['speeds']):.1f} tok/s")
        print(f"  Speed range: {min(stats['speeds']):.1f} - {max(stats['speeds']):.1f} tok/s")
        print(f"  Avg load time: {statistics.mean(stats['load_times']):.1f}s")
        print(f"  Cache hits: {stats['cache_hits']}, misses: {stats['cache_misses']}")
        
        if len(stats['speeds']) > 1:
            print(f"  Speed std dev: {statistics.stdev(stats['speeds']):.1f} tok/s")
        
        # Device-specific recommendations
        if device == 'NPU':
            if stats['cache_misses'] > 0:
                print(f"  💡 Run benchmark again to benefit from cache (first compilation takes longer)")
            avg_speed = statistics.mean(stats['speeds'])
            if avg_speed < 15:
                print(f"  💡 NPU performs best with smaller models (1-2B parameters)")

def generate_html_report(all_results, enabled_models, devices_to_test, config, output_file="benchmark_report.html"):
    """Generate a comprehensive HTML report with charts and styling"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Prepare data for charts
    chart_data = {
        'models': [],
        'devices': devices_to_test,
        'speeds': {device: [] for device in devices_to_test},
        'load_times': {device: [] for device in devices_to_test}
    }
    
    comparison_data = []
    for model_name, device_results in all_results.items():
        model_info = next((m for m in enabled_models if m['name'] == model_name), {})
        model_size = model_info.get('size', 'Unknown')
        chart_data['models'].append(model_name)
        
        for device in devices_to_test:
            result = device_results.get(device, {})
            if result.get('success'):
                chart_data['speeds'][device].append(result['avg_speed'])
                chart_data['load_times'][device].append(result['load_time'])
                comparison_data.append({
                    'model': model_name,
                    'size': model_size,
                    'device': device,
                    'speed': result['avg_speed'],
                    'load_time': result['load_time'],
                    'avg_time': result['avg_time'],
                    'total_tokens': result['total_tokens'],
                    'cache': result.get('cache_status', 'unknown')
                })
            else:
                chart_data['speeds'][device].append(0)
                chart_data['load_times'][device].append(0)
    
    # Find best performers
    best_overall = max(comparison_data, key=lambda x: x['speed']) if comparison_data else None
    
    # Device statistics
    device_stats = {}
    for device in devices_to_test:
        device_results_list = [item for item in comparison_data if item['device'] == device]
        if device_results_list:
            speeds = [item['speed'] for item in device_results_list]
            device_stats[device] = {
                'avg_speed': statistics.mean(speeds),
                'min_speed': min(speeds),
                'max_speed': max(speeds),
                'count': len(speeds)
            }
    
    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenVINO Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }}
        
        header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        header .timestamp {{
            margin-top: 15px;
            font-size: 0.9em;
            opacity: 0.8;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .highlight-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }}
        
        .highlight-box h3 {{
            font-size: 1.3em;
            margin-bottom: 15px;
        }}
        
        .highlight-box .stat {{
            font-size: 2em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }}
        
        .metric-card h4 {{
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }}
        
        .metric-card .value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
        }}
        
        .metric-card .label {{
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }}
        
        thead {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}
        
        th {{
            padding: 15px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        
        tbody tr:hover {{
            background: #f8f9fa;
        }}
        
        tbody tr:last-child td {{
            border-bottom: none;
        }}
        
        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        
        .badge.success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .badge.warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .badge.fastest {{
            background: #ffd700;
            color: #333;
        }}
        
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 30px 0;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .device-cpu {{ color: #4CAF50; }}
        .device-gpu {{ color: #2196F3; }}
        .device-npu {{ color: #FF9800; }}
        
        footer {{
            background: #f8f9fa;
            padding: 20px 40px;
            text-align: center;
            color: #666;
            border-top: 1px solid #eee;
        }}
        
        .recommendations {{
            background: #e8f5e9;
            border-left: 4px solid #4caf50;
            padding: 20px;
            border-radius: 4px;
            margin: 20px 0;
        }}
        
        .recommendations h4 {{
            color: #2e7d32;
            margin-bottom: 10px;
        }}
        
        .recommendations ul {{
            list-style-position: inside;
            color: #1b5e20;
        }}
        
        .recommendations li {{
            margin: 8px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 OpenVINO Benchmark Report</h1>
            <div class="subtitle">Multi-Model Device Performance Analysis</div>
            <div class="timestamp">Generated: {timestamp}</div>
        </header>
        
        <div class="content">
"""
    
    # Best Performance Section
    if best_overall:
        html += f"""
            <section class="section">
                <div class="highlight-box">
                    <h3>🏆 Best Overall Performance</h3>
                    <div class="stat">{best_overall['model']}</div>
                    <div>on {best_overall['device']} device</div>
                    <div class="stat">{best_overall['speed']:.1f} tok/s</div>
                    <div>Model Size: {best_overall['size']}</div>
                </div>
            </section>
"""
    
    # Device Statistics
    if device_stats:
        html += """
            <section class="section">
                <h2>📊 Device Performance Summary</h2>
                <div class="metrics-grid">
"""
        for device, stats in device_stats.items():
            device_class = f"device-{device.lower()}"
            html += f"""
                    <div class="metric-card">
                        <h4 class="{device_class}">{device}</h4>
                        <div class="value">{stats['avg_speed']:.1f} tok/s</div>
                        <div class="label">Average Speed</div>
                        <div style="margin-top: 10px; font-size: 0.9em;">
                            <div>Range: {stats['min_speed']:.1f} - {stats['max_speed']:.1f} tok/s</div>
                            <div>Models tested: {stats['count']}</div>
                        </div>
                    </div>
"""
        html += """
                </div>
            </section>
"""
    
    # Detailed Results Table
    if comparison_data:
        comparison_data_sorted = sorted(comparison_data, key=lambda x: x['speed'], reverse=True)
        fastest_speed = comparison_data_sorted[0]['speed']
        
        html += """
            <section class="section">
                <h2>📈 Detailed Benchmark Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Size</th>
                            <th>Device</th>
                            <th>Speed (tok/s)</th>
                            <th>Load Time (s)</th>
                            <th>Avg Time (s)</th>
                            <th>Total Tokens</th>
                            <th>Cache</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        for item in comparison_data_sorted:
            fastest_badge = ' <span class="badge fastest">FASTEST</span>' if item['speed'] == fastest_speed else ''
            cache_badge_class = 'success' if item['cache'] == 'hit' else 'warning'
            device_class = f"device-{item['device'].lower()}"
            
            html += f"""
                        <tr>
                            <td><strong>{item['model']}</strong>{fastest_badge}</td>
                            <td>{item['size']}</td>
                            <td class="{device_class}"><strong>{item['device']}</strong></td>
                            <td><strong>{item['speed']:.1f}</strong></td>
                            <td>{item['load_time']:.1f}</td>
                            <td>{item['avg_time']:.2f}</td>
                            <td>{item['total_tokens']}</td>
                            <td><span class="badge {cache_badge_class}">{item['cache']}</span></td>
                        </tr>
"""
        html += """
                    </tbody>
                </table>
            </section>
"""
    
    # Charts Section
    html += """
            <section class="section">
                <h2>📊 Performance Charts</h2>
                
                <div class="chart-container">
                    <canvas id="speedChart"></canvas>
                </div>
                
                <div class="chart-container">
                    <canvas id="loadTimeChart"></canvas>
                </div>
            </section>
"""
    
    # Recommendations
    html += """
            <section class="section">
                <h2>💡 Recommendations</h2>
                <div class="recommendations">
                    <h4>General Guidelines:</h4>
                    <ul>
                        <li>Smaller models (1-4B parameters) are faster and work well on NPU</li>
                        <li>Larger models (7-8B parameters) have better quality but prefer GPU</li>
                        <li>NPU offers best power efficiency for mobile/battery use cases</li>
                        <li>GPU provides best raw performance for desktop/workstation</li>
                        <li>CPU is most compatible, good for servers and cloud deployment</li>
                        <li>First run compiles and caches models - second run is much faster!</li>
                    </ul>
                </div>
"""
    
    # NPU-specific recommendations
    npu_data = [item for item in comparison_data if item['device'] == 'NPU']
    if npu_data:
        avg_npu_speed = statistics.mean([item['speed'] for item in npu_data])
        html += f"""
                <div class="recommendations" style="background: #fff3cd; border-color: #ff9800;">
                    <h4 style="color: #f57c00;">NPU-Specific Insights:</h4>
                    <ul style="color: #e65100;">
                        <li>Average NPU speed: {avg_npu_speed:.1f} tok/s</li>
"""
        
        if any(item['cache'] == 'miss' for item in npu_data):
            html += """
                        <li>⚠️ First run detected - run benchmark again for up to 10x faster load times</li>
"""
        
        if avg_npu_speed < 15:
            html += """
                        <li>💡 Consider testing smaller models (1-4B) for better NPU performance</li>
"""
        
        html += """
                    </ul>
                </div>
"""
    
    html += """
            </section>
            
            <section class="section">
                <h2>⚙️ Test Configuration</h2>
                <table>
                    <tbody>
"""
    
    gen_config = config['benchmark_config']['generation_config']
    html += f"""
                        <tr><td><strong>Test Prompts</strong></td><td>{len(config['benchmark_config']['test_prompts'])}</td></tr>
                        <tr><td><strong>Max New Tokens</strong></td><td>{gen_config['max_new_tokens']}</td></tr>
                        <tr><td><strong>Temperature</strong></td><td>{gen_config['temperature']}</td></tr>
                        <tr><td><strong>Top P</strong></td><td>{gen_config['top_p']}</td></tr>
                        <tr><td><strong>Sampling</strong></td><td>{'Yes' if gen_config['do_sample'] else 'No'}</td></tr>
                        <tr><td><strong>Devices Tested</strong></td><td>{', '.join(devices_to_test)}</td></tr>
                        <tr><td><strong>Models Tested</strong></td><td>{len(enabled_models)}</td></tr>
"""
    
    html += """
                    </tbody>
                </table>
            </section>
        </div>
        
        <footer>
            <p>Generated by OpenVINO Benchmark Tool</p>
            <p>OpenVINO™ is a trademark of Intel Corporation</p>
        </footer>
    </div>
    
    <script>
"""
    
    # Add chart data and configuration
    html += f"""
        // Speed Chart
        const speedCtx = document.getElementById('speedChart').getContext('2d');
        const speedChart = new Chart(speedCtx, {{
            type: 'bar',
            data: {{
                labels: {json.dumps(chart_data['models'])},
                datasets: [
"""
    
    colors = {'CPU': '#4CAF50', 'GPU': '#2196F3', 'NPU': '#FF9800'}
    for device in devices_to_test:
        color = colors.get(device, '#666')
        html += f"""
                    {{
                        label: '{device}',
                        data: {json.dumps(chart_data['speeds'][device])},
                        backgroundColor: '{color}',
                        borderColor: '{color}',
                        borderWidth: 2
                    }},
"""
    
    html += """
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Average Speed Comparison (tokens/second)',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Speed (tokens/sec)'
                        }
                    }
                }
            }
        });
        
        // Load Time Chart
        const loadTimeCtx = document.getElementById('loadTimeChart').getContext('2d');
        const loadTimeChart = new Chart(loadTimeCtx, {
            type: 'bar',
            data: {
                labels: """ + json.dumps(chart_data['models']) + """,
                datasets: [
"""
    
    for device in devices_to_test:
        color = colors.get(device, '#666')
        html += f"""
                    {{
                        label: '{device}',
                        data: {json.dumps(chart_data['load_times'][device])},
                        backgroundColor: '{color}',
                        borderColor: '{color}',
                        borderWidth: 2
                    }},
"""
    
    html += """
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Load Time Comparison (seconds)',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Load Time (seconds)'
                        }
                    }
                }
            }
        });
    </script>
</body>
</html>
"""
    
    # Write HTML file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ HTML report generated: {output_file}")
    return output_file

def main():
    print("=" * 100)
    print("🚀 OpenVINO Multi-Model Device Benchmark (NPU Optimized)")
    print("=" * 100)
    
    # Load configuration
    print("\n[1/6] Loading configuration...")
    config = load_benchmark_config()
    
    # Get enabled models
    enabled_models = [m for m in config['models'] if m.get('enabled', False)]
    
    if not enabled_models:
        print("❌ No models enabled in benchmark.json!")
        print("   Set 'enabled': true for at least one model")
        sys.exit(1)
    
    print(f"✅ Found {len(enabled_models)} enabled model(s):")
    for model in enabled_models:
        print(f"   • {model['name']} ({model['size']}) - {model['quantization']}")
        
        # Warn about large models on NPU
        size_val = float(model['size'].replace('B', ''))
        if 'NPU' in config['benchmark_config']['devices_to_test'] and size_val > 5:
            print(f"     ⚠️  Note: {model['size']} may not be optimal for NPU (prefer <5B)")
    
    # Setup cache
    cache_dir = Path(config['benchmark_config']['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Cache directory: {cache_dir}")
    
    # Check if cache exists
    existing_cache = list(cache_dir.glob("*"))
    if existing_cache:
        print(f"  ℹ️  Found {len(existing_cache)} cached files - should see faster load times")
    else:
        print(f"  ℹ️  No cache found - first run will compile models (slower)")
    
    # Check available devices
    print("\n[2/6] Checking available devices...")
    core = ov.Core()
    available_devices = core.available_devices
    print(f"Available devices: {available_devices}")
    
    # Determine which devices to test
    requested_devices = config['benchmark_config']['devices_to_test']
    devices_to_test = []
    
    if 'CPU' in available_devices and 'CPU' in requested_devices:
        devices_to_test.append('CPU')
    if any('GPU' in d for d in available_devices) and 'GPU' in requested_devices:
        devices_to_test.append('GPU')
    if any('NPU' in d for d in available_devices) and 'NPU' in requested_devices:
        devices_to_test.append('NPU')
        print("  ℹ️  NPU detected - optimizations will be applied")
    
    print(f"✅ Will test devices: {', '.join(devices_to_test)}")
    
    if not devices_to_test:
        print("❌ No devices available for testing!")
        sys.exit(1)
    
    # Load test configuration
    test_prompts = config['benchmark_config']['test_prompts']
    gen_config_dict = config['benchmark_config']['generation_config']
    
    gen_config = ov_genai.GenerationConfig()
    gen_config.max_new_tokens = gen_config_dict['max_new_tokens']
    gen_config.temperature = gen_config_dict['temperature']
    gen_config.top_p = gen_config_dict['top_p']
    gen_config.do_sample = gen_config_dict['do_sample']
    
    print(f"✅ Test configuration:")
    print(f"   • {len(test_prompts)} test prompts")
    print(f"   • Max tokens: {gen_config.max_new_tokens}")
    print(f"   • Temperature: {gen_config.temperature}")
    
    # Store all results
    all_results = {}
    
    # Download and benchmark each model
    print("\n[3/6] Downloading and benchmarking models...")
    print("=" * 100)
    
    for model_idx, model_info in enumerate(enabled_models, 1):
        print(f"\n{'=' * 100}")
        print(f"📦 MODEL {model_idx}/{len(enabled_models)}: {model_info['name']}")
        print(f"   Size: {model_info['size']} | Quantization: {model_info['quantization']}")
        print(f"   Description: {model_info['description']}")
        print('=' * 100)
        
        # Download model
        model_dir = download_model(model_info['model_id'], model_info['name'])
        if not model_dir:
            print(f"⚠️  Skipping {model_info['name']} due to download error")
            continue
        
        # Benchmark on each device
        model_results = {}
        
        for device in devices_to_test:
            print(f"\n{'─' * 80}")
            print(f"🔧 Testing {model_info['name']} on {device}")
            print('─' * 80)
            
            result = benchmark_model_device(
                model_dir, device, test_prompts, gen_config, cache_dir, core, model_info['size']
            )
            model_results[device] = result
        
        all_results[model_info['name']] = model_results
        
        # Print summary for this model
        print_model_summary(model_info['name'], model_results, model_info['size'])
    
    # Print comprehensive comparisons
    print("\n" + "=" * 100)
    print("[4/6] COMPREHENSIVE RESULTS")
    print("=" * 100)
    
    print_cross_model_comparison(all_results, enabled_models)
    print_device_analysis(all_results)
    
    # Recommendations
    print("\n" + "=" * 100)
    print("[5/6] RECOMMENDATIONS")
    print("=" * 100)
    
    # Find best overall
    best_combo = None
    best_speed = 0
    
    for model_name, device_results in all_results.items():
        for device, result in device_results.items():
            if result['success'] and result['avg_speed'] > best_speed:
                best_speed = result['avg_speed']
                best_combo = (model_name, device)
    
    if best_combo:
        print(f"\n✨ Optimal Configuration:")
        print(f"   Model: {best_combo[0]}")
        print(f"   Device: {best_combo[1]}")
        print(f"   Speed: {best_speed:.1f} tok/s")
        print(f"\n   Code:")
        model_info = next((m for m in enabled_models if m['name'] == best_combo[0]), None)
        if model_info:
            print(f'   model_id = "{model_info["model_id"]}"')
            print(f'   device = "{best_combo[1]}"')
            print(f'   pipe = ov_genai.LLMPipeline(model_dir, device)')
    
    # NPU-specific recommendations
    npu_results = []
    for model_name, device_results in all_results.items():
        if 'NPU' in device_results and device_results['NPU']['success']:
            npu_results.append((model_name, device_results['NPU']))
    
    if npu_results:
        print("\n💡 NPU-Specific Recommendations:")
        best_npu = max(npu_results, key=lambda x: x[1]['avg_speed'])
        print(f"   Best model for NPU: {best_npu[0]} ({best_npu[1]['avg_speed']:.1f} tok/s)")
        
        # Check if first run
        if any(r[1].get('cache_status') == 'miss' for r in npu_results):
            print(f"   ⚠️  First run detected - run benchmark again for up to 10x faster load times")
        
        # Check if models are too large
        avg_npu_speed = statistics.mean([r[1]['avg_speed'] for r in npu_results])
        if avg_npu_speed < 15:
            print(f"   💡 Consider testing smaller models for better NPU performance:")
            print(f"      • TinyLlama-1.1B (optimal)")
            print(f"      • Qwen3-1.7B (very good)")
            print(f"      • Qwen3-4B (good)")
    
    print("\n💡 General Guidelines:")
    print("   • Smaller models (1-4B): Faster, good for NPU")
    print("   • Larger models (7-8B): Better quality, prefer GPU")
    print("   • NPU: Best power efficiency for mobile/battery use")
    print("   • GPU: Best raw performance for desktop/workstation")
    print("   • CPU: Most compatible, good for servers/cloud")
    print("   • First run compiles and caches - second run is much faster!")
    
    # Save results to JSON
    print("\n[6/6] Saving results...")
    results_file = "benchmark_results.json"
    
    output_data = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': config['benchmark_config'],
        'devices_tested': devices_to_test,
        'models_tested': [m['name'] for m in enabled_models],
        'results': {}
    }
    
    for model_name, device_results in all_results.items():
        output_data['results'][model_name] = {}
        for device, result in device_results.items():
            if result['success']:
                output_data['results'][model_name][device] = {
                    'load_time': result['load_time'],
                    'avg_time': result['avg_time'],
                    'avg_speed': result['avg_speed'],
                    'avg_tokens': result['avg_tokens'],
                    'total_time': result['total_time'],
                    'total_tokens': result['total_tokens'],
                    'cache_status': result.get('cache_status', 'unknown')
                }
    
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Results saved to {results_file}")
    
    # Generate HTML report
    print("\n📄 Generating HTML report...")
    html_file = generate_html_report(all_results, enabled_models, devices_to_test, config)
    print(f"✅ HTML report saved to {html_file}")
    print(f"   Open {html_file} in your browser to view the report")
    
    print("\n" + "=" * 100)
    print("✅ Benchmark Complete!")
    print("=" * 100)
    
    # Final NPU note
    if 'NPU' in devices_to_test and npu_results:
        print("\n💡 NPU Performance Note:")
        print("   If NPU seems slow, remember:")
        print("   1. Run benchmark twice - second run uses cache (much faster)")
        print("   2. NPU works best with models <5B parameters")
        print("   3. NPU optimizes for power efficiency, not raw speed")
        print("   4. For large models (7-8B), GPU will generally be faster")

if __name__ == "__main__":
    main()
