#!/usr/bin/env python3
"""
Test local OpenVINO model(s)

Usage:
    python test_local_model.py /path/to/model_directory [device]
    python test_local_model.py /path/to/models_folder [device]  # Tests all models in folder

Examples:
    # Test single model on NPU
    python test_local_model.py ~/models/Llama_3.1_NPU_INT4 NPU

    # Test single model on all devices
    python test_local_model.py ~/models/Llama_3.1_NPU_INT4

    # Test all models in folder on all devices (comprehensive comparison)
    python test_local_model.py ~/models

    # Test all models in folder on NPU only
    python test_local_model.py ~/models NPU
"""

import sys
import openvino_genai as ov_genai
from pathlib import Path
import time
import json
import statistics
from datetime import datetime


def generate_html_report(results, model_path, output_file=None):
    """Generate HTML report for local model testing"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    model_name = Path(model_path).name
    
    # Generate filename with model name and timestamp if not provided
    if output_file is None:
        timestamp_file = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_model_name = model_name.replace(" ", "_").replace("-", "_")
        output_file = f"local_model_report_{safe_model_name}_{timestamp_file}.html"
    
    # Calculate statistics
    successful_results = [r for r in results if r['success']]
    
    if not successful_results:
        print("⚠️ No successful tests to report")
        return None
    
    # Find best performer
    best_result = max(successful_results, key=lambda x: x['avg_speed'])
    
    # Extract model info
    model_info = successful_results[0].get('model_info', {})
    params = model_info.get('parameters', 'Unknown')
    quant = model_info.get('quantization', 'Unknown')
    arch = model_info.get('architecture', 'Unknown')
    
    # HTML template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local Model Test Report - {model_name}</title>
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
            max-width: 1200px;
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
        
        .info-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin-bottom: 20px;
        }}
        
        .info-box strong {{
            color: #667eea;
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
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧪 Local Model Test Report</h1>
            <div class="subtitle">{model_name}</div>
            <div class="timestamp">Generated: {timestamp}</div>
        </header>
        
        <div class="content">
            <section class="section">
                <div class="info-box">
                    <strong>📁 Model Path:</strong> {model_path}<br>
                    <strong>🏗️ Architecture:</strong> {arch}<br>
                    <strong>📊 Parameters:</strong> {params}<br>
                    <strong>🔢 Quantization:</strong> {quant}<br>
                    <strong>🖥️ Devices Tested:</strong> {len(successful_results)}<br>
                    <strong>✅ Successful Tests:</strong> {len(successful_results)}
                </div>
            </section>
            
            <section class="section">
                <div class="highlight-box">
                    <h3>🏆 Best Performance</h3>
                    <div class="stat">{best_result['device']}</div>
                    <div class="stat">{best_result['avg_speed']:.1f} tok/s</div>
                    <div>Load Time: {best_result['load_time']:.1f}s</div>
                </div>
            </section>
            
            <section class="section">
                <h2>📊 Device Performance Summary</h2>
                <div class="metrics-grid">
"""
    
    # Add metric cards for each device
    for result in sorted(successful_results, key=lambda x: x['avg_speed'], reverse=True):
        device_class = f"device-{result['device'].lower()}"
        html += f"""
                    <div class="metric-card">
                        <h4 class="{device_class}">{result['device']}</h4>
                        <div class="value">{result['avg_speed']:.1f} tok/s</div>
                        <div class="label">Average Speed</div>
                        <div style="margin-top: 10px; font-size: 0.9em;">
                            <div>Load: {result['load_time']:.1f}s</div>
                            <div>Avg Time: {result['avg_time']:.2f}s</div>
                        </div>
                    </div>
"""
    
    html += """
                </div>
            </section>
            
            <section class="section">
                <h2>📈 Detailed Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Device</th>
                            <th>Load Time (s)</th>
                            <th>Avg Time (s)</th>
                            <th>Speed (tok/s)</th>
                            <th>Total Tokens</th>
                            <th>Total Time (s)</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # Add table rows
    fastest_speed = best_result['avg_speed']
    for result in sorted(successful_results, key=lambda x: x['avg_speed'], reverse=True):
        fastest_badge = ' <span class="badge fastest">FASTEST</span>' if result['avg_speed'] == fastest_speed else ''
        device_class = f"device-{result['device'].lower()}"
        
        html += f"""
                        <tr>
                            <td class="{device_class}"><strong>{result['device']}</strong>{fastest_badge}</td>
                            <td>{result['load_time']:.1f}</td>
                            <td>{result['avg_time']:.2f}</td>
                            <td><strong>{result['avg_speed']:.1f}</strong></td>
                            <td>{result['total_tokens']}</td>
                            <td>{result['total_time']:.1f}</td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
            </section>
"""
    
    # Add charts if multiple devices tested
    if len(successful_results) > 1:
        devices = [r['device'] for r in successful_results]
        speeds = [r['avg_speed'] for r in successful_results]
        load_times = [r['load_time'] for r in successful_results]
        
        html += f"""
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
    
    html += """
        </div>
        
        <footer>
            <p>Generated by OpenVINO Local Model Test Tool</p>
            <p>OpenVINO™ is a trademark of Intel Corporation</p>
        </footer>
    </div>
"""
    
    # Add chart JavaScript if multiple devices
    if len(successful_results) > 1:
        devices_json = json.dumps(devices)
        speeds_json = json.dumps(speeds)
        load_times_json = json.dumps(load_times)
        
        colors = {'CPU': '#4CAF50', 'GPU': '#2196F3', 'NPU': '#FF9800'}
        bg_colors = [colors.get(d, '#666') for d in devices]
        bg_colors_json = json.dumps(bg_colors)
        
        html += f"""
    <script>
        // Speed Chart
        const speedCtx = document.getElementById('speedChart').getContext('2d');
        const speedChart = new Chart(speedCtx, {{
            type: 'bar',
            data: {{
                labels: {devices_json},
                datasets: [{{
                    label: 'Speed (tokens/second)',
                    data: {speeds_json},
                    backgroundColor: {bg_colors_json},
                    borderColor: {bg_colors_json},
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Device Speed Comparison',
                        font: {{ size: 16, weight: 'bold' }}
                    }},
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Speed (tokens/sec)'
                        }}
                    }}
                }}
            }}
        }});
        
        // Load Time Chart
        const loadTimeCtx = document.getElementById('loadTimeChart').getContext('2d');
        const loadTimeChart = new Chart(loadTimeCtx, {{
            type: 'bar',
            data: {{
                labels: {devices_json},
                datasets: [{{
                    label: 'Load Time (seconds)',
                    data: {load_times_json},
                    backgroundColor: {bg_colors_json},
                    borderColor: {bg_colors_json},
                    borderWidth: 2
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Device Load Time Comparison',
                        font: {{ size: 16, weight: 'bold' }}
                    }},
                    legend: {{
                        display: false
                    }}
                }},
                scales: {{
                    y: {{
                        beginAtZero: true,
                        title: {{
                            display: true,
                            text: 'Load Time (seconds)'
                        }}
                    }}
                }}
            }}
        }});
    </script>
"""
    
    html += """
</body>
</html>
"""
    
    # Write file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✅ HTML report generated: {output_file}")
    print(f"   Open {output_file} in your browser to view the report\n")
    
    return output_file


def test_model(model_path, device="CPU"):
    """Test a local OpenVINO model"""
    
    model_path = Path(model_path).expanduser().resolve()
    
    if not model_path.exists():
        print(f"❌ Error: Model directory not found: {model_path}")
        return None
    
    # Check for required files
    required_files = ['openvino_model.xml', 'openvino_model.bin']
    missing_files = [f for f in required_files if not (model_path / f).exists()]
    
    if missing_files:
        print(f"❌ Error: Missing required files: {', '.join(missing_files)}")
        return None
    
    # Extract model information
    model_info = extract_model_info(model_path)
    
    print(f"\n{'='*80}")
    print(f"🚀 Testing Local OpenVINO Model")
    print(f"{'='*80}")
    print(f"📁 Model: {model_path.name}")
    print(f"🏗️  Architecture: {model_info['architecture']}")
    print(f"📊 Parameters: {model_info['parameters']}")
    print(f"🔢 Quantization: {model_info['quantization']}")
    if model_info['hidden_layers']:
        print(f"📐 Layers: {model_info['hidden_layers']}, Hidden Size: {model_info['hidden_size']}")
    print(f"🖥️  Device: {device}")
    print(f"{'='*80}\n")
    
    result = {
        'device': device,
        'model_info': model_info,
        'success': False,
        'load_time': 0,
        'generation_times': [],
        'token_counts': [],
        'avg_time': 0,
        'avg_tokens': 0,
        'avg_speed': 0,
        'total_time': 0,
        'total_tokens': 0,
        'error': None
    }
    
    try:
        # Load model
        print(f"📥 Loading model to {device}...")
        load_start = time.time()
        pipe = ov_genai.LLMPipeline(str(model_path), device)
        result['load_time'] = time.time() - load_start
        print(f"✅ Model loaded in {result['load_time']:.1f}s\n")
        
        # Configure generation
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 100
        config.temperature = 0.7
        config.top_p = 0.9
        config.do_sample = True
        
        # Test prompts
        prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "Write a haiku about technology."
        ]
        
        print(f"🧪 Running {len(prompts)} test prompts...\n")
        
        generation_times = []
        token_counts = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"{'─'*80}")
            print(f"[{i}/{len(prompts)}] Prompt: {prompt}")
            print(f"{'─'*80}")
            
            start = time.time()
            response = pipe.generate(prompt, config)
            elapsed = time.time() - start
            
            tokens = len(response.split())
            generation_times.append(elapsed)
            token_counts.append(tokens)
            
            speed = tokens / elapsed if elapsed > 0 else 0
            
            print(f"Response: {response}\n")
            print(f"⏱️  Time: {elapsed:.2f}s | Tokens: {tokens} | Speed: {speed:.1f} tok/s\n")
        
        # Calculate statistics
        result['generation_times'] = generation_times
        result['token_counts'] = token_counts
        result['avg_time'] = statistics.mean(generation_times)
        result['avg_tokens'] = statistics.mean(token_counts)
        result['total_tokens'] = sum(token_counts)
        result['total_time'] = sum(generation_times)
        result['avg_speed'] = result['total_tokens'] / result['total_time'] if result['total_time'] > 0 else 0
        result['success'] = True
        
        print(f"{'='*80}")
        print(f"✅ Test completed successfully!")
        print(f"   Average Speed: {result['avg_speed']:.1f} tok/s")
        print(f"{'='*80}\n")
        
    except Exception as e:
        result['error'] = str(e)
        print(f"\n❌ Error: {e}\n")
    
    return result


def is_openvino_model(path):
    """Check if a directory contains an OpenVINO model"""
    path = Path(path)
    return (path / 'openvino_model.xml').exists() and (path / 'openvino_model.bin').exists()


def extract_model_info(model_path):
    """Extract model metadata from config files"""
    model_path = Path(model_path)
    info = {
        'name': model_path.name,
        'parameters': 'Unknown',
        'quantization': 'Unknown',
        'architecture': 'Unknown',
        'hidden_layers': None,
        'hidden_size': None
    }
    
    # Try to read config.json
    config_file = model_path / 'config.json'
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                
            # Extract architecture
            if 'architectures' in config and config['architectures']:
                info['architecture'] = config['architectures'][0]
            elif 'model_type' in config:
                info['architecture'] = config['model_type']
            
            # Extract layer/size info for parameter estimation
            if 'num_hidden_layers' in config:
                info['hidden_layers'] = config['num_hidden_layers']
            if 'hidden_size' in config:
                info['hidden_size'] = config['hidden_size']
            
            # Estimate parameters from model name or calculate from config
            # Common patterns: 7B, 8B, 1.1B, etc.
            original_name = config.get('_name_or_path', '')
            if 'Llama-3.1-8B' in original_name or 'Llama-3-8B' in original_name:
                info['parameters'] = '8B'
            elif 'Llama-3.2-1B' in original_name:
                info['parameters'] = '1B'
            elif 'Llama-3.2-3B' in original_name:
                info['parameters'] = '3B'
            elif 'TinyLlama' in original_name:
                info['parameters'] = '1.1B'
            elif 'Qwen' in original_name:
                # Extract from name pattern
                for part in original_name.split('/'):
                    if 'B' in part and any(c.isdigit() for c in part):
                        # Extract number before B
                        import re
                        match = re.search(r'(\d+\.?\d*)B', part)
                        if match:
                            info['parameters'] = f"{match.group(1)}B"
                            break
        except Exception as e:
            pass
    
    # Try to read model_config.json for quantization info
    model_config_file = model_path / 'model_config.json'
    if model_config_file.exists():
        try:
            with open(model_config_file, 'r') as f:
                model_config = json.load(f)
            
            # Extract quantization from arguments
            if 'arguments' in model_config:
                args = model_config['arguments']
                if 'int4' in args or '--weight-format' in args:
                    idx = args.index('--weight-format') if '--weight-format' in args else -1
                    if idx >= 0 and idx + 1 < len(args):
                        info['quantization'] = args[idx + 1].upper()
                    elif 'int4' in str(args).lower():
                        info['quantization'] = 'INT4'
                elif 'int8' in str(args).lower():
                    info['quantization'] = 'INT8'
        except Exception as e:
            pass
    
    # Try to infer from directory name as fallback
    dir_name = model_path.name
    if info['quantization'] == 'Unknown':
        if 'INT4' in dir_name.upper() or '_I4' in dir_name.upper():
            info['quantization'] = 'INT4'
        elif 'INT8' in dir_name.upper() or '_I8' in dir_name.upper():
            info['quantization'] = 'INT8'
        elif 'FP16' in dir_name.upper():
            info['quantization'] = 'FP16'
    
    if info['parameters'] == 'Unknown':
        # Try to extract from directory name
        import re
        match = re.search(r'(\d+\.?\d*)[Bb]', dir_name)
        if match:
            info['parameters'] = f"{match.group(1)}B"
    
    return info


def find_models_in_directory(directory):
    """Find all OpenVINO models in a directory"""
    directory = Path(directory).expanduser().resolve()
    models = []
    
    # Check if directory itself is a model
    if is_openvino_model(directory):
        return [directory]
    
    # Check subdirectories
    for subdir in directory.iterdir():
        if subdir.is_dir() and is_openvino_model(subdir):
            models.append(subdir)
    
    return models


def generate_comparison_report(all_results, models_tested, output_file=None):
    """Generate comprehensive HTML comparison report for multiple models"""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Generate filename if not provided
    if output_file is None:
        timestamp_file = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"local_models_comparison_{timestamp_file}.html"
    
    # Collect successful results
    comparison_data = []
    for model_name, device_results in all_results.items():
        for result in device_results:
            if result['success']:
                model_info = result.get('model_info', {})
                comparison_data.append({
                    'model': model_name,
                    'device': result['device'],
                    'speed': result['avg_speed'],
                    'load_time': result['load_time'],
                    'avg_time': result['avg_time'],
                    'total_tokens': result['total_tokens'],
                    'total_time': result['total_time'],
                    'parameters': model_info.get('parameters', 'Unknown'),
                    'quantization': model_info.get('quantization', 'Unknown'),
                    'architecture': model_info.get('architecture', 'Unknown')
                })
    
    if not comparison_data:
        print("⚠️ No successful tests to compare")
        return None
    
    # Find best overall
    best_overall = max(comparison_data, key=lambda x: x['speed'])
    
    # Get unique devices
    devices = sorted(list(set(item['device'] for item in comparison_data)))
    
    # Device statistics
    device_stats = {}
    for device in devices:
        device_data = [item for item in comparison_data if item['device'] == device]
        if device_data:
            speeds = [item['speed'] for item in device_data]
            device_stats[device] = {
                'avg_speed': statistics.mean(speeds),
                'min_speed': min(speeds),
                'max_speed': max(speeds),
                'count': len(speeds)
            }
    
    # HTML template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local Models Comparison Report</title>
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
        
        .info-box {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin-bottom: 20px;
        }}
        
        .info-box strong {{
            color: #667eea;
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
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🚀 Local Models Comparison</h1>
            <div class="subtitle">Multi-Model Device Performance Analysis</div>
            <div class="timestamp">Generated: {timestamp}</div>
        </header>
        
        <div class="content">
            <section class="section">
                <div class="info-box">
                    <strong>📁 Models Tested:</strong> {len(models_tested)}<br>
                    <strong>🖥️ Devices:</strong> {', '.join(devices)}<br>
                    <strong>✅ Successful Tests:</strong> {len(comparison_data)}
                </div>
            </section>
            
            <section class="section">
                <div class="highlight-box">
                    <h3>🏆 Best Overall Performance</h3>
                    <div class="stat">{best_overall['model']}</div>
                    <div>on {best_overall['device']} device</div>
                    <div class="stat">{best_overall['speed']:.1f} tok/s</div>
                    <div style="margin-top: 15px; font-size: 0.9em;">
                        <div>Parameters: {best_overall['parameters']} | Quantization: {best_overall['quantization']}</div>
                    </div>
                </div>
            </section>
            
            <section class="section">
                <h2>📊 Device Performance Summary</h2>
                <div class="metrics-grid">
"""
    
    # Add device metric cards
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
            
            <section class="section">
                <h2>📈 Detailed Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Params</th>
                            <th>Quant</th>
                            <th>Device</th>
                            <th>Speed (tok/s)</th>
                            <th>Load Time (s)</th>
                            <th>Avg Time (s)</th>
                            <th>Total Tokens</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    # Sort by speed
    comparison_data_sorted = sorted(comparison_data, key=lambda x: x['speed'], reverse=True)
    fastest_speed = comparison_data_sorted[0]['speed']
    
    for item in comparison_data_sorted:
        fastest_badge = ' <span class="badge fastest">FASTEST</span>' if item['speed'] == fastest_speed else ''
        device_class = f"device-{item['device'].lower()}"
        
        html += f"""
                        <tr>
                            <td><strong>{item['model']}</strong>{fastest_badge}</td>
                            <td>{item['parameters']}</td>
                            <td>{item['quantization']}</td>
                            <td class="{device_class}"><strong>{item['device']}</strong></td>
                            <td><strong>{item['speed']:.1f}</strong></td>
                            <td>{item['load_time']:.1f}</td>
                            <td>{item['avg_time']:.2f}</td>
                            <td>{item['total_tokens']}</td>
                        </tr>
"""
    
    html += """
                    </tbody>
                </table>
            </section>
"""
    
    # Add charts
    if len(comparison_data) > 1:
        # Prepare chart data - group by model
        models = sorted(list(set(item['model'] for item in comparison_data)))
        chart_speeds = {device: [] for device in devices}
        
        for model in models:
            for device in devices:
                speed = next((item['speed'] for item in comparison_data 
                            if item['model'] == model and item['device'] == device), 0)
                chart_speeds[device].append(speed)
        
        html += f"""
            <section class="section">
                <h2>📊 Performance Charts</h2>
                
                <div class="chart-container">
                    <canvas id="speedChart"></canvas>
                </div>
            </section>
"""
    
    html += """
        </div>
        
        <footer>
            <p>Generated by OpenVINO Local Model Test Tool</p>
            <p>OpenVINO™ is a trademark of Intel Corporation</p>
        </footer>
    </div>
"""
    
    # Add chart JavaScript
    if len(comparison_data) > 1:
        models_json = json.dumps(models)
        colors = {'CPU': '#4CAF50', 'GPU': '#2196F3', 'NPU': '#FF9800'}
        
        html += """
    <script>
        const speedCtx = document.getElementById('speedChart').getContext('2d');
        const speedChart = new Chart(speedCtx, {
            type: 'bar',
            data: {
                labels: """ + models_json + """,
                datasets: [
"""
        
        for device in devices:
            color = colors.get(device, '#666')
            speeds_json = json.dumps(chart_speeds[device])
            html += f"""
                    {{
                        label: '{device}',
                        data: {speeds_json},
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
                        text: 'Model Performance Comparison Across Devices',
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
    </script>
"""
    
    html += """
</body>
</html>
"""
    
    # Write file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✅ Comparison report generated: {output_file}")
    print(f"   Open {output_file} in your browser to view the report\n")
    
    return output_file


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n❌ Error: Model path required\n")
        return 1
    
    input_path = Path(sys.argv[1]).expanduser().resolve()
    
    if not input_path.exists():
        print(f"❌ Error: Path not found: {input_path}")
        return 1
    
    # Find all models
    models = find_models_in_directory(input_path)
    
    if not models:
        print(f"❌ Error: No OpenVINO models found in {input_path}")
        print("   Looking for directories with openvino_model.xml and openvino_model.bin")
        return 1
    
    print(f"\n{'='*80}")
    print(f"🔍 Found {len(models)} model(s) to test")
    print(f"{'='*80}")
    for model in models:
        model_info = extract_model_info(model)
        print(f"   • {model.name}")
        print(f"     Architecture: {model_info['architecture']} | "
              f"Parameters: {model_info['parameters']} | "
              f"Quantization: {model_info['quantization']}")
    print(f"{'='*80}\n")
    
    # Determine devices to test
    devices_to_test = []
    if len(sys.argv) >= 3:
        # Specific device requested
        devices_to_test = [sys.argv[2].upper()]
    else:
        # Test all available devices
        devices_to_test = ['CPU', 'GPU', 'NPU']
    
    # Get available devices
    try:
        import openvino as ov
        core = ov.Core()
        available = core.available_devices
        
        # Filter to only available devices
        final_devices = []
        for device in devices_to_test:
            if device == 'CPU' and 'CPU' in available:
                final_devices.append('CPU')
            elif device == 'GPU' and any('GPU' in d for d in available):
                final_devices.append('GPU')
            elif device == 'NPU' and any('NPU' in d for d in available):
                final_devices.append('NPU')
        
        devices_to_test = final_devices
        
        if not devices_to_test:
            print("❌ No requested devices available")
            return 1
            
        print(f"🖥️  Testing on: {', '.join(devices_to_test)}\n")
        
    except Exception as e:
        print(f"⚠️  Could not detect devices: {e}")
        devices_to_test = ['CPU']  # Fallback to CPU
    
    # Test all models on all devices
    all_results = {}
    
    for model_idx, model_path in enumerate(models, 1):
        model_name = model_path.name
        print(f"\n{'='*80}")
        print(f"📦 MODEL {model_idx}/{len(models)}: {model_name}")
        print(f"{'='*80}")
        
        model_results = []
        
        for device in devices_to_test:
            result = test_model(model_path, device)
            if result:
                model_results.append(result)
        
        all_results[model_name] = model_results
    
    # Generate appropriate report
    if len(models) == 1:
        # Single model - generate single model report
        model_name = models[0].name
        results = all_results[model_name]
        if results:
            generate_html_report(results, models[0])
    else:
        # Multiple models - generate comparison report
        models_tested = [m.name for m in models]
        generate_comparison_report(all_results, models_tested)
    
    print(f"\n{'='*80}")
    print(f"✅ Testing Complete!")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    sys.exit(main())
