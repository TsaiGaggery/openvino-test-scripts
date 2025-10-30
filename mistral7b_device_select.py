#!/usr/bin/env python3
# mistral7b_device_select.py
"""
Mistral 7B Interactive Chatbot with Device Selection
Choose CPU, GPU, or NPU at startup
"""
import os
import sys
import time
import re

# Clear NPU environment variables
for key in list(os.environ.keys()):
    if 'NPU' in key or 'OV_NPU' in key:
        del os.environ[key]

from pathlib import Path
from huggingface_hub import snapshot_download
import openvino as ov
import openvino_genai as ov_genai

print("=" * 70)
print("Mistral 7B Interactive Chatbot")
print("=" * 70)

# Download model
print("\nDownloading model...")
MODEL_ID = "OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov"
MODEL_DIR = snapshot_download(repo_id=MODEL_ID, local_dir="models/mistral7b_ov")
print(f"✅ Model ready: {MODEL_DIR}")

# Check available devices
print("\nChecking available devices...")
core = ov.Core()
available_devices = core.available_devices
print(f"Available: {', '.join(available_devices)}")

# Setup cache
cache_dir = Path("./ov_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

# Device selection
print("\n" + "=" * 70)
print("Device Selection")
print("=" * 70)

valid_devices = []
device_info = {}

if 'CPU' in available_devices:
    valid_devices.append('CPU')
    device_info['CPU'] = "Most compatible, moderate speed"

if any('GPU' in d for d in available_devices):
    valid_devices.append('GPU')
    device_info['GPU'] = "High performance, fast inference"

if any('NPU' in d for d in available_devices):
    valid_devices.append('NPU')
    device_info['NPU'] = "Power efficient, good speed"

print("\nAvailable devices:")
for i, device in enumerate(valid_devices, 1):
    print(f"  {i}. {device:<5} - {device_info[device]}")

# Get user choice
while True:
    try:
        choice = input(f"\nSelect device (1-{len(valid_devices)}) or name: ").strip()
        
        # Check if numeric choice
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(valid_devices):
                DEVICE = valid_devices[idx]
                break
        # Check if device name
        elif choice.upper() in valid_devices:
            DEVICE = choice.upper()
            break
        else:
            print(f"Invalid choice. Please enter 1-{len(valid_devices)} or device name.")
    except (ValueError, KeyboardInterrupt):
        print("\nExiting...")
        sys.exit(0)

print(f"\n✅ Selected: {DEVICE}")

# Configure device
if DEVICE in ['GPU', 'NPU']:
    try:
        core.set_property(DEVICE, {
            "CACHE_DIR": str(cache_dir),
            "PERFORMANCE_HINT": "LATENCY"
        })
        print(f"✅ {DEVICE} configured")
    except Exception as e:
        print(f"⚠️  Configuration warning: {e}")

# Load model
print(f"\nLoading model to {DEVICE}...")
if DEVICE == 'NPU':
    print("(First run on NPU takes 30-60 seconds for compilation)")

load_start = time.time()
pipe = ov_genai.LLMPipeline(MODEL_DIR, DEVICE)
load_time = time.time() - load_start

print(f"✅ Model loaded in {load_time:.1f} seconds")

# Generation config
config = ov_genai.GenerationConfig()
config.max_new_tokens = 150
config.temperature = 0.7
config.top_p = 0.9
config.do_sample = True
config.repetition_penalty = 1.1

# Conversation state
history = []
stats = {
    'total_time': 0,
    'total_tokens': 0,
    'interactions': 0
}

def clean_response(text: str) -> str:
    """Remove unwanted continuation from response"""
    text = text.replace("[INST]", "").replace("[/INST]", "").strip()
    
    stop_patterns = [
        "\n\nNow,", "\nNow,",
        "\n\nFor example,", "\nFor example,",
        "\n\nLet me", "\nLet me",
        "\n\nUser:", "\nUser:",
    ]
    
    min_pos = len(text)
    for pattern in stop_patterns:
        pos = text.find(pattern)
        if pos != -1 and pos < min_pos:
            min_pos = pos
    
    if min_pos < len(text):
        text = text[:min_pos].strip()
    
    sentences = re.split(r'[.!?]\s+', text)
    if len(sentences) > 1 and len(sentences[-1]) < 20:
        text = '. '.join(sentences[:-1])
        if not text.endswith('.'):
            text += '.'
    
    return text.strip()

def format_prompt(user_msg: str) -> str:
    """Format with Mistral instruction format"""
    if not history:
        return f"[INST] {user_msg} [/INST]"
    
    recent = history[-4:] if len(history) >= 4 else history
    
    prompt = ""
    for msg in recent:
        if msg['role'] == 'user':
            prompt += f"[INST] {msg['content']} [/INST] "
        elif msg['role'] == 'assistant':
            prompt += f"{msg['content']} "
    
    prompt += f"[INST] {user_msg} [/INST]"
    return prompt

def chat(user_msg: str):
    """Generate and display response"""
    prompt = format_prompt(user_msg)
    
    start = time.time()
    response = pipe.generate(prompt, config)
    elapsed = time.time() - start
    
    response = clean_response(response)
    
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": response})
    
    if len(history) > 6:
        history[:] = history[-6:]
    
    # Update stats
    tokens = len(response.split())
    speed = tokens / elapsed if elapsed > 0 else 0
    stats['total_time'] += elapsed
    stats['total_tokens'] += tokens
    stats['interactions'] += 1
    
    # Display
    print(f"AI: {response}")
    print(f"⏱️  {elapsed:.1f}s | {speed:.1f} tok/s\n")

def show_stats():
    """Display session statistics"""
    print(f"\n{'='*70}")
    print("Session Statistics")
    print('='*70)
    print(f"Device: {DEVICE}")
    print(f"Load time: {load_time:.1f}s")
    print(f"Interactions: {stats['interactions']}")
    print(f"Total generation time: {stats['total_time']:.1f}s")
    print(f"Total tokens: {stats['total_tokens']}")
    if stats['total_time'] > 0:
        avg_speed = stats['total_tokens'] / stats['total_time']
        print(f"Average speed: {avg_speed:.1f} tok/s")
    print('='*70 + '\n')

# Interactive mode
print("\n" + "=" * 70)
print("Interactive Mode - Ready to Chat!")
print("=" * 70)
print(f"\nUsing: {DEVICE}")
print("\nCommands:")
print("  'reset'  - Clear conversation history")
print("  'stats'  - Show performance statistics")
print("  'quit'   - Exit program\n")

try:
    while True:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            show_stats()
            print("Goodbye! 👋")
            break
        
        if user_input.lower() in ['reset', 'clear']:
            history.clear()
            print("✅ Conversation history cleared\n")
            continue
        
        if user_input.lower() == 'stats':
            show_stats()
            continue
        
        chat(user_input)

except KeyboardInterrupt:
    print("\n")
    show_stats()
    print("Goodbye! 👋")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
