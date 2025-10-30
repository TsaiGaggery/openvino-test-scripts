#!/usr/bin/env python3
# mistral7b_interactive.py
"""
Mistral 7B Interactive Chatbot
Simple, direct-to-chat version with clean responses
All comments and output in English
"""
import os
import sys
import time
import re

# Clear NPU environment variables to avoid configuration conflicts
for key in list(os.environ.keys()):
    if 'NPU' in key or 'OV_NPU' in key:
        del os.environ[key]

from pathlib import Path
from huggingface_hub import snapshot_download
import openvino as ov
import openvino_genai as ov_genai

print("Loading Mistral 7B on Intel NPU...")

# Download and load model
MODEL_ID = "OpenVINO/Mistral-7B-Instruct-v0.3-int4-cw-ov"
MODEL_DIR = snapshot_download(repo_id=MODEL_ID, local_dir="models/mistral7b_ov")

# Initialize OpenVINO Core
core = ov.Core()
cache_dir = Path("./ov_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

# Configure NPU
try:
    core.set_property("NPU", {
        "CACHE_DIR": str(cache_dir),
        "PERFORMANCE_HINT": "LATENCY"
    })
except:
    pass

# Load model to NPU
DEVICE = "NPU"
print(f"Loading model to {DEVICE}... (first run takes 30-60 sec)")

pipe = ov_genai.LLMPipeline(MODEL_DIR, DEVICE)

# Generation configuration
config = ov_genai.GenerationConfig()
config.max_new_tokens = 150
config.temperature = 0.7
config.top_p = 0.9
config.do_sample = True
config.repetition_penalty = 1.1

print("✅ Ready!\n")

# Conversation history
history = []

def clean_response(text: str) -> str:
    """Remove unwanted continuation patterns from response"""
    # Remove instruction tags
    text = text.replace("[INST]", "").replace("[/INST]", "").strip()
    
    # Stop at continuation patterns
    stop_patterns = [
        "\n\nNow,", "\nNow,",
        "\n\nFor example,", "\nFor example,",
        "\n\nLet me", "\nLet me",
        "\n\nTo continue", "\nTo continue",
        "\n\nUser:", "\nUser:",
        "\n\nQuestion:", "\nQuestion:",
    ]
    
    # Find earliest stop pattern
    min_pos = len(text)
    for pattern in stop_patterns:
        pos = text.find(pattern)
        if pos != -1 and pos < min_pos:
            min_pos = pos
    
    if min_pos < len(text):
        text = text[:min_pos].strip()
    
    # Remove incomplete last sentence if it's very short
    sentences = re.split(r'[.!?]\s+', text)
    if len(sentences) > 1 and len(sentences[-1]) < 20:
        text = '. '.join(sentences[:-1])
        if not text.endswith('.'):
            text += '.'
    
    return text.strip()

def format_prompt(user_msg: str) -> str:
    """Format prompt using Mistral instruction format"""
    if not history:
        return f"[INST] {user_msg} [/INST]"
    
    # Keep only last 2 exchanges (4 messages)
    recent = history[-4:] if len(history) >= 4 else history
    
    prompt = ""
    for msg in recent:
        if msg['role'] == 'user':
            prompt += f"[INST] {msg['content']} [/INST] "
        elif msg['role'] == 'assistant':
            prompt += f"{msg['content']} "
    
    prompt += f"[INST] {user_msg} [/INST]"
    return prompt

def chat(user_msg: str) -> str:
    """Generate response to user message"""
    prompt = format_prompt(user_msg)
    
    # Generate
    start = time.time()
    response = pipe.generate(prompt, config)
    elapsed = time.time() - start
    
    # Clean response
    response = clean_response(response)
    
    # Update history
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": response})
    
    # Keep history manageable
    if len(history) > 6:
        history[:] = history[-6:]
    
    # Display with timing
    tokens = len(response.split())
    speed = tokens / elapsed if elapsed > 0 else 0
    print(f"AI: {response}")
    print(f"⏱️  {elapsed:.1f}s | {speed:.1f} tok/s\n")
    
    return response

# Interactive mode
print("=" * 70)
print("Mistral 7B Chatbot - Interactive Mode")
print("=" * 70)
print("\nCommands:")
print("  'reset' or 'clear' - Clear conversation history")
print("  'quit' or 'exit'   - Exit program\n")

while True:
    try:
        user_input = input("You: ").strip()
        
        if not user_input:
            continue
        
        # Handle commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! 👋")
            break
        
        if user_input.lower() in ['reset', 'clear']:
            history.clear()
            print("✅ Conversation history cleared\n")
            continue
        
        # Generate response
        chat(user_input)
        
    except KeyboardInterrupt:
        print("\n\nGoodbye! 👋")
        break
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        break
