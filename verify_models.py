#!/usr/bin/env python3
"""
Verify that all models in benchmark.json are available on HuggingFace
"""
import json
from huggingface_hub import model_info

def verify_model(model_id):
    """Check if a model exists on HuggingFace"""
    try:
        info = model_info(model_id)
        return True, f"✅ Available (last modified: {info.lastModified.strftime('%Y-%m-%d')})"
    except Exception as e:
        error_msg = str(e)
        if "404" in error_msg or "not found" in error_msg.lower():
            return False, "❌ Not Found - Model doesn't exist"
        elif "401" in error_msg or "unauthorized" in error_msg.lower():
            return False, "❌ Not Found - Repository doesn't exist"
        else:
            return False, f"❌ Error: {error_msg[:100]}"

def main():
    print("=" * 80)
    print("🔍 Verifying OpenVINO Models from benchmark.json")
    print("=" * 80)
    
    # Load config
    with open('benchmark.json', 'r') as f:
        config = json.load(f)
    
    results = []
    for model in config['models']:
        model_name = model['name']
        model_id = model['model_id']
        enabled = model.get('enabled', False)
        
        print(f"\nChecking: {model_name}")
        print(f"  Model ID: {model_id}")
        print(f"  Enabled: {enabled}")
        
        success, message = verify_model(model_id)
        print(f"  Status: {message}")
        
        results.append({
            'name': model_name,
            'model_id': model_id,
            'enabled': enabled,
            'available': success,
            'message': message
        })
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    available = [r for r in results if r['available']]
    not_available = [r for r in results if not r['available']]
    enabled_available = [r for r in results if r['available'] and r['enabled']]
    enabled_not_available = [r for r in results if not r['available'] and r['enabled']]
    
    print(f"\n✅ Available models: {len(available)}/{len(results)}")
    for r in available:
        status = "🟢 ENABLED" if r['enabled'] else "⚪ disabled"
        print(f"   {status} {r['name']}")
    
    if not_available:
        print(f"\n❌ Not available models: {len(not_available)}/{len(results)}")
        for r in not_available:
            status = "🔴 ENABLED" if r['enabled'] else "⚪ disabled"
            print(f"   {status} {r['name']}")
            print(f"      {r['model_id']}")
    
    if enabled_not_available:
        print("\n⚠️  WARNING: These ENABLED models are NOT AVAILABLE:")
        for r in enabled_not_available:
            print(f"   🔴 {r['name']} - {r['model_id']}")
        print("\n   Please disable these models or fix their model_id!")
    
    print("\n" + "=" * 80)
    print(f"Result: {len(enabled_available)} enabled models are ready to benchmark")
    print("=" * 80)
    
    # Return exit code
    if enabled_not_available:
        print("\n⚠️  Some enabled models are not available. Fix benchmark.json before running.")
        return 1
    else:
        print("\n✅ All enabled models are available! Ready to run benchmark.")
        return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
