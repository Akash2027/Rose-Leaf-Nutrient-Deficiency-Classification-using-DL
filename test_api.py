import google.generativeai as genai
import os

# -----------------------------------------------------------
# 🔴 PASTE YOUR KEY HERE FOR TESTING
# -----------------------------------------------------------
TEST_KEY = "AIzaSyCU0S0os_FqRK0VGdbm1ui-g0SmVcwhUIc"

print("--- STARTING CONNECTION TEST ---")

try:
    genai.configure(api_key=TEST_KEY)
    
    # 1. List available models for this key
    print("\n1. Checking available models for your API key...")
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"   - Found: {m.name}")
            available_models.append(m.name)
            
    if not available_models:
        print("\n❌ ERROR: No models found. Your API key might be invalid or has no access.")
    else:
        print(f"\n✅ Success! Found {len(available_models)} models.")
        
        # 2. Try to generate text
        print("\n2. Sending test message to AI...")
        # Pick the first available model
        model_name = available_models[0].name 
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Hello! Are you working?")
        print(f"   AI Response: {response.text}")
        print("\n✅ TEST PASSED. Your API Key is good.")

except Exception as e:
    print(f"\n❌ CRITICAL ERROR: {e}")