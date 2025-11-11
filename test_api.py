"""
Quick API Test Script
Run this after starting the backend to verify everything works!
"""
import requests
import json

API_BASE = "http://localhost:8000/api"

print("🧪 Testing AutoML Forge API...\n")

# Test 1: Health Check
print("1️⃣  Testing health endpoint...")
try:
    response = requests.get(f"{API_BASE}/health")
    if response.status_code == 200:
        print("   ✅ Health check passed!")
        print(f"   {json.dumps(response.json(), indent=2)}")
    else:
        print(f"   ❌ Health check failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("   💡 Make sure backend is running: py run_backend.py")

print()

# Test 2: Upload sample data
print("2️⃣  Testing file upload...")
try:
    with open("tests/fixtures/sample_data.csv", "rb") as f:
        files = {"file": ("sample_data.csv", f, "text/csv")}
        response = requests.post(f"{API_BASE}/upload", files=files)

    if response.status_code == 200:
        data = response.json()
        print("   ✅ Upload successful!")
        print(f"   📊 Rows: {data['rows']}, Columns: {data['columns']}")
        print(f"   📁 File ID: {data['file_id']}")
        file_id = data['file_id']
    else:
        print(f"   ❌ Upload failed: {response.status_code}")
        file_id = None
except Exception as e:
    print(f"   ❌ Error: {e}")
    file_id = None

print()

# Test 3: Get cleaning suggestions
if file_id:
    print("3️⃣  Testing data cleaning suggestions...")
    try:
        response = requests.get(f"{API_BASE}/clean/suggestions/{file_id}")

        if response.status_code == 200:
            data = response.json()
            suggestions = data['suggestions']
            print(f"   ✅ Found {len(suggestions)} data issues!")

            for i, suggestion in enumerate(suggestions[:3], 1):  # Show first 3
                print(f"\n   Issue {i}: {suggestion['issue']}")
                print(f"   Severity: {suggestion['severity'].upper()}")
                print(f"   Suggestion: {suggestion['suggestion']}")
                print(f"   Reason: {suggestion['reason'][:80]}...")
        else:
            print(f"   ❌ Cleaning suggestions failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("✅ API tests complete!")
print("🎉 Your backend is working! Now start the frontend:")
print("   streamlit run frontend/app.py")
print("="*60)
