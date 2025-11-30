#!/usr/bin/env python3
"""
Test script for the Inference endpoints
"""

import sys
import os
import requests
import time
import pandas as pd
import io

def create_test_data():
    """Create a simple test dataset."""
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
        'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20] * 10,
        'feature3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 10,
        'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10,
    }
    return pd.DataFrame(data)

def create_inference_data():
    """Create test data for inference (without target)."""
    data = {
        'feature1': [1, 3, 5, 7, 9],
        'feature2': [2, 6, 10, 14, 18],
        'feature3': [0.1, 0.3, 0.5, 0.7, 0.9],
    }
    return pd.DataFrame(data)

def upload_and_train(base_url="http://localhost:8888"):
    """Upload data and train a model (synchronous)."""
    print("📤 Uploading training data...")
    
    # Create and upload training data
    df = create_test_data()
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer)
    csv_bytes = csv_buffer.getvalue().encode('utf-8')
    
    files = {'file': ('train.csv', csv_bytes, 'text/csv')}
    response = requests.post(f"{base_url}/upload", files=files, timeout=30)
    
    if response.status_code != 201:
        print(f"❌ Upload failed: {response.status_code}")
        return None, None
    
    analyzer_id = response.json()['analyzer_id']
    print(f"✅ Uploaded. Analyzer ID: {analyzer_id}")
    
    # Create and train (synchronous - training completes before response)
    print("🤖 Creating and training model (this may take a moment)...")
    trainer_request = {
        "analyzer_id": analyzer_id,
        "target_variable": "target",
        "task": "binary",
        "k_folds": 2
    }
    
    response = requests.post(f"{base_url}/trainers", json=trainer_request, timeout=300)
    
    if response.status_code != 200:
        print(f"❌ Trainer creation failed: {response.status_code} - {response.text}")
        return analyzer_id, None
    
    data = response.json()
    trainer_id = data['trainer_id']
    print(f"✅ Trainer created and trained: {trainer_id}")
    if data.get('training_time'):
        print(f"   Training time: {data['training_time']:.2f}s")
    
    return analyzer_id, trainer_id

def test_csv_inference(base_url, trainer_id):
    """Test inference with CSV file."""
    print("\n🔍 Test: CSV File Inference...")
    
    try:
        # Create inference data
        df = create_inference_data()
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        
        files = {'file': ('test.csv', csv_bytes, 'text/csv')}
        response = requests.post(
            f"{base_url}/trainers/{trainer_id}/infer",
            files=files,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ CSV inference successful!")
            print(f"   Predictions: {data['predictions']}")
            print(f"   Num samples: {data['num_samples']}")
            if data.get('probabilities'):
                print(f"   Has probabilities: Yes ({len(data['probabilities'])} samples)")
            return True
        else:
            print(f"❌ CSV inference failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ CSV inference error: {e}")
        return None

def test_json_inference(base_url, trainer_id):
    """Test inference with JSON payload."""
    print("\n🔍 Test: JSON Inference...")
    
    try:
        df = create_inference_data()
        data = {"data": df.to_dict('records')}
        
        response = requests.post(
            f"{base_url}/trainers/{trainer_id}/infer/json",
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ JSON inference successful!")
            print(f"   Predictions: {result['predictions']}")
            print(f"   Num samples: {result['num_samples']}")
            if result.get('probabilities'):
                print(f"   Has probabilities: Yes")
            return True
        else:
            print(f"❌ JSON inference failed: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ JSON inference error: {e}")
        return None

def test_error_cases(base_url, trainer_id):
    """Test error handling."""
    print("\n🔍 Test: Error Cases...")
    
    # Test inference with non-existent trainer
    response = requests.post(
        f"{base_url}/trainers/invalid-id/infer/json",
        json={"data": [{"feature1": 1}]},
        timeout=10
    )
    if response.status_code == 404:
        print("✅ Correctly handled non-existent trainer")
    else:
        print(f"⚠️  Expected 404 for invalid trainer, got {response.status_code}")
    
def cleanup(base_url, analyzer_id, trainer_id):
    """Cleanup resources."""
    print("\n🧹 Cleaning up...")
    
    if trainer_id:
        try:
            response = requests.delete(f"{base_url}/trainers/{trainer_id}", timeout=10)
            if response.status_code == 200:
                print(f"✅ Deleted trainer {trainer_id}")
        except Exception as e:
            print(f"⚠️  Error deleting trainer: {e}")
    
    if analyzer_id:
        try:
            response = requests.delete(f"{base_url}/analyzers/{analyzer_id}", timeout=10)
            if response.status_code == 200:
                print(f"✅ Deleted analyzer {analyzer_id}")
        except Exception as e:
            print(f"⚠️  Error deleting analyzer: {e}")

def main():
    """Run all inference tests."""
    print("🧪 Testing Inference Endpoints...")
    print("=" * 70)
    
    base_url = os.environ.get('TEST_BASE_URL', 'http://localhost:8888')
    print(f"Testing against: {base_url}\n")
    
    analyzer_id = None
    trainer_id = None
    
    try:
        # Setup: Upload and train
        analyzer_id, trainer_id = upload_and_train(base_url)
        if not trainer_id:
            print("\n❌ Cannot run inference tests without trained model")
            return 1
        
        # Test 1: CSV inference
        test_csv_inference(base_url, trainer_id)
        
        # Test 2: JSON inference
        test_json_inference(base_url, trainer_id)
        
        # Test 3: Error cases
        test_error_cases(base_url, trainer_id)
        
        print("\n" + "=" * 70)
        print("🎉 Inference endpoint tests completed!")
        
        return 0
        
    finally:
        cleanup(base_url, analyzer_id, trainer_id)

if __name__ == "__main__":
    sys.exit(main())

