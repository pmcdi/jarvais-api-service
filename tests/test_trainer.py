#!/usr/bin/env python3
"""
Test script for the Trainer endpoints
"""

import sys
import os
import requests
import time
import pandas as pd
import io

def create_test_data():
    """Create a simple test dataset for training."""
    # Create a simple binary classification dataset
    data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,
        'feature2': [2, 4, 6, 8, 10, 12, 14, 16, 18, 20] * 10,
        'feature3': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] * 10,
        'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1] * 10,
    }
    df = pd.DataFrame(data)
    return df

def test_upload_data(base_url="http://localhost:8888"):
    """Upload test data and return analyzer ID."""
    try:
        # Create test data
        df = create_test_data()
        
        # Convert to CSV bytes
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        csv_bytes = csv_buffer.getvalue().encode('utf-8')
        
        # Upload the file
        files = {'file': ('test_data.csv', csv_bytes, 'text/csv')}
        response = requests.post(f"{base_url}/upload", files=files, timeout=30)
        
        if response.status_code == 201:
            data = response.json()
            analyzer_id = data['analyzer_id']
            print(f"✅ Data uploaded successfully. Analyzer ID: {analyzer_id}")
            return analyzer_id
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Upload failed: {e}")
        return None

def test_create_trainer(base_url="http://localhost:8888", analyzer_id=None):
    """Create and train a model (synchronous)."""
    if not analyzer_id:
        print("❌ No analyzer ID provided")
        return None
        
    try:
        # Create trainer request
        trainer_request = {
            "analyzer_id": analyzer_id,
            "target_variable": "target",
            "task": "binary",
            "k_folds": 2,
            "time_limit": 120,
            "feature_reduction_method": None,
            "n_features": None
        }
        
        print("   (Training is synchronous, this may take a moment...)")
        response = requests.post(
            f"{base_url}/trainers",
            json=trainer_request,
            timeout=300  # Increased timeout for synchronous training
        )
        
        if response.status_code == 200:
            data = response.json()
            trainer_id = data['trainer_id']
            print(f"✅ Trainer created and trained successfully. Trainer ID: {trainer_id}")
            print(f"   Task: {data['task']}")
            print(f"   Target: {data['target_variable']}")
            if data.get('training_time'):
                print(f"   Training time: {data['training_time']:.2f}s")
            return trainer_id
        else:
            print(f"❌ Trainer creation failed: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Trainer creation failed: {e}")
        return None

def test_list_trainers(base_url="http://localhost:8888"):
    """List all trainers."""
    try:
        response = requests.get(f"{base_url}/trainers", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Trainers list endpoint working: {data['count']} trainers")
            for trainer in data['trainers']:
                print(f"   - {trainer['trainer_id']}: {trainer['task']} (target: {trainer['target_variable']})")
            return True
        else:
            print(f"❌ Trainers list failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Trainers list failed: {e}")
        return False

def test_trainer_info(base_url="http://localhost:8888", trainer_id=None):
    """Get trainer information."""
    if not trainer_id:
        print("❌ No trainer ID provided")
        return False
        
    try:
        response = requests.get(f"{base_url}/trainers/{trainer_id}", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Trainer info retrieved:")
            print(f"   Trainer ID: {data['trainer_id']}")
            print(f"   Task: {data['task']}")
            print(f"   K-folds: {data['k_folds']}")
            print(f"   Target: {data['target_variable']}")
            if data.get('training_time'):
                print(f"   Training time: {data['training_time']:.2f}s")
            return True
        else:
            print(f"❌ Trainer info failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Trainer info failed: {e}")
        return False

def test_trainer_results(base_url="http://localhost:8888", trainer_id=None):
    """Get trainer results."""
    if not trainer_id:
        print("❌ No trainer ID provided")
        return False
        
    try:
        response = requests.get(f"{base_url}/trainers/{trainer_id}/results", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Trainer results retrieved:")
            print(f"   Best model: {data['best_model']}")
            print(f"   Training time: {data['training_time']:.2f}s")
            print(f"   Leaderboard entries: {len(data['leaderboard'])}")
            return True
        elif response.status_code == 400:
            print(f"⏳ Trainer not completed yet: {response.json()['detail']}")
            return False
        else:
            print(f"❌ Trainer results failed: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Trainer results failed: {e}")
        return False

def test_delete_trainer(base_url="http://localhost:8888", trainer_id=None):
    """Delete a trainer."""
    if not trainer_id:
        print("❌ No trainer ID provided")
        return False
        
    try:
        response = requests.delete(f"{base_url}/trainers/{trainer_id}", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Trainer deleted: {data['message']}")
            return True
        else:
            print(f"❌ Trainer deletion failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Trainer deletion failed: {e}")
        return False

def main():
    """Run all trainer tests."""
    print("🧪 Testing Trainer Endpoints...")
    print("=" * 60)
    
    base_url = os.environ.get('TEST_BASE_URL', 'http://localhost:8888')
    print(f"Testing against: {base_url}")
    print()
    
    # Test 1: Upload data
    print("\n🔍 Test 1: Uploading test data...")
    analyzer_id = test_upload_data(base_url)
    if not analyzer_id:
        print("❌ Cannot continue without analyzer ID")
        return 1
    
    # Test 2: Create and train (synchronous)
    print("\n🔍 Test 2: Creating and training model...")
    trainer_id = test_create_trainer(base_url, analyzer_id)
    if not trainer_id:
        print("❌ Cannot continue without trainer ID")
        return 1
    
    # Test 3: List trainers
    print("\n🔍 Test 3: Listing trainers...")
    test_list_trainers(base_url)
    
    # Test 4: Get trainer info
    print("\n🔍 Test 4: Getting trainer info...")
    test_trainer_info(base_url, trainer_id)
    
    # Test 5: Get trainer results (training already completed synchronously)
    print("\n🔍 Test 5: Getting trainer results...")
    test_trainer_results(base_url, trainer_id)
    
    # Test 6: Delete trainer (cleanup)
    print("\n🔍 Test 6: Deleting trainer (cleanup)...")
    test_delete_trainer(base_url, trainer_id)
    
    print("\n" + "=" * 60)
    print("🎉 Trainer endpoint tests completed!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

