"""
Tests for the DICOM parsing module and API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


class TestDicomEndpoints:
    """Test DICOM API endpoints."""

    def test_create_database(self):
        """Test creating a new DICOM database."""
        response = client.post("/dicom")
        assert response.status_code == 201
        data = response.json()
        assert "database_id" in data
        assert "created_at" in data
        assert data["message"] == "DICOM database created successfully"
        return data["database_id"]

    def test_list_databases(self):
        """Test listing DICOM databases."""
        # Create a database first
        create_response = client.post("/dicom")
        assert create_response.status_code == 201
        
        response = client.get("/dicom")
        assert response.status_code == 200
        data = response.json()
        assert "count" in data
        assert "databases" in data
        assert data["count"] >= 1

    def test_get_database_info(self):
        """Test getting database info."""
        # Create a database first
        create_response = client.post("/dicom")
        database_id = create_response.json()["database_id"]
        
        response = client.get(f"/dicom/{database_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["database_id"] == database_id
        assert data["patient_count"] == 0
        assert data["keys"] == []

    def test_get_nonexistent_database(self):
        """Test getting a nonexistent database."""
        response = client.get("/dicom/nonexistent-id")
        assert response.status_code == 404

    def test_upload_single_patient(self):
        """Test uploading a single patient record."""
        # Create a database first
        create_response = client.post("/dicom")
        database_id = create_response.json()["database_id"]
        
        patient_data = {
            "PatientID": "12345",
            "PatientName": "Test Patient",
            "PatientAge": "045Y",
            "Modality": "CT",
            "StudyDescription": "Test Study"
        }
        
        response = client.post(f"/dicom/{database_id}/patient", json=patient_data)
        assert response.status_code == 200
        data = response.json()
        assert data["database_id"] == database_id
        assert data["patients_added"] == 1
        assert data["total_patients"] == 1

    def test_upload_single_patient_nonexistent_db(self):
        """Test uploading to a nonexistent database."""
        patient_data = {"PatientID": "12345"}
        response = client.post("/dicom/nonexistent-id/patient", json=patient_data)
        assert response.status_code == 404

    def test_upload_multiple_patients(self):
        """Test uploading multiple patient records."""
        # Create a database first
        create_response = client.post("/dicom")
        database_id = create_response.json()["database_id"]
        
        patients_data = {
            "patients": [
                {
                    "PatientID": "12345",
                    "PatientName": "Patient One",
                    "Modality": "CT"
                },
                {
                    "PatientID": "12346",
                    "PatientName": "Patient Two",
                    "Modality": "MR"
                },
                {
                    "PatientID": "12347",
                    "PatientName": "Patient Three",
                    "Modality": "US"
                }
            ]
        }
        
        response = client.post(f"/dicom/{database_id}/patients", json=patients_data)
        assert response.status_code == 200
        data = response.json()
        assert data["database_id"] == database_id
        assert data["patients_added"] == 3
        assert data["total_patients"] == 3

    def test_upload_multiple_patients_nonexistent_db(self):
        """Test uploading multiple patients to a nonexistent database."""
        patients_data = {"patients": [{"PatientID": "12345"}]}
        response = client.post("/dicom/nonexistent-id/patients", json=patients_data)
        assert response.status_code == 404

    def test_export_empty_database(self):
        """Test exporting an empty database."""
        # Create a database first
        create_response = client.post("/dicom")
        database_id = create_response.json()["database_id"]
        
        response = client.get(f"/dicom/{database_id}/export")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert response.text == ""

    def test_export_database_with_data(self):
        """Test exporting a database with patient records."""
        # Create a database first
        create_response = client.post("/dicom")
        database_id = create_response.json()["database_id"]
        
        # Add some patients
        patients_data = {
            "patients": [
                {"PatientID": "12345", "PatientName": "John Doe", "Modality": "CT"},
                {"PatientID": "12346", "PatientName": "Jane Smith", "Modality": "MR"}
            ]
        }
        client.post(f"/dicom/{database_id}/patients", json=patients_data)
        
        response = client.get(f"/dicom/{database_id}/export")
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        
        # Check CSV content
        csv_content = response.text
        lines = csv_content.strip().split("\n")
        assert len(lines) == 3  # Header + 2 data rows
        
        # Check header contains expected columns
        header = lines[0]
        assert "PatientID" in header
        assert "PatientName" in header
        assert "Modality" in header

    def test_export_nonexistent_database(self):
        """Test exporting a nonexistent database."""
        response = client.get("/dicom/nonexistent-id/export")
        assert response.status_code == 404

    def test_delete_database(self):
        """Test deleting a database."""
        # Create a database first
        create_response = client.post("/dicom")
        database_id = create_response.json()["database_id"]
        
        response = client.delete(f"/dicom/{database_id}")
        assert response.status_code == 200
        data = response.json()
        assert "deleted successfully" in data["message"]
        
        # Verify it's gone
        get_response = client.get(f"/dicom/{database_id}")
        assert get_response.status_code == 404

    def test_delete_nonexistent_database(self):
        """Test deleting a nonexistent database."""
        response = client.delete("/dicom/nonexistent-id")
        assert response.status_code == 404

    def test_database_info_after_upload(self):
        """Test that database info updates after patient uploads."""
        # Create a database
        create_response = client.post("/dicom")
        database_id = create_response.json()["database_id"]
        
        # Upload patients
        patients_data = {
            "patients": [
                {"PatientID": "12345", "PatientAge": "045Y", "Modality": "CT"},
                {"PatientID": "12346", "PatientAge": "032Y", "Modality": "MR"}
            ]
        }
        client.post(f"/dicom/{database_id}/patients", json=patients_data)
        
        # Check info
        response = client.get(f"/dicom/{database_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["patient_count"] == 2
        # Keys should contain all unique keys from patient records
        assert "PatientID" in data["keys"]
        assert "PatientAge" in data["keys"]
        assert "Modality" in data["keys"]

    def test_dashboard_empty_database(self):
        """Test that dashboard fails for empty database."""
        # Create a database
        create_response = client.post("/dicom")
        database_id = create_response.json()["database_id"]
        
        response = client.get(f"/dicom/{database_id}/dashboard")
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_dashboard_nonexistent_database(self):
        """Test that dashboard fails for nonexistent database."""
        response = client.get("/dicom/nonexistent-id/dashboard")
        assert response.status_code == 404


class TestDicomTags:
    """Test DICOM tags module."""

    def test_dicom_tags_loaded(self):
        """Test that DICOM tags are loaded correctly."""
        from src.modules.dicom_tags import DICOM_TAGS, VALID_DICOM_KEYWORDS
        
        assert len(DICOM_TAGS) > 0
        assert len(VALID_DICOM_KEYWORDS) > 0

    def test_tags_by_keyword(self):
        """Test looking up tags by keyword."""
        from src.modules.dicom_tags import DICOM_TAGS_BY_KEYWORD
        
        assert "PatientID" in DICOM_TAGS_BY_KEYWORD
        assert "Modality" in DICOM_TAGS_BY_KEYWORD
        assert "StudyDate" in DICOM_TAGS_BY_KEYWORD

    def test_tags_by_category(self):
        """Test looking up tags by category."""
        from src.modules.dicom_tags import get_tags_by_category, get_all_categories
        
        categories = get_all_categories()
        assert "demographics" in categories
        assert "study" in categories
        assert "imaging" in categories
        
        demographics = get_tags_by_category("demographics")
        assert len(demographics) > 0
        
        # PatientID should be in demographics
        keywords = [tag.keyword for tag in demographics]
        assert "PatientID" in keywords

    def test_get_tag_info(self):
        """Test getting tag info by keyword or tag."""
        from src.modules.dicom_tags import get_tag_info
        
        # By keyword
        info = get_tag_info("PatientID")
        assert info is not None
        assert info.keyword == "PatientID"
        assert info.category == "demographics"
        
        # By tag
        info = get_tag_info("(0010,0020)")
        assert info is not None
        assert info.keyword == "PatientID"
        
        # Nonexistent
        info = get_tag_info("NonexistentTag")
        assert info is None


class TestDicomStorage:
    """Test DICOM storage functionality."""

    def test_dicom_database_class(self):
        """Test DicomDatabase class."""
        from src.storage import DicomDatabase
        
        db = DicomDatabase("test-id")
        assert db.database_id == "test-id"
        assert db.get_patient_count() == 0
        assert db.get_all_keys() == []
        
        # Add a patient
        db.add_patient({"PatientID": "123", "Modality": "CT"})
        assert db.get_patient_count() == 1
        assert "PatientID" in db.get_all_keys()
        assert "Modality" in db.get_all_keys()
        
        # Add multiple patients
        db.add_patients([
            {"PatientID": "124", "PatientAge": "45Y"},
            {"PatientID": "125", "StudyDate": "20240101"}
        ])
        assert db.get_patient_count() == 3
        assert "PatientAge" in db.get_all_keys()
        assert "StudyDate" in db.get_all_keys()

    def test_storage_manager_dicom_methods(self):
        """Test storage manager DICOM methods."""
        from src.storage import storage_manager, DicomDatabase
        
        db_id = "test-storage-id"
        db = DicomDatabase(db_id)
        db.add_patient({"PatientID": "123"})
        
        # Store
        storage_manager.store_dicom(db_id, db)
        
        # Check exists
        assert storage_manager.check_dicom(db_id) is True
        assert storage_manager.check_dicom("nonexistent") is False
        
        # Get
        retrieved = storage_manager.get_dicom(db_id)
        assert retrieved is not None
        assert retrieved.database_id == db_id
        assert retrieved.get_patient_count() == 1
        
        # List
        ids = storage_manager.list_dicom_ids()
        assert db_id in ids
        
        # Delete
        assert storage_manager.delete_dicom(db_id) is True
        assert storage_manager.check_dicom(db_id) is False
        assert storage_manager.delete_dicom(db_id) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
