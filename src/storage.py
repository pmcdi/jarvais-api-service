import redis
from redis.exceptions import ConnectionError
import pickle
import logging
from typing import Dict, Optional, Protocol, List, Any
from datetime import datetime

from jarvais import Analyzer
from jarvais.trainer import TrainerSupervised
from .config import settings

logger = logging.getLogger(__name__)


class DicomDatabase:
    """In-memory representation of a DICOM patient database."""
    
    def __init__(self, database_id: str):
        self.database_id = database_id
        self.patients: List[Dict[str, Any]] = []
        self.created_at = datetime.now().isoformat()
    
    def add_patient(self, patient_data: Dict[str, Any]) -> int:
        """Add a single patient record. Returns the patient index."""
        self.patients.append(patient_data)
        return len(self.patients) - 1
    
    def add_patients(self, patients_data: List[Dict[str, Any]]) -> int:
        """Add multiple patient records. Returns the number added."""
        self.patients.extend(patients_data)
        return len(patients_data)
    
    def get_patient_count(self) -> int:
        """Get the number of patients in the database."""
        return len(self.patients)
    
    def get_all_keys(self) -> List[str]:
        """Get all unique keys across all patient records."""
        keys = set()
        for patient in self.patients:
            keys.update(patient.keys())
        return sorted(list(keys))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert database to a dictionary for serialization."""
        return {
            'database_id': self.database_id,
            'patients': self.patients,
            'created_at': self.created_at,
            'patient_count': len(self.patients),
            'keys': self.get_all_keys()
        }


class StorageBackend(Protocol):
    """Protocol for storage backends."""
    
    def exists(self, analyzer_id: str) -> bool:
        """Check if analyzer exists."""
        ...
    
    def store(self, analyzer_id: str, analyzer: Analyzer) -> None:
        """Store analyzer instance."""
        ...
    
    def get(self, analyzer_id: str) -> Optional[Analyzer]:
        """Retrieve analyzer instance."""
        ...
    
    def delete(self, analyzer_id: str) -> bool:
        """Delete analyzer instance."""
        ...
    
    def list_ids(self) -> List[str]:
        """List all analyzer IDs."""
        ...


class RedisStorage:
    """Redis-based storage backend."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.analyzer_prefix = "analyzer:"
        self.trainer_prefix = "trainer:"
    
    def exists(self, analyzer_id: str) -> bool:
        """Check if analyzer exists in Redis."""
        try:
            return self.redis_client.exists(f"{self.analyzer_prefix}{analyzer_id}") == 1
        except ConnectionError as e: 
            logger.error(f"Redis connection error: {e}")
            return False
    
    def store(self, analyzer_id: str, analyzer: Analyzer) -> None:
        """Store analyzer instance in Redis."""
        serialized = pickle.dumps(analyzer)
        self.redis_client.setex(
            f"{self.analyzer_prefix}{analyzer_id}",
            settings.session_ttl,
            serialized
        )
    
    def get(self, analyzer_id: str) -> Optional[Analyzer]:
        """Retrieve analyzer instance from Redis."""
        serialized = self.redis_client.get(f"{self.analyzer_prefix}{analyzer_id}")
        if serialized:
            return pickle.loads(serialized) # type: ignore
        return None
    
    def delete(self, analyzer_id: str) -> bool:
        """Delete analyzer instance from Redis."""
        return self.redis_client.delete(f"{self.analyzer_prefix}{analyzer_id}") > 0 # type: ignore
    
    def list_ids(self) -> List[str]:
        """List all analyzer IDs in Redis."""
        keys = self.redis_client.keys(f"{self.analyzer_prefix}*")
        return [key.decode('utf-8').split(':', 1)[1] for key in keys] # type: ignore
    
    # Trainer-specific methods
    def exists_trainer(self, trainer_id: str) -> bool:
        """Check if trainer exists in Redis."""
        try:
            return self.redis_client.exists(f"{self.trainer_prefix}{trainer_id}") == 1
        except ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            return False
    
    def store_trainer(self, trainer_id: str, trainer_data: Dict[str, Any]) -> None:
        """Store trainer data in Redis."""
        serialized = pickle.dumps(trainer_data)
        self.redis_client.setex(
            f"{self.trainer_prefix}{trainer_id}",
            settings.session_ttl,
            serialized
        )
    
    def get_trainer(self, trainer_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve trainer data from Redis."""
        serialized = self.redis_client.get(f"{self.trainer_prefix}{trainer_id}")
        if serialized:
            return pickle.loads(serialized) # type: ignore
        return None
    
    def delete_trainer(self, trainer_id: str) -> bool:
        """Delete trainer from Redis."""
        return self.redis_client.delete(f"{self.trainer_prefix}{trainer_id}") > 0 # type: ignore
    
    def list_trainer_ids(self) -> List[str]:
        """List all trainer IDs in Redis."""
        keys = self.redis_client.keys(f"{self.trainer_prefix}*")
        return [key.decode('utf-8').split(':', 1)[1] for key in keys] # type: ignore
    
    # DICOM Database methods
    def exists_dicom(self, database_id: str) -> bool:
        """Check if DICOM database exists in Redis."""
        try:
            return self.redis_client.exists(f"dicom:{database_id}") == 1
        except ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            return False
    
    def store_dicom(self, database_id: str, database: DicomDatabase) -> None:
        """Store DICOM database in Redis."""
        serialized = pickle.dumps(database)
        self.redis_client.setex(
            f"dicom:{database_id}",
            settings.session_ttl,
            serialized
        )
    
    def get_dicom(self, database_id: str) -> Optional[DicomDatabase]:
        """Retrieve DICOM database from Redis."""
        serialized = self.redis_client.get(f"dicom:{database_id}")
        if serialized:
            return pickle.loads(serialized)  # type: ignore
        return None
    
    def delete_dicom(self, database_id: str) -> bool:
        """Delete DICOM database from Redis."""
        return self.redis_client.delete(f"dicom:{database_id}") > 0  # type: ignore
    
    def list_dicom_ids(self) -> List[str]:
        """List all DICOM database IDs in Redis."""
        keys = self.redis_client.keys("dicom:*")
        return [key.decode('utf-8').split(':', 1)[1] for key in keys]  # type: ignore


class MemoryStorage:
    """In-memory storage backend."""
    
    def __init__(self):
        self.analyzers: Dict[str, Analyzer] = {}
        self.trainers: Dict[str, Dict[str, Any]] = {}
        self.dicom_databases: Dict[str, DicomDatabase] = {}
    
    def exists(self, analyzer_id: str) -> bool:
        """Check if analyzer exists in memory."""
        return analyzer_id in self.analyzers
    
    def store(self, analyzer_id: str, analyzer: Analyzer) -> None:
        """Store analyzer instance in memory."""
        self.analyzers[analyzer_id] = analyzer
    
    def get(self, analyzer_id: str) -> Optional[Analyzer]:
        """Retrieve analyzer instance from memory."""
        return self.analyzers.get(analyzer_id)
    
    def delete(self, analyzer_id: str) -> bool:
        """Delete analyzer instance from memory."""
        if analyzer_id in self.analyzers:
            del self.analyzers[analyzer_id]
            return True
        return False
    
    def list_ids(self) -> List[str]:
        """List all analyzer IDs in memory."""
        return list(self.analyzers.keys())
    
    # Trainer-specific methods
    def exists_trainer(self, trainer_id: str) -> bool:
        """Check if trainer exists in memory."""
        return trainer_id in self.trainers
    
    def store_trainer(self, trainer_id: str, trainer_data: Dict[str, Any]) -> None:
        """Store trainer data in memory."""
        self.trainers[trainer_id] = trainer_data
    
    def get_trainer(self, trainer_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve trainer data from memory."""
        return self.trainers.get(trainer_id)
    
    def delete_trainer(self, trainer_id: str) -> bool:
        """Delete trainer from memory."""
        if trainer_id in self.trainers:
            del self.trainers[trainer_id]
            return True
        return False
    
    def list_trainer_ids(self) -> List[str]:
        """List all trainer IDs in memory."""
        return list(self.trainers.keys())
    
    # DICOM Database methods
    def exists_dicom(self, database_id: str) -> bool:
        """Check if DICOM database exists in memory."""
        return database_id in self.dicom_databases
    
    def store_dicom(self, database_id: str, database: DicomDatabase) -> None:
        """Store DICOM database in memory."""
        self.dicom_databases[database_id] = database
    
    def get_dicom(self, database_id: str) -> Optional[DicomDatabase]:
        """Retrieve DICOM database from memory."""
        return self.dicom_databases.get(database_id)
    
    def delete_dicom(self, database_id: str) -> bool:
        """Delete DICOM database from memory."""
        if database_id in self.dicom_databases:
            del self.dicom_databases[database_id]
            return True
        return False
    
    def list_dicom_ids(self) -> List[str]:
        """List all DICOM database IDs in memory."""
        return list(self.dicom_databases.keys())


class StorageManager:
    """Manages storage backend with automatic fallback."""
    
    def __init__(self):
        self._setup_backend()
    
    def _setup_backend(self):
        """Setup storage backend with Redis fallback to memory."""
        try:
            redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            redis_client.ping()
            self.backend = RedisStorage(redis_client)
            self.use_redis = True
            logger.info(f"Connected to Redis storage at {settings.redis_host}:{settings.redis_port}")
        except Exception as e:
            self.backend = MemoryStorage()
            self.use_redis = False
            logger.warning(f"Redis not available, using in-memory storage: {e}")
    
    def get_storage_type(self) -> str:
        """Get current storage type."""
        return "redis" if self.use_redis else "memory"
    
    def check_analyzer(self, analyzer_id: str) -> bool:
        """Check if analyzer exists."""
        return self.backend.exists(analyzer_id)
    
    def store_analyzer(self, analyzer_id: str, analyzer: Analyzer) -> None:
        """Store analyzer instance."""
        self.backend.store(analyzer_id, analyzer)
    
    def get_analyzer(self, analyzer_id: str) -> Optional[Analyzer]:
        """Retrieve analyzer instance."""
        return self.backend.get(analyzer_id)
    
    def delete_analyzer(self, analyzer_id: str) -> bool:
        """Delete analyzer instance."""
        return self.backend.delete(analyzer_id)
    
    def list_analyzer_ids(self) -> List[str]:
        """List all analyzer IDs."""
        return self.backend.list_ids()
    
    # Trainer management methods
    def check_trainer(self, trainer_id: str) -> bool:
        """Check if trainer exists."""
        return self.backend.exists_trainer(trainer_id)
    
    def store_trainer(self, trainer_id: str, trainer_data: Dict[str, Any]) -> None:
        """Store trainer data."""
        self.backend.store_trainer(trainer_id, trainer_data)
    
    def get_trainer(self, trainer_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve trainer data."""
        return self.backend.get_trainer(trainer_id)
    
    def delete_trainer(self, trainer_id: str) -> bool:
        """Delete trainer."""
        return self.backend.delete_trainer(trainer_id)
    
    def list_trainer_ids(self) -> List[str]:
        """List all trainer IDs."""
        return self.backend.list_trainer_ids()
    
    # DICOM database management methods
    def check_dicom(self, database_id: str) -> bool:
        """Check if DICOM database exists."""
        return self.backend.exists_dicom(database_id)
    
    def store_dicom(self, database_id: str, database: DicomDatabase) -> None:
        """Store DICOM database."""
        self.backend.store_dicom(database_id, database)
    
    def get_dicom(self, database_id: str) -> Optional[DicomDatabase]:
        """Retrieve DICOM database."""
        return self.backend.get_dicom(database_id)
    
    def delete_dicom(self, database_id: str) -> bool:
        """Delete DICOM database."""
        return self.backend.delete_dicom(database_id)
    
    def list_dicom_ids(self) -> List[str]:
        """List all DICOM database IDs."""
        return self.backend.list_dicom_ids()
    
    def health_check(self) -> dict:
        """Perform health check on storage backend."""
        health_info = {
            'storage_type': self.get_storage_type(),
            'status': 'healthy'
        }
        
        if self.use_redis:
            try:
                self.backend.redis_client.ping() # type: ignore
                health_info['redis'] = 'connected'
            except Exception as e:
                logger.error(f"Redis health check failed: {e}")
                health_info['status'] = 'degraded'
                health_info['redis'] = 'disconnected'
        
        return health_info


# Global storage manager instance
storage_manager = StorageManager() 