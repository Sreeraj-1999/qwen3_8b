import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import hashlib
from sqlalchemy import create_engine, Column, String, Text, DateTime, JSON, Integer,text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
import asyncpg
import asyncio

logger = logging.getLogger(__name__)

Base = declarative_base()

class AlarmAnalysis(Base):
    """
    PostgreSQL table for storing vessel-specific alarm analysis results
    """
    __tablename__ = "alarm_analysis"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    vessel_imo = Column(String(50), nullable=False, index=True)
    alarm_name = Column(String(500), nullable=False)
    alarm_hash = Column(String(64), nullable=False, index=True)  # MD5 hash for fast lookup
    possible_reasons = Column(Text, nullable=False)
    corrective_actions = Column(Text, nullable=False)
    suspected_tags = Column(JSON, nullable=True)  # List of tags
    analysis_metadata = Column(JSON, nullable=True)  # Additional metadata  # Additional metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<AlarmAnalysis(vessel_imo='{self.vessel_imo}', alarm='{self.alarm_name[:50]}...')>"


class DatabaseManager:
    """
    Manages PostgreSQL database operations for alarm caching
    """
    
    def __init__(self, database_url: str = None):
        if database_url is None:
            # Default PostgreSQL URL - modify as needed
            database_url = os.getenv(
                'DATABASE_URL', 
                'postgresql://postgres:Memphis%401234%21@localhost:5433/vessel_alarms'
            )
        
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        
        logger.info("Initializing DatabaseManager")
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize database connection and create tables"""
        try:
            self.engine = create_engine(
                self.database_url,
                pool_pre_ping=True,
                pool_recycle=300,
                echo=False  # Set to True for SQL debugging
            )
            
            self.SessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.engine
            )
            
            # Create tables if they don't exist
            Base.metadata.create_all(bind=self.engine)
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def get_db_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def _generate_alarm_hash(self, vessel_imo: str, alarm_name: str) -> str:
        """Generate unique hash for vessel + alarm combination"""
        combined = f"{vessel_imo}_{alarm_name.lower().strip()}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def check_alarm_cache(self, vessel_imo: str, alarm_name: str) -> Optional[Dict]:
        """
        Check if alarm analysis exists in cache for this vessel
        Returns cached result or None
        """
        alarm_hash = self._generate_alarm_hash(vessel_imo, alarm_name)
        
        try:
            with self.get_db_session() as session:
                cached = session.query(AlarmAnalysis).filter_by(
                    vessel_imo=vessel_imo,
                    alarm_hash=alarm_hash
                ).first()
                
                if cached:
                    logger.info(f"Cache HIT for vessel {vessel_imo}, alarm: {alarm_name}")
                    return {
                        'alarm_name': cached.alarm_name,
                        'possible_reasons': cached.possible_reasons,
                        'corrective_actions': cached.corrective_actions,
                        'suspected_tags': cached.suspected_tags or [],
                        'vessel_imo': cached.vessel_imo,
                        'cached': True,
                        'cached_at': cached.created_at.isoformat(),
                        'metadata': cached.analysis_metadata or {}
                    }
                else:
                    logger.info(f"Cache MISS for vessel {vessel_imo}, alarm: {alarm_name}")
                    return None
                    
        except SQLAlchemyError as e:
            logger.error(f"Database error checking cache: {e}")
            return None
    
    def store_alarm_analysis(self, 
                           vessel_imo: str, 
                           alarm_name: str,
                           possible_reasons: str,
                           corrective_actions: str,
                           suspected_tags: List[str] = None,
                           metadata: Dict = None) -> bool:
        """
        Store alarm analysis result in database
        Returns True if successful, False otherwise
        """
        alarm_hash = self._generate_alarm_hash(vessel_imo, alarm_name)
        
        try:
            with self.get_db_session() as session:
                # Check if already exists (update scenario)
                existing = session.query(AlarmAnalysis).filter_by(
                    vessel_imo=vessel_imo,
                    alarm_hash=alarm_hash
                ).first()
                
                if existing:
                    # Update existing record
                    existing.possible_reasons = possible_reasons
                    existing.corrective_actions = corrective_actions
                    existing.suspected_tags = suspected_tags or []
                    existing.analysis_metadata = metadata or {}
                    existing.updated_at = datetime.utcnow()
                    
                    logger.info(f"Updated alarm cache for vessel {vessel_imo}, alarm: {alarm_name}")
                else:
                    # Create new record
                    new_analysis = AlarmAnalysis(
                        vessel_imo=vessel_imo,
                        alarm_name=alarm_name,
                        alarm_hash=alarm_hash,
                        possible_reasons=possible_reasons,
                        corrective_actions=corrective_actions,
                        suspected_tags=suspected_tags or [],
                        analysis_metadata=metadata or {}
                    )
                    session.add(new_analysis)
                    
                    logger.info(f"Stored new alarm cache for vessel {vessel_imo}, alarm: {alarm_name}")
                
                session.commit()
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Database error storing alarm analysis: {e}")
            return False
    
    def get_vessel_alarm_history(self, vessel_imo: str, limit: int = 50) -> List[Dict]:
        """
        Get alarm analysis history for a specific vessel
        """
        try:
            with self.get_db_session() as session:
                results = session.query(AlarmAnalysis).filter_by(
                    vessel_imo=vessel_imo
                ).order_by(AlarmAnalysis.created_at.desc()).limit(limit).all()
                
                history = []
                for result in results:
                    history.append({
                        'alarm_name': result.alarm_name,
                        'possible_reasons': result.possible_reasons,
                        'corrective_actions': result.corrective_actions,
                        'suspected_tags': result.suspected_tags or [],
                        'created_at': result.created_at.isoformat(),
                        'updated_at': result.updated_at.isoformat(),
                        'metadata': result.metadata or {}
                    })
                
                return history
                
        except SQLAlchemyError as e:
            logger.error(f"Database error getting vessel history: {e}")
            return []
    
    def delete_vessel_alarms(self, vessel_imo: str) -> bool:
        """
        Delete all alarm analysis records for a specific vessel
        """
        try:
            with self.get_db_session() as session:
                deleted_count = session.query(AlarmAnalysis).filter_by(
                    vessel_imo=vessel_imo
                ).delete()
                
                session.commit()
                logger.info(f"Deleted {deleted_count} alarm records for vessel {vessel_imo}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Database error deleting vessel alarms: {e}")
            return False
    def clear_vessel_alarm_cache(self, vessel_imo: str) -> bool:
        """Clear all cached alarms for specific vessel"""
        return self.delete_vessel_alarms(vessel_imo)

    def get_database_stats(self) -> Dict:
        """
        Get database statistics
        """
        try:
            with self.get_db_session() as session:
                total_alarms = session.query(AlarmAnalysis).count()
                unique_vessels = session.query(AlarmAnalysis.vessel_imo).distinct().count()
                
                return {
                    'total_alarm_records': total_alarms,
                    'unique_vessels': unique_vessels,
                    'status': 'connected'
                }
                
        except SQLAlchemyError as e:
            logger.error(f"Database error getting stats: {e}")
            return {
                'total_alarm_records': 0,
                'unique_vessels': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def test_connection(self) -> bool:
        """
        Test database connection
        """
        try:
            with self.get_db_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False


# Global database manager instance
db_manager = None

def initialize_database(database_url: str = None) -> DatabaseManager:
    """
    Initialize global database manager
    Call this at app startup
    """
    global db_manager
    db_manager = DatabaseManager(database_url)
    return db_manager

def get_database_manager() -> DatabaseManager:
    """
    Get global database manager instance
    """
    global db_manager
    if db_manager is None:
        raise RuntimeError("Database not initialized. Call initialize_database() first.")
    return db_manager

def clear_vessel_alarm_cache(self, vessel_imo: str) -> bool:
    """Clear all cached alarms for specific vessel"""
    return self.delete_vessel_alarms(vessel_imo)

# Async version for FastAPI
class AsyncDatabaseManager:
    """
    Async version of database manager for FastAPI
    """
    
    def __init__(self, database_url: str = None):
        if database_url is None:
            # Convert PostgreSQL URL to asyncpg format
            database_url = os.getenv(
                'ASYNC_DATABASE_URL', 
                'postgresql://postgres:password@localhost:5432/vessel_alarms'
            )
        
        # Remove 'postgresql://' and replace with asyncpg format if needed
        if database_url.startswith('postgresql://'):
            self.database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://', 1)
        else:
            self.database_url = database_url
            
        logger.info("AsyncDatabaseManager initialized")
    
    async def check_alarm_cache_async(self, vessel_imo: str, alarm_name: str) -> Optional[Dict]:
        """
        Async version of cache check
        """
        # For now, use sync version in thread pool
        # In production, you might want to use SQLAlchemy async or asyncpg directly
        loop = asyncio.get_event_loop()
        sync_manager = get_database_manager()
        
        return await loop.run_in_executor(
            None, 
            sync_manager.check_alarm_cache, 
            vessel_imo, 
            alarm_name
        )
    
    async def store_alarm_analysis_async(self, 
                                       vessel_imo: str, 
                                       alarm_name: str,
                                       possible_reasons: str,
                                       corrective_actions: str,
                                       suspected_tags: List[str] = None,
                                       metadata: Dict = None) -> bool:
        """
        Async version of store operation
        """
        loop = asyncio.get_event_loop()
        sync_manager = get_database_manager()
        
        return await loop.run_in_executor(
            None,
            sync_manager.store_alarm_analysis,
            vessel_imo,
            alarm_name, 
            possible_reasons,
            corrective_actions,
            suspected_tags,
            metadata
        )