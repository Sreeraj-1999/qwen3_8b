import os
import logging
from typing import Dict
from pathlib import Path
import torch
import shutil
from database import get_database_manager
logger = logging.getLogger(__name__)

class VesselSpecificManager:
    """
    Manages all vessel-specific operations
    Each vessel gets its own isolated environment
    """
    
    def __init__(self, base_data_path: str = "./vessel_data"):
        self.base_data_path = Path(base_data_path)
        self.base_data_path.mkdir(exist_ok=True)
        
        # Cache for loaded vessel instances
        self._vessel_cache = {}
        
        logger.info(f"VesselSpecificManager initialized with base path: {self.base_data_path}")
    
    def get_vessel_instance(self, imo: str) -> 'VesselInstance':
        """Get or create vessel instance"""
        if imo not in self._vessel_cache:
            self._vessel_cache[imo] = VesselInstance(imo, self.base_data_path)
        return self._vessel_cache[imo]
    
    def clear_vessel_cache(self, imo: str = None):
        """Clear vessel cache for memory management"""
        if imo:
            if imo in self._vessel_cache:
                del self._vessel_cache[imo]
        else:
            self._vessel_cache.clear()
        torch.cuda.empty_cache()


class VesselInstance:
    """
    Individual vessel instance with its own tags and manuals
    """
    
    def __init__(self, imo: str, base_path: Path):
        self.imo = imo
        self.vessel_path = base_path / f"IMO{imo}"
        self.vessel_path.mkdir(exist_ok=True)
        
        # Vessel-specific paths
        self.tags_path = self.vessel_path / "tags"
        self.manuals_path = self.vessel_path / "manuals" 
        self.db_path = self.vessel_path / "databases"
        
        # Create directories
        self.tags_path.mkdir(exist_ok=True)
        self.manuals_path.mkdir(exist_ok=True)
        self.db_path.mkdir(exist_ok=True)
        
        # Initialize components
        self.tag_matcher = None
        self.manual_processor = None
        self._load_existing_tags()                #Added 
        
        logger.info(f"VesselInstance created for IMO{imo}")

    # def _load_existing_tags(self):
    #     """Load existing tags if Excel file exists"""                       ############### ADDED
    #     try:
    #         excel_path = self.tags_path / "current_tags.xlsx"
    #         if excel_path.exists():
    #             from tag_matcher1 import initialize_tag_matcher
    #             self.tag_matcher = initialize_tag_matcher(str(excel_path))
    #     except Exception as e:
    #         logger.error(f"Failed to auto-load tags for vessel IMO{self.imo}: {e}")
    #         self.tag_matcher = None
    def _load_existing_tags(self):
        """Load existing tags if Excel file exists"""
        try:
            excel_path = self.tags_path / "current_tags.xlsx"
            if excel_path.exists():
                from tag_matcher1 import MarineTagMatcher  # Import class directly
                self.tag_matcher = MarineTagMatcher(
                    excel_path=str(excel_path),
                    db_path=str(self.db_path / "tags_db"),
                    vessel_imo=self.imo  # Use vessel-specific path
                )
                logger.info(f"Auto-loaded existing tags for vessel IMO{self.imo}")
        except Exception as e:
            logger.error(f"Failed to auto-load tags for vessel IMO{self.imo}: {e}")
            self.tag_matcher = None    
    
    # ============= TAG MANAGEMENT =============
    
    def upload_tags_excel(self, file_path: str) -> Dict:
        """Upload and process vessel-specific tags Excel"""
        try:
            # Save to vessel-specific location
            excel_storage_path = self.tags_path / "current_tags.xlsx"
            
            # Copy uploaded file
            shutil.copy2(file_path, excel_storage_path)
            
            # Initialize vessel-specific tag matcher (USING YOUR EXISTING CLASS)
            from tag_matcher1 import initialize_tag_matcher
            # self.tag_matcher = initialize_tag_matcher(str(excel_storage_path))
            from tag_matcher1 import MarineTagMatcher
            self.tag_matcher = MarineTagMatcher(
                excel_path=str(excel_storage_path),
                db_path=str(self.db_path / "tags_db"),
                vessel_imo=self.imo
            )
                        
            torch.cuda.empty_cache()
            
            return {
                'status': 'success', 
                'message': f'Tags Excel uploaded for vessel IMO{self.imo}',
                'vessel_imo': self.imo
            }
            
        except Exception as e:
            logger.error(f"Error uploading tags for vessel IMO{self.imo}: {e}")
            return {'status': 'error', 'error': str(e), 'vessel_imo': self.imo}
        finally:
            torch.cuda.empty_cache() 
    def delete_tags_excel(self) -> Dict:
        """Delete vessel-specific tags"""
        try:
            excel_path = self.tags_path / "current_tags.xlsx"
            if excel_path.exists():
                excel_path.unlink()
            
            if self.tag_matcher:
                try:
                    self.tag_matcher.collection.delete()
                except:
                    pass
            
            self.tag_matcher = None
            torch.cuda.empty_cache()
            
            return {
                'status': 'success', 
                'message': f'Tags deleted for vessel IMO{self.imo}',
                'vessel_imo': self.imo
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e), 'vessel_imo': self.imo}
        finally:
            torch.cuda.empty_cache() 
    
    def analyze_alarm(self, alarm_name: str, possible_reasons: str, corrective_actions: str) -> Dict:
        """Analyze alarm for this specific vessel"""
        if not self.tag_matcher:
            return {
                'error': f'No tags uploaded for vessel IMO{self.imo}',
                'vessel_imo': self.imo
            }
        
        try:
            result = self.tag_matcher.match_tags(alarm_name, possible_reasons, corrective_actions)
            # Add vessel info to result
            result['vessel_imo'] = self.imo
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing alarm for vessel IMO{self.imo}: {e}")
            return {
                'error': str(e),
                'vessel_imo': self.imo,
                'alarm_name': alarm_name
            }
    
    # ============= MANUAL MANAGEMENT =============
    
    def get_manual_processor(self):
        """Get or create vessel-specific manual processor"""
        if not self.manual_processor:
            # from test import initialize_fixed_processor
            from test3 import initialize_fixed_processor
            self.manual_processor = initialize_fixed_processor(
                db_path=str(self.db_path / "manuals_db")
            )
        return self.manual_processor
    
    def upload_manual(self, file_path: str) -> Dict:
        """Upload and process vessel-specific manual"""
        try:
            processor = self.get_manual_processor()
            result = processor.process_document(file_path)
            db = get_database_manager()
            db.clear_vessel_alarm_cache(self.imo)
            result['vessel_imo'] = self.imo
            return result
            
        except Exception as e:
            logger.error(f"Error uploading manual for vessel IMO{self.imo}: {e}")
            return {
                'status': 'error', 
                'error': str(e), 
                'vessel_imo': self.imo
            }
    
    def query_manuals(self, question: str, n_results: int = 10) -> Dict:
        """Query vessel-specific manuals"""
        try:
            processor = self.get_manual_processor()
            result = processor.query_manuals(question, n_results)
            result['vessel_imo'] = self.imo
            return result
            
        except Exception as e:
            logger.error(f"Error querying manuals for vessel IMO{self.imo}: {e}")
            return {
                'error': str(e),
                'vessel_imo': self.imo,
                'question': question
            }
    
    def delete_manual(self, filename: str) -> Dict:
        """Delete specific manual for this vessel"""
        try:
            processor = self.get_manual_processor()
            success = processor.delete_document_by_name(filename)
            db = get_database_manager()
            db.clear_vessel_alarm_cache(self.imo)
                        
            if success:
                torch.cuda.empty_cache()
                return {
                    'status': 'success', 
                    'message': f'{filename} deleted from vessel IMO{self.imo}',
                    'vessel_imo': self.imo
                }
            else:
                return {
                    'error': 'Document not found',
                    'vessel_imo': self.imo,
                    'filename': filename
                }
                
        except Exception as e:
            return {
                'status': 'error', 
                'error': str(e), 
                'vessel_imo': self.imo
            }
    
    def list_manuals(self) -> Dict:
        """List all manuals for this vessel"""
        try:
            processor = self.get_manual_processor()
            stats = processor.get_stats()
            
            # Get unique document names
            collection = processor.vectorstore._collection
            data = collection.get()
            doc_names = set()
            for metadata in data.get('metadatas', []):
                if metadata and metadata.get('document_name'):
                    doc_names.add(metadata['document_name'])
            
            return {
                'vessel_imo': self.imo,
                'documents': list(doc_names),
                'total_chunks': stats.get('total_chunks', 0),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'vessel_imo': self.imo,
                'status': 'error'
            }