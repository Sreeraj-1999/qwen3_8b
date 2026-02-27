import pandas as pd
import chromadb
from chromadb.config import Settings
import re
# from sentence_transformers import SentenceTransformer
from embedding_service import embedding_service
import logging
import os
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict
import csv
import torch

logger = logging.getLogger(__name__)

class MarineTagMatcher:
    """
    Marine vessel tag matching system with strict engine type and numbering rules
    """
    
    def __init__(self, excel_path: str, db_path: str = "./chroma_db",vessel_imo: str = None):
        self.excel_path = excel_path
        self.db_path = db_path
        self.vessel_imo = vessel_imo
        self.max_results = 10  # Reasonable limit for accuracy

        if vessel_imo:
            self.collection_name = f"marine_tags_imo{vessel_imo}"
        else:
            self.collection_name = "marine_tags"
        
        # Engine type categorization - STRICT RULES
        self.engine_categories = {
            "ME": ["main engine", "main propulsion", "propulsion engine", "me_", "main_engine"],
            "AE": ["auxiliary engine", "aux engine", "ae_", "auxiliary_engine"],
            "GE": ["generator engine", "gen engine", "generator", "ge_", "generator_engine"],
            "DG": ["diesel generator", "emergency generator", "dg_", "diesel_generator"],
            "AB": ["auxiliary boiler", "aux boiler", "boiler", "ab_"]
        }
        
        # Cross-reference rules: ME is isolated, others can cross-reference
        self.cross_reference_allowed = {
            "ME": [],  # ME NEVER cross-references
            "AE": ["GE", "DG"],
            "GE": ["AE", "DG"], 
            "DG": ["AE", "GE"],
            "AB": [] 
        }
        
        logger.info("Initializing Marine Tag Matcher...")
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize embedding model and ChromaDB"""
        logger.info("Loading embedding model: BAAI/bge-large-en-v1.5")
        # logger.info("Loading embedding model: google/embeddinggemma-300m")
        # self.embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
        self.embedding_model = embedding_service.get_model()
        self.embedding_model = None
        
        logger.info("Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=self.db_path)
        
        # Try to get existing collection or create new one
        # CHANGE THIS SECTION (around line 78-84):
        try:
            self.collection = self.chroma_client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
        except Exception:
            logger.info(f"Creating new ChromaDB collection: {self.collection_name}")
            self._create_tag_database()
            
    
    def _create_tag_database(self):
        """Load Excel and create ChromaDB collection"""
        logger.info(f"Loading Excel file: {self.excel_path}")
        
        try:
            
            df = pd.read_excel(self.excel_path)
            logger.info(f"Loaded {len(df)} rows from Excel")
        except Exception as e:
            logger.error(f"Failed to load Excel: {e}")
            raise
        
        # Validate required columns
        if 'tag_from_vessel' not in df.columns or 'description' not in df.columns:
            raise ValueError("Excel must contain 'tag_from_vessel' and 'description' columns")
        
        # Clean and prepare data
        df = df.dropna(subset=['tag_from_vessel', 'description'])
        df['tag_from_vessel'] = df['tag_from_vessel'].astype(str).str.strip()
        df['description'] = df['description'].astype(str).str.strip()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['tag_from_vessel'])
        
        logger.info(f"Processing {len(df)} unique tags")
        
        # Create collection
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Process in batches due to ChromaDB limits
        batch_size = 160  # Under the 166 limit
        
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            
            # Prepare data for ChromaDB
            ids = batch_df['tag_from_vessel'].tolist()
            documents = batch_df['description'].tolist()
            
            # Add metadata for each tag
            metadatas = []
            for _, row in batch_df.iterrows():
                engine_number = self._extract_engine_number(row['tag_from_vessel'])
                secondary_number = self._extract_secondary_number(row['tag_from_vessel'])
                # print(f"DEBUG DB: Tag '{row['tag_from_vessel']}' -> engine: '{engine_number}', secondary: '{secondary_number}'")  # Add this line
                metadata = {
                    'tag': row['tag_from_vessel'],
                    'description': row['description'],
                    'engine_type': self._classify_engine_type(row['tag_from_vessel'], row['description']),
                    'engine_number': engine_number if engine_number is not None else "",
                    'secondary_number': secondary_number if secondary_number is not None else ""  # Add this    # Convert None to empty string
                }
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added batch {i//batch_size + 1}: {len(batch_df)} tags")
        
        logger.info("ChromaDB collection created successfully")
        torch.cuda.empty_cache()
    
    def _classify_engine_type(self, tag: str, description: str) -> str:
        """Classify engine type from tag and description"""
        text = f"{tag.lower()} {description.lower()}"
        
        # Check each engine category
        for engine_type, keywords in self.engine_categories.items():
            for keyword in keywords:
                if keyword in text:
                    return engine_type
        
        return "UNKNOWN"
    
    def _extract_engine_number(self, tag: str) -> Optional[str]:
        """Extract engine number from tag"""

        # Look for patterns like "1", "NO.1", "ENGINE 1", "G_E4_", etc.
    #     patterns = [
    # r'\bno\.?\s*(\d+)\b',     # NO.1, NO 1
    # r'\bengine\s+(\d+)\b',    # ENGINE 1
    # r'\bgen\s+(\d+)\b',       # GEN 1  
    # r'\b(\d+)\s+engine\b',    # 1 ENGINE
    # r'_e(\d+)_',              # _E4_
    # r'_(\d+)_',               # _4_
    # r'\be(\d+)\b', 
    # r'\bno(\d+)\b',                     
    # r'\bg_e(\d+)_', 
    # r'\bno(\d+)[\b_]',                    # G_E4_
    # r'(\d+)$'                 # ending with number
    #     ]
#         patterns = [
#     r'\bmain\s+engine\s+no(\d+)\b',       # Main Engine NO2 (NEW - add this)
#     r'\baux(?:iliary)?\s+engine\s+no(\d+)\b',  # Auxiliary Engine NO2 (NEW - add this) 
#     r'\bno\.?\s*(\d+)\b',                 # NO.1, NO 1 (KEEP - works for many tags)
#     r'\bno(\d+)(?=_|\W|$)',               # NO1_ (KEEP - works for steering gear)
#     r'\bengine\s+(\d+)\b',                # ENGINE 1 (KEEP)
#     r'\bgen\s+(\d+)\b',                   # GEN 1 (KEEP)
#     r'\b(\d+)\s+engine\b',                # 1 ENGINE (KEEP)
#     r'_e(\d+)_',                          # _E4_ (KEEP)
#     r'_(\d+)_',                           # _4_ (KEEP)
#     r'\be(\d+)\b',                        # (KEEP)
#     r'\bg_e(\d+)_',                       # G_E4_ (KEEP)
#     r'(\d+)$'                             # ending with number (KEEP)
# ]
        patterns = [
    r'\bmain\s+engine\s+no(\d+)\b',       # Main Engine NO2
    r'\baux(?:iliary)?\s+engine\s+no(\d+)\b',  # Auxiliary Engine NO2
    # r'\bgen(?:erator)?\s+engine\s+no(\d+)\b',  # Generator Engine NO2
    r'\bgen(?:erator)?\s+engine\s+(\d+)\b',    # Generator Engine 1 (extract the engine number)
    r'\bgenerator\s+(\d+)\b',
    r'\bme_no(\d+)_',                     # ME_NO2_
    r'\bm_e_no(\d+)_',                    # M_E_NO2_
    r'\bae_no(\d+)_',                     # AE_NO2_
    r'\ba_e_no(\d+)_',                    # A_E_NO2_
    r'\bge_no(\d+)_',                     # GE_NO2_
    r'\bg_e_no(\d+)_',                    # G_E_NO2_
    r'\bdg_no(\d+)_',                     # DG_NO2_
    r'\bd_g_no(\d+)_',                    # D_G_NO2_
    r'\bno\.?\s*(\d+)\b',                 # NO.1, NO 1
    r'\bno(\d+)(?=_|\W|$)',               # NO1_
    r'\bengine\s+(\d+)\b',                # ENGINE 1
    r'\bgen\s+(\d+)\b',                   # GEN 1
    r'\b(\d+)\s+engine\b',                # 1 ENGINE
    r'_e(\d+)_',                          # _E4_
    r'_(\d+)_',                           # _4_
    r'\be(\d+)\b',
    r'\bg_e(\d+)_',                       # G_E4_
    r'(\d+)$',                             # ending with number
    r'\bgen(\d+)_',                          # GEN1_ (ADD THIS)
]
    
        tag_lower = tag.lower()
    
   

        # tag_lower = tag.lower()
        # print(f"DEBUG EXTRACT: Processing tag '{tag}' -> '{tag_lower}'")
       
        for i, pattern in enumerate(patterns):
            match = re.search(pattern, tag_lower)
            if match:
                # print(f"DEBUG EXTRACT: Pattern {i} '{pattern}' matched! Extracted: '{match.group(1)}'")
                return match.group(1)
            # else:
            #     print(f"DEBUG EXTRACT: Pattern {i} '{pattern}' - no match")
    
        # print(f"DEBUG EXTRACT: No patterns matched for '{tag}'")
        # self._log_no_match_to_csv(tag, "engine_number")
        
        return None
    def _log_no_match_to_csv(self, tag: str, extraction_type: str):
        """Log tags with no pattern matches to CSV"""
        csv_file = f"no_patterns_matched_{extraction_type}.csv"
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Tag', 'Extraction_Type', 'Timestamp'])
            
            from datetime import datetime
            writer.writerow([tag, extraction_type, datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
    
    def _get_allowed_engine_types(self, alarm_engine_type: str) -> List[str]:
        """Get allowed engine types based on cross-reference rules"""
        allowed = [alarm_engine_type]  # Always include same type
        
        if alarm_engine_type in self.cross_reference_allowed:
            allowed.extend(self.cross_reference_allowed[alarm_engine_type])
        if alarm_engine_type != "ME" and "UNKNOWN" not in allowed:
            allowed.append("UNKNOWN")    
        
        return allowed
        
    def _extract_secondary_number(self, tag: str) -> Optional[str]:

        """Extract secondary numbers like cylinder, pump, valve numbers"""
        patterns = [
       r'_no(\d+)_cyl',                     # _NO1_CYL, _NO4_CYL (most specific first)
       r'\bno(\d+)\s+cylinder\b',           # NO2 CYLINDER
       r'\bcylinder\s+no(\d+)\b',           # CYLINDER NO2  
       r'\bno(\d+)\s+pump\b',               # NO2 PUMP
       r'\bpump\s+no(\d+)\b',               # PUMP NO2
       r'\bno(\d+)\s+valve\b',              # NO2 VALVE
       r'\bvalve\s+no(\d+)\b',              # VALVE NO2
       r'_no(\d+)_',                        # Generic _NO1_, _NO4_ (fallback)
       r'\bno(\d+)\s+\w+\b',                # NO2 [anything] (generic fallback)
   ]
        tag_lower = tag.lower()
        for pattern in patterns:
            match = re.search(pattern, tag_lower)
            if match:
                return match.group(1)
        # self._log_no_match_to_csv(tag, "secondary_number")    
        return None        
    
    def _filter_by_engine_rules(self, alarm_name: str) -> Dict:
        """Apply engine type and numbering filters"""
        # Determine alarm engine type and number
        alarm_text = alarm_name.lower()
        
        alarm_engine_type = "UNKNOWN"
        for engine_type, keywords in self.engine_categories.items():
            for keyword in keywords:
                if keyword in alarm_text:
                    alarm_engine_type = engine_type
                    break
            if alarm_engine_type != "UNKNOWN":
                break
        
        # Extract alarm number
        alarm_number = self._extract_engine_number(alarm_name)
        alarm_secondary_number = self._extract_secondary_number(alarm_name)
        
        # Get allowed engine types
        allowed_engine_types = self._get_allowed_engine_types(alarm_engine_type)
        
        # Always include UNKNOWN (independent/general tags)
        if "UNKNOWN" not in allowed_engine_types:
            allowed_engine_types.append("UNKNOWN")
        
        # logger.info(f"Alarm engine type: {alarm_engine_type}, number: {alarm_number}")
        # logger.info(f"Allowed engine types: {allowed_engine_types}")
        
        return {
            'allowed_engine_types': allowed_engine_types,
            'alarm_number': alarm_number,
            'alarm_secondary_number': alarm_secondary_number 
        }
    
    def match_tags(self, alarm_name: str, possible_reasons: str, corrective_actions: str) -> Dict:
        """
        Match relevant vessel tags based on LLM response and alarm context
        
        Args:
            alarm_name: Name of the alarm
            possible_reasons: LLM generated possible reasons
            corrective_actions: LLM generated corrective actions  
            
        Returns:
            Dictionary with matched tags and metadata
        """
        logger.info(f"Matching tags for alarm: {alarm_name}")
        
        # Apply engine type and numbering filters
        filter_rules = self._filter_by_engine_rules(alarm_name)
        
        # Combine all text for semantic search
        search_text = f"{alarm_name} {possible_reasons} {corrective_actions}"
        
        # Build where clause for filtering
        if len(filter_rules['allowed_engine_types']) == 1:
            # Single condition - no need for $or
            where_conditions = {"engine_type": filter_rules['allowed_engine_types'][0]}
        else:
            # Multiple conditions - use $or
            where_conditions = {"$or": []}
            for engine_type in filter_rules['allowed_engine_types']:
                condition = {"engine_type": engine_type}
                where_conditions["$or"].append(condition)
        
        # Query ChromaDB with filters
        try:
            results = self.collection.query(
                query_texts=[search_text],
                n_results=min(50, self.collection.count()),  # Get more for filtering
                where=where_conditions
            )
            
            # Post-process results for numbering rules
            filtered_results = self._apply_numbering_filter(
                results, filter_rules['alarm_number'],filter_rules['alarm_secondary_number'] 
            )
            
            # Format response
            matched_tags = []
            if filtered_results['ids'][0]:  # Check if we have results
                for i, (tag_id, distance, metadata) in enumerate(zip(
                    filtered_results['ids'][0][:self.max_results],
                    filtered_results['distances'][0][:self.max_results], 
                    filtered_results['metadatas'][0][:self.max_results]
                )):
                    similarity_score = 1 - distance  # Convert distance to similarity
                    # print(f"DEBUG METADATA: {metadata}")
                    tag_value = metadata.get('tag')
                    # print(f"DEBUG TAG VALUE: '{tag_value}' (type: {type(tag_value)})")
                    if tag_value:  # Check if tag exists and is truthy
                        matched_tags.append(tag_value)
                    else:
                        print(f"DEBUG: Skipping null/empty tag: {tag_value}")
                    # if metadata.get('tag'):
                    #     matched_tags.append(
                    #         # 'tag': metadata['tag'],
                    #         matched_tags.append(metadata['tag'])
                    #         # 'description': metadata['description'],
                    #         # 'engine_type': metadata['engine_type'],
                    #         # 'engine_number': metadata.get('engine_number', "") or "",
                    #         # 'similarity_score': round(similarity_score, 4),
                    #         # 'rank': i + 1
                    #     )
            
            # logger.info(f"Found {len(matched_tags)} matching tags")
            
            return {
                'alarm_name': alarm_name,
                'possible_reasons': possible_reasons,
                'corrective_actions': corrective_actions,
                'suspected_tags': matched_tags,
            }
            
        except Exception as e:
            logger.error(f"Error during tag matching: {e}")
            return {
                'alarm_name': alarm_name,
                'possible_reasons': possible_reasons,
                'corrective_actions': corrective_actions,
                'suspected_tags': [],
                'error': str(e)
            }
    
    def _apply_numbering_filter(self, results: Dict, alarm_number: Optional[str],alarm_secondary_number: Optional[str]) -> Dict:
        """Apply strict numbering filter to ChromaDB results"""
        # if not alarm_number or not results['ids'][0]:
        #     return results
        if not results['ids'][0]:  # Only check if results exist
            return results
        
        filtered_ids = []
        filtered_distances = []
        filtered_metadatas = []
        
        # logger.info(f"Applying numbering filter for alarm_number: {alarm_number}")
        
        for tag_id, distance, metadata in zip(
            results['ids'][0], results['distances'][0], results['metadatas'][0]
        ):
            tag_number = metadata.get('engine_number', "")
            tag_secondary_number = metadata.get('secondary_number', "") 
            tag_name = metadata.get('tag', "")
            # print(f"DEBUG: Checking tag {tag_name}, stored engine_number: '{tag_number}', alarm_number: '{alarm_number}'")  # Add this line
            engine_match = tag_number == "" or tag_number == alarm_number
            secondary_match = tag_secondary_number == "" or not alarm_secondary_number or tag_secondary_number == alarm_secondary_number
            
            # Include if: same number OR no number (empty string = generic tags)  
            # if tag_number == "" or tag_number == alarm_number:
            #     filtered_ids.append(tag_id)
            #     filtered_distances.append(distance)
            #     filtered_metadatas.append(metadata)
            #     logger.debug(f"INCLUDED: {tag_name} (number: '{tag_number}')")
            # else:
            #     logger.debug(f"EXCLUDED: {tag_name} (number: '{tag_number}', wanted: '{alarm_number}')")
            if engine_match and secondary_match:
                filtered_ids.append(tag_id)
                filtered_distances.append(distance)
                filtered_metadatas.append(metadata)
                logger.debug(f"INCLUDED: {tag_name} (engine: '{tag_number}', secondary: '{tag_secondary_number}')")
            else:
                logger.debug(f"EXCLUDED: {tag_name} (engine: '{tag_number}', secondary: '{tag_secondary_number}', wanted engine: '{alarm_number}', wanted secondary: '{alarm_secondary_number}')")
        
        # logger.info(f"Filtered from {len(results['ids'][0])} to {len(filtered_ids)} tags")
                
         
        
        return {
            'ids': [filtered_ids],
            'distances': [filtered_distances],
            'metadatas': [filtered_metadatas]
        }
    
    def get_collection_stats(self) -> Dict:
        """Get database statistics"""
        try:
            count = self.collection.count()
            return {
                'total_tags': count,
                'status': 'ready'
            }
        except Exception as e:
            return {
                'total_tags': 0,
                'status': 'error',
                'error': str(e)
            }


# Integration function for Flask app
def initialize_tag_matcher(excel_path: str) -> MarineTagMatcher:
    """Initialize tag matcher - call this at app startup"""
    return MarineTagMatcher(excel_path)


def enhanced_alarm_analysis(tag_matcher: MarineTagMatcher, alarm_name: str, 
                          possible_reasons: str, corrective_actions: str) -> Dict:
    """Enhanced alarm analysis with tag matching"""
    return tag_matcher.match_tags(alarm_name, possible_reasons, corrective_actions)