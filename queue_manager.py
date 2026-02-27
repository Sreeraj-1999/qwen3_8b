import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
import uuid
import json
from enum import IntEnum
from dataclasses import dataclass
import requests
import time

logger = logging.getLogger(__name__)

class Priority(IntEnum):
    """Priority levels for queue operations"""
    CHAT = 1          # Highest - immediate processing
    AUDIO = 2         # High - real-time operations  
    FACE_RECOGNITION = 1  # High - real-time operations            # 2
    MANUAL_QUERY = 3  # Medium - user waiting
    MAINTENANCE_PREDICTION = 3
    FUEL_ANALYSIS = 3 # Medium - user waiting
    ALARM_ANALYSIS = 4    # Lower - can wait
    MANUAL_UPLOAD = 5     # Lowest - background task

@dataclass
class QueueTask:
    """Represents a task in the queue"""
    id: str
    priority: Priority
    task_type: str
    endpoint: str
    payload: Dict[Any, Any]
    vessel_imo: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    
    def __lt__(self, other):
        """Enable comparison for priority queue"""
        return self.priority < other.priority

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()

class GPUQueueManager:
    """
    Manages priority queue for GPU operations
    Chat operations bypass queue, others are queued by priority
    """
    
    def __init__(self, gpu_service_url: str = "http://localhost:5005"):
        self.gpu_service_url = gpu_service_url.rstrip('/')
        self.queue = asyncio.PriorityQueue()
        self.active_tasks = {}  # task_id -> QueueTask
        self.is_processing = False
        self.current_task = None
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'chat_bypassed': 0,
            'queued_tasks': 0,
            'failed_tasks': 0,
            'average_wait_time': 0.0
        }
        
        logger.info(f"GPUQueueManager initialized with service URL: {gpu_service_url}")
    
    def _make_gpu_request(self, endpoint: str, payload: Dict) -> Dict:
        """Make HTTP request to GPU service"""
        try:
            url = f"{self.gpu_service_url}{endpoint}"
            
            # Handle different request types
            if endpoint.startswith('/gpu/stt') or endpoint.startswith('/gpu/face'):
                # File upload endpoints
                files = payload.get('files', {})
                data = {k: v for k, v in payload.items() if k != 'files'}
                response = requests.post(url, files=files, data=data, timeout=60)
            else:
                # JSON endpoints
                response = requests.post(url, json=payload, timeout=60)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"GPU service request failed: {e}")
            return {'error': str(e), 'status': 'gpu_service_error'}
        except Exception as e:
            logger.error(f"Unexpected error in GPU request: {e}")
            return {'error': str(e), 'status': 'unknown_error'}
    
    async def process_immediately(self, task_type: str, endpoint: str, payload: Dict, vessel_imo: str = None) -> Dict:
        """
        Process task immediately (for CHAT operations)
        Bypasses queue entirely
        """
        task_id = str(uuid.uuid4())
        
        logger.info(f"Processing IMMEDIATE task {task_id}: {task_type}")
        
        task = QueueTask(
            id=task_id,
            priority=Priority.CHAT,
            task_type=task_type,
            endpoint=endpoint,
            payload=payload,
            vessel_imo=vessel_imo
        )
        
        task.started_at = datetime.utcnow()
        
        try:
            # Make GPU service call
            result = await asyncio.get_event_loop().run_in_executor(
                None, self._make_gpu_request, endpoint, payload
            )
            
            task.completed_at = datetime.utcnow()
            task.result = result
            
            # Update stats
            self.stats['chat_bypassed'] += 1
            self.stats['total_processed'] += 1
            
            logger.info(f"IMMEDIATE task {task_id} completed successfully")
            return result
            
        except Exception as e:
            task.error = str(e)
            task.completed_at = datetime.utcnow()
            
            self.stats['failed_tasks'] += 1
            logger.error(f"IMMEDIATE task {task_id} failed: {e}")
            
            return {'error': str(e), 'status': 'processing_error'}
    
    async def add_task(self, task_type: str, endpoint: str, payload: Dict, 
                      priority: Priority = Priority.MANUAL_QUERY, vessel_imo: str = None) -> str:
        """
        Add task to queue
        Returns task_id for tracking
        """
        if self.queue.empty() and not self.is_processing:
            return await self._execute_immediately(task_type, endpoint, payload, vessel_imo)

        task_id = str(uuid.uuid4())
        
        task = QueueTask(
            id=task_id,
            priority=priority,
            task_type=task_type,
            endpoint=endpoint,
            payload=payload,
            vessel_imo=vessel_imo
        )
        
        # Add to queue (priority queue sorts by priority value - lower is higher priority)
        await self.queue.put((priority.value, task))
        
        # Track task
        self.active_tasks[task_id] = task
        
        logger.info(f"Task {task_id} ({task_type}) added to queue with priority {priority.name}")
        
        # Start processing if not already running
        if not self.is_processing:
            asyncio.create_task(self._process_queue())
        
        self.stats['queued_tasks'] += 1
        return task_id
    
    async def _execute_immediately(self, task_type: str, endpoint: str, payload: Dict, vessel_imo: str = None) -> str:
        task_id = str(uuid.uuid4())
        logger.info(f"Executing immediately: {task_id} ({task_type})")
        
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._make_gpu_request, endpoint, payload
        )
        
        # Store result for retrieval
        task = QueueTask(
            id=task_id, priority=Priority.CHAT, task_type=task_type,
            endpoint=endpoint, payload=payload, vessel_imo=vessel_imo,
            result=result, completed_at=datetime.utcnow()
        )
        self.active_tasks[task_id] = task
        self.stats['total_processed'] += 1
        
        return task_id    

    async def _process_queue(self):
        """
        Main queue processing loop
        Processes one task at a time to avoid GPU conflicts
        """
        if self.is_processing:
            return
            
        self.is_processing = True
        logger.info("Starting queue processing")
        
        try:
            while not self.queue.empty():
                try:
                    # Get next task
                    priority_value, task = await self.queue.get()
                    
                    self.current_task = task
                    task.started_at = datetime.utcnow()
                    
                    logger.info(f"Processing queued task {task.id}: {task.task_type}")
                    
                    # Calculate wait time
                    wait_time = (task.started_at - task.created_at).total_seconds()
                    
                    # Make GPU service call
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, self._make_gpu_request, task.endpoint, task.payload
                    )
                    
                    # Update task
                    task.completed_at = datetime.utcnow()
                    task.result = result
                    
                    # Update stats
                    self.stats['total_processed'] += 1
                    current_avg = self.stats['average_wait_time']
                    total_processed = self.stats['total_processed']
                    self.stats['average_wait_time'] = ((current_avg * (total_processed - 1)) + wait_time) / total_processed
                    
                    logger.info(f"Queued task {task.id} completed (waited {wait_time:.2f}s)")
                    
                    # Mark task as done
                    self.queue.task_done()
                    
                except Exception as e:
                    if 'task' in locals():
                        task.error = str(e)
                        task.completed_at = datetime.utcnow()
                        self.stats['failed_tasks'] += 1
                        
                    logger.error(f"Error processing queued task: {e}")
                    
                finally:
                    self.current_task = None
                    
        except Exception as e:
            logger.error(f"Queue processing error: {e}")
            
        finally:
            self.is_processing = False
            logger.info("Queue processing stopped")
    
    def get_task_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a specific task"""
        if task_id not in self.active_tasks:
            return None
            
        task = self.active_tasks[task_id]
        
        status = {
            'task_id': task_id,
            'task_type': task.task_type,
            'priority': task.priority.name,
            'vessel_imo': task.vessel_imo,
            'created_at': task.created_at.isoformat(),
            'started_at': task.started_at.isoformat() if task.started_at else None,
            'completed_at': task.completed_at.isoformat() if task.completed_at else None,
            'status': 'pending' if not task.started_at else 
                     'processing' if not task.completed_at else 
                     'completed' if not task.error else 'failed',
            'result': task.result,
            'error': task.error
        }
        
        return status
    
    def get_task_result(self, task_id: str, timeout: float = 60.0) -> Optional[Dict]:
        """
        Wait for task completion and return result
        Blocks until task is complete or timeout
        """
        if task_id not in self.active_tasks:
            return None
        
        task = self.active_tasks[task_id]
        start_time = time.time()
        
        # Poll for completion
        while time.time() - start_time < timeout:
            if task.completed_at:
                # Clean up completed task
                del self.active_tasks[task_id]
                
                if task.error:
                    return {'error': task.error, 'status': 'failed'}
                else:
                    return task.result
                    
            time.sleep(0.1)  # Short poll interval
        
        # Timeout
        return {'error': 'Task timeout', 'status': 'timeout'}
    
    async def get_task_result_async(self, task_id: str, timeout: float = 60.0) -> Optional[Dict]:
        """
        Async version of get_task_result
        """
        if task_id not in self.active_tasks:
            return None
        
        task = self.active_tasks[task_id]
        start_time = time.time()
        
        # Async poll for completion
        while time.time() - start_time < timeout:
            if task.completed_at:
                # Clean up completed task
                del self.active_tasks[task_id]
                
                if task.error:
                    return {'error': task.error, 'status': 'failed'}
                else:
                    return task.result
                    
            await asyncio.sleep(0.1)  # Async sleep
        
        # Timeout
        return {'error': 'Task timeout', 'status': 'timeout'}
    
    def get_queue_status(self) -> Dict:
        """Get current queue status"""
        return {
            'queue_size': self.queue.qsize(),
            'is_processing': self.is_processing,
            'current_task': {
                'id': self.current_task.id,
                'type': self.current_task.task_type,
                'vessel_imo': self.current_task.vessel_imo,
                'started_at': self.current_task.started_at.isoformat()
            } if self.current_task else None,
            'active_tasks': len(self.active_tasks),
            'statistics': self.stats.copy()
        }
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Clean up old completed tasks to prevent memory leaks"""
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        
        to_remove = []
        for task_id, task in self.active_tasks.items():
            if task.completed_at and task.completed_at < cutoff_time:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.active_tasks[task_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old completed tasks")


# Global queue manager instance
queue_manager = None

def initialize_queue_manager(gpu_service_url: str = "http://localhost:5005") -> GPUQueueManager:
    """
    Initialize global queue manager
    Call this at app startup
    """
    global queue_manager
    queue_manager = GPUQueueManager(gpu_service_url)
    return queue_manager

def get_queue_manager() -> GPUQueueManager:
    """
    Get global queue manager instance
    """
    global queue_manager
    if queue_manager is None:
        raise RuntimeError("Queue manager not initialized. Call initialize_queue_manager() first.")
    return queue_manager


# Helper functions for common operations
async def process_chat_immediately(messages: list, response_type: str = "chat", vessel_imo: str = None) -> Dict:
    """Process chat request immediately (highest priority)"""
    manager = get_queue_manager()
    return await manager.process_immediately(
        task_type="chat",
        endpoint="/gpu/llm/generate",
        payload={"messages": messages, "response_type": response_type},
        vessel_imo=vessel_imo
    )

async def process_audio_transcription(audio_file, vessel_imo: str = None) -> str:
    """Add audio transcription to high priority queue"""
    manager = get_queue_manager()
    task_id = await manager.add_task(
        task_type="audio_transcription",
        endpoint="/gpu/stt/transcribe",
        payload={"files": {"audio": audio_file}},
        priority=Priority.AUDIO,
        vessel_imo=vessel_imo
    )
    
    # Wait for result
    result = await manager.get_task_result_async(task_id)
    return result

async def process_face_comparison(image1, image2, vessel_imo: str = None) -> str:
    """Add face comparison to high priority queue"""
    manager = get_queue_manager()
    task_id = await manager.add_task(
        task_type="face_comparison",
        endpoint="/gpu/face/compare",
        payload={"files": {"image": image1, "profilePicture": image2}},
        priority=Priority.FACE_RECOGNITION,
        vessel_imo=vessel_imo
    )
    
    # Wait for result
    result = await manager.get_task_result_async(task_id)
    return result