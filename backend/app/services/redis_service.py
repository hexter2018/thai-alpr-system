"""
Redis Service for Caching, Deduplication, and Queue Management
"""
import logging
import json
from typing import Optional, Any, List, Dict
from datetime import timedelta
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisService:
    """
    Redis service for ALPR system
    Handles deduplication, caching, and real-time data
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: Optional[str] = None,
        db: int = 0,
        decode_responses: bool = True
    ):
        """
        Initialize Redis service
        
        Args:
            host: Redis host
            port: Redis port
            password: Redis password
            db: Database number
            decode_responses: Auto-decode responses to strings
        """
        self.host = host
        self.port = port
        self.password = password
        self.db = db
        self.decode_responses = decode_responses
        
        self.client: Optional[redis.Redis] = None
        self.pool: Optional[redis.ConnectionPool] = None
    
    async def connect(self):
        """Establish Redis connection"""
        try:
            self.pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                password=self.password,
                db=self.db,
                decode_responses=self.decode_responses,
                max_connections=50
            )
            
            self.client = redis.Redis(connection_pool=self.pool)
            
            # Test connection
            await self.client.ping()
            
            logger.info(f"Redis connected: {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connection"""
        try:
            if self.client:
                await self.client.close()
            if self.pool:
                await self.pool.disconnect()
            logger.info("Redis disconnected")
        except Exception as e:
            logger.error(f"Redis disconnect error: {e}")
    
    async def is_connected(self) -> bool:
        """Check if Redis is connected"""
        try:
            if self.client:
                await self.client.ping()
                return True
            return False
        except:
            return False
    
    # ==================== Deduplication Methods ====================
    
    async def check_processed(
        self,
        tracking_id: str,
        camera_id: str
    ) -> bool:
        """
        Check if vehicle was recently processed (deduplication)
        
        Args:
            tracking_id: Vehicle tracking ID
            camera_id: Camera identifier
        
        Returns:
            True if already processed
        """
        key = f"processed:{camera_id}:{tracking_id}"
        
        try:
            exists = await self.client.exists(key)
            return bool(exists)
        except Exception as e:
            logger.error(f"Dedup check failed: {e}")
            return False
    
    async def mark_processed(
        self,
        tracking_id: str,
        camera_id: str,
        ttl: int = 60,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Mark vehicle as processed with TTL
        
        Args:
            tracking_id: Vehicle tracking ID
            camera_id: Camera identifier
            ttl: Time to live in seconds
            metadata: Optional metadata to store
        
        Returns:
            Success status
        """
        key = f"processed:{camera_id}:{tracking_id}"
        
        try:
            value = json.dumps(metadata) if metadata else "1"
            await self.client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Failed to mark processed: {e}")
            return False
    
    async def get_processed_info(
        self,
        tracking_id: str,
        camera_id: str
    ) -> Optional[Dict]:
        """Get metadata of processed vehicle"""
        key = f"processed:{camera_id}:{tracking_id}"
        
        try:
            value = await self.client.get(key)
            if value and value != "1":
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Failed to get processed info: {e}")
            return None
    
    # ==================== Caching Methods ====================
    
    async def set_cache(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set cache value
        
        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time to live in seconds
        
        Returns:
            Success status
        """
        try:
            serialized = json.dumps(value)
            if ttl:
                await self.client.setex(key, ttl, serialized)
            else:
                await self.client.set(key, serialized)
            return True
        except Exception as e:
            logger.error(f"Cache set failed for {key}: {e}")
            return False
    
    async def get_cache(self, key: str) -> Optional[Any]:
        """Get cached value"""
        try:
            value = await self.client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get failed for {key}: {e}")
            return None
    
    async def delete_cache(self, key: str) -> bool:
        """Delete cache key"""
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete failed for {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        try:
            return bool(await self.client.exists(key))
        except Exception as e:
            logger.error(f"Exists check failed for {key}: {e}")
            return False
    
    # ==================== Statistics & Counters ====================
    
    async def increment_counter(
        self,
        key: str,
        amount: int = 1
    ) -> int:
        """Increment counter"""
        try:
            return await self.client.incrby(key, amount)
        except Exception as e:
            logger.error(f"Counter increment failed for {key}: {e}")
            return 0
    
    async def get_counter(self, key: str) -> int:
        """Get counter value"""
        try:
            value = await self.client.get(key)
            return int(value) if value else 0
        except Exception as e:
            logger.error(f"Get counter failed for {key}: {e}")
            return 0
    
    async def reset_counter(self, key: str) -> bool:
        """Reset counter to 0"""
        try:
            await self.client.set(key, 0)
            return True
        except Exception as e:
            logger.error(f"Reset counter failed for {key}: {e}")
            return False
    
    # ==================== Real-time Data Methods ====================
    
    async def publish_detection(
        self,
        camera_id: str,
        detection_data: Dict
    ) -> int:
        """
        Publish detection event to Redis pub/sub
        
        Args:
            camera_id: Camera identifier
            detection_data: Detection data
        
        Returns:
            Number of subscribers that received the message
        """
        channel = f"detections:{camera_id}"
        
        try:
            message = json.dumps(detection_data)
            return await self.client.publish(channel, message)
        except Exception as e:
            logger.error(f"Publish failed for {channel}: {e}")
            return 0
    
    async def subscribe_detections(self, camera_id: str):
        """
        Subscribe to detection events
        Returns async generator
        """
        channel = f"detections:{camera_id}"
        
        try:
            pubsub = self.client.pubsub()
            await pubsub.subscribe(channel)
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"])
        except Exception as e:
            logger.error(f"Subscribe failed for {channel}: {e}")
    
    # ==================== List Operations (Queue) ====================
    
    async def push_to_queue(
        self,
        queue_name: str,
        item: Any,
        max_size: Optional[int] = None
    ) -> bool:
        """
        Push item to queue (list)
        
        Args:
            queue_name: Queue name
            item: Item to push
            max_size: Maximum queue size (trim if exceeded)
        
        Returns:
            Success status
        """
        try:
            serialized = json.dumps(item)
            await self.client.lpush(queue_name, serialized)
            
            if max_size:
                await self.client.ltrim(queue_name, 0, max_size - 1)
            
            return True
        except Exception as e:
            logger.error(f"Queue push failed for {queue_name}: {e}")
            return False
    
    async def pop_from_queue(self, queue_name: str) -> Optional[Any]:
        """Pop item from queue"""
        try:
            value = await self.client.rpop(queue_name)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Queue pop failed for {queue_name}: {e}")
            return None
    
    async def get_queue_length(self, queue_name: str) -> int:
        """Get queue length"""
        try:
            return await self.client.llen(queue_name)
        except Exception as e:
            logger.error(f"Queue length failed for {queue_name}: {e}")
            return 0
    
    # ==================== Hash Operations ====================
    
    async def set_hash(
        self,
        hash_key: str,
        field: str,
        value: Any
    ) -> bool:
        """Set hash field"""
        try:
            serialized = json.dumps(value)
            await self.client.hset(hash_key, field, serialized)
            return True
        except Exception as e:
            logger.error(f"Hash set failed for {hash_key}:{field}: {e}")
            return False
    
    async def get_hash(
        self,
        hash_key: str,
        field: str
    ) -> Optional[Any]:
        """Get hash field"""
        try:
            value = await self.client.hget(hash_key, field)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Hash get failed for {hash_key}:{field}: {e}")
            return None
    
    async def get_all_hash(self, hash_key: str) -> Dict:
        """Get all fields in hash"""
        try:
            data = await self.client.hgetall(hash_key)
            return {k: json.loads(v) for k, v in data.items()}
        except Exception as e:
            logger.error(f"Hash getall failed for {hash_key}: {e}")
            return {}
    
    # ==================== Camera State Management ====================
    
    async def set_camera_active(
        self,
        camera_id: str,
        is_active: bool
    ) -> bool:
        """Set camera active state"""
        key = f"camera:state:{camera_id}"
        return await self.set_cache(key, {"active": is_active, "camera_id": camera_id})
    
    async def is_camera_active(self, camera_id: str) -> bool:
        """Check if camera is active"""
        key = f"camera:state:{camera_id}"
        state = await self.get_cache(key)
        return state.get("active", False) if state else False
    
    async def get_active_cameras(self) -> List[str]:
        """Get list of active camera IDs"""
        try:
            pattern = "camera:state:*"
            camera_ids = []
            
            async for key in self.client.scan_iter(match=pattern):
                state = await self.get_cache(key)
                if state and state.get("active"):
                    camera_ids.append(state.get("camera_id"))
            
            return camera_ids
        except Exception as e:
            logger.error(f"Get active cameras failed: {e}")
            return []
    
    # ==================== Utility Methods ====================
    
    async def get_info(self) -> Dict:
        """Get Redis server info"""
        try:
            info = await self.client.info()
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "uptime_in_days": info.get("uptime_in_days")
            }
        except Exception as e:
            logger.error(f"Get info failed: {e}")
            return {}
    
    async def flush_db(self) -> bool:
        """Flush current database (use with caution!)"""
        try:
            await self.client.flushdb()
            logger.warning("Redis database flushed!")
            return True
        except Exception as e:
            logger.error(f"Flush DB failed: {e}")
            return False
    
    async def clear_processed(self, camera_id: Optional[str] = None) -> int:
        """
        Clear processed vehicle records
        
        Args:
            camera_id: Optional camera ID to clear (None = all cameras)
        
        Returns:
            Number of keys deleted
        """
        try:
            pattern = f"processed:{camera_id}:*" if camera_id else "processed:*"
            deleted = 0
            
            async for key in self.client.scan_iter(match=pattern):
                await self.client.delete(key)
                deleted += 1
            
            logger.info(f"Cleared {deleted} processed records")
            return deleted
        except Exception as e:
            logger.error(f"Clear processed failed: {e}")
            return 0


# Global Redis instance
_redis_service: Optional[RedisService] = None


async def get_redis() -> RedisService:
    """Get Redis service instance (singleton)"""
    global _redis_service
    
    if _redis_service is None:
        raise RuntimeError("Redis service not initialized. Call init_redis() first.")
    
    return _redis_service


async def init_redis(
    host: str = "localhost",
    port: int = 6379,
    password: Optional[str] = None,
    db: int = 0
) -> RedisService:
    """
    Initialize Redis service
    
    Args:
        host: Redis host
        port: Redis port
        password: Redis password
        db: Database number
    
    Returns:
        RedisService instance
    """
    global _redis_service
    
    _redis_service = RedisService(
        host=host,
        port=port,
        password=password,
        db=db
    )
    
    await _redis_service.connect()
    
    return _redis_service


async def close_redis():
    """Close Redis connection"""
    global _redis_service
    
    if _redis_service:
        await _redis_service.disconnect()
        _redis_service = None