"""
Storage Service for File Management
"""
import logging
import shutil
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class StorageService:
    """Manage file storage for images and data"""
    
    def __init__(self, base_path: str = "./storage"):
        """
        Initialize storage service
        
        Args:
            base_path: Base storage directory
        """
        self.base_path = Path(base_path)
        
        # Create directories
        self.images_path = self.base_path / "images"
        self.plates_path = self.base_path / "plates"
        self.temp_path = self.base_path / "temp"
        self.dataset_path = self.base_path / "dataset"
        
        for path in [self.images_path, self.plates_path, self.temp_path, self.dataset_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Storage initialized: {self.base_path}")
    
    async def save_image(
        self,
        image_data: bytes,
        filename: str,
        subfolder: str = "images"
    ) -> Path:
        """
        Save image to storage
        
        Args:
            image_data: Image bytes
            filename: Filename
            subfolder: Subfolder (images, plates, temp)
        
        Returns:
            Path to saved file
        """
        folder = self.base_path / subfolder
        folder.mkdir(exist_ok=True)
        
        filepath = folder / filename
        
        try:
            await asyncio.to_thread(filepath.write_bytes, image_data)
            logger.debug(f"Image saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise
    
    async def delete_file(self, filepath: str) -> bool:
        """Delete file"""
        try:
            path = Path(filepath)
            if path.exists():
                await asyncio.to_thread(path.unlink)
                logger.debug(f"File deleted: {filepath}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False
    
    async def cleanup_old_files(
        self,
        days: int = 30,
        subfolder: str = "images"
    ) -> int:
        """
        Clean up files older than specified days
        
        Args:
            days: Delete files older than this
            subfolder: Folder to clean
        
        Returns:
            Number of files deleted
        """
        folder = self.base_path / subfolder
        
        if not folder.exists():
            return 0
        
        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        try:
            for file in folder.iterdir():
                if file.is_file():
                    file_mtime = datetime.fromtimestamp(file.stat().st_mtime)
                    
                    if file_mtime < cutoff_time:
                        await self.delete_file(str(file))
                        deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} files older than {days} days")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return deleted_count
    
    def get_storage_stats(self) -> dict:
        """Get storage usage statistics"""
        def get_dir_size(path: Path) -> int:
            total = 0
            for item in path.rglob('*'):
                if item.is_file():
                    total += item.stat().st_size
            return total
        
        stats = {
            "total_size_bytes": get_dir_size(self.base_path),
            "images_size_bytes": get_dir_size(self.images_path),
            "plates_size_bytes": get_dir_size(self.plates_path),
            "dataset_size_bytes": get_dir_size(self.dataset_path),
            "images_count": len(list(self.images_path.glob('*'))),
            "plates_count": len(list(self.plates_path.glob('*')))
        }
        
        # Convert to MB
        stats["total_size_mb"] = stats["total_size_bytes"] / (1024 * 1024)
        stats["images_size_mb"] = stats["images_size_bytes"] / (1024 * 1024)
        stats["plates_size_mb"] = stats["plates_size_bytes"] / (1024 * 1024)
        
        return stats
    
    async def move_file(self, src: str, dst: str) -> bool:
        """Move file from src to dst"""
        try:
            src_path = Path(src)
            dst_path = Path(dst)
            
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            await asyncio.to_thread(shutil.move, str(src_path), str(dst_path))
            logger.debug(f"File moved: {src} -> {dst}")
            return True
        except Exception as e:
            logger.error(f"Failed to move file: {e}")
            return False
    
    async def copy_file(self, src: str, dst: str) -> bool:
        """Copy file from src to dst"""
        try:
            src_path = Path(src)
            dst_path = Path(dst)
            
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            
            await asyncio.to_thread(shutil.copy2, str(src_path), str(dst_path))
            logger.debug(f"File copied: {src} -> {dst}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy file: {e}")
            return False
    
    def file_exists(self, filepath: str) -> bool:
        """Check if file exists"""
        return Path(filepath).exists()
    
    def get_file_info(self, filepath: str) -> Optional[dict]:
        """Get file information"""
        path = Path(filepath)
        
        if not path.exists():
            return None
        
        stat = path.stat()
        
        return {
            "path": str(path),
            "name": path.name,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "created_at": datetime.fromtimestamp(stat.st_ctime),
            "modified_at": datetime.fromtimestamp(stat.st_mtime)
        }


# Global instance
_storage_service: Optional[StorageService] = None


def get_storage() -> StorageService:
    """Get storage service instance"""
    global _storage_service
    
    if _storage_service is None:
        raise RuntimeError("Storage service not initialized")
    
    return _storage_service


def init_storage(base_path: str = "./storage") -> StorageService:
    """Initialize storage service"""
    global _storage_service
    
    _storage_service = StorageService(base_path)
    return _storage_service