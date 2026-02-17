"""
Active Learning Service
Manages training dataset for continuous model improvement
"""
import logging
import shutil
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from pathlib import Path
import json
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ActiveLearningService:
    """
    Active learning service for ALPR system
    Collects and manages training data from manual corrections
    """
    
    def __init__(
        self,
        dataset_path: str = "./storage/dataset",
        min_samples_for_training: int = 100
    ):
        """
        Initialize active learning service
        
        Args:
            dataset_path: Base path for dataset storage
            min_samples_for_training: Minimum samples before retraining
        """
        self.dataset_path = Path(dataset_path)
        self.min_samples = min_samples_for_training
        
        # Create directory structure
        self.train_images = self.dataset_path / "train" / "images"
        self.train_labels = self.dataset_path / "train" / "labels"
        self.val_images = self.dataset_path / "val" / "images"
        self.val_labels = self.dataset_path / "val" / "labels"
        
        for path in [self.train_images, self.train_labels, self.val_images, self.val_labels]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.dataset_path / "metadata.json"
        self.metadata = self._load_metadata()
        
        logger.info(f"Active learning initialized: {self.dataset_path}")
    
    def _load_metadata(self) -> Dict:
        """Load metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
        
        return {
            "samples_count": 0,
            "last_updated": None,
            "classes": {},
            "samples": []
        }
    
    def _save_metadata(self):
        """Save metadata to file"""
        try:
            self.metadata["last_updated"] = datetime.now().isoformat()
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    async def add_correction_sample(
        self,
        plate_image_path: str,
        original_text: str,
        corrected_text: str,
        corrected_province: str,
        confidence: float,
        verified_by: str
    ) -> bool:
        """
        Add manually corrected sample to training dataset
        
        Args:
            plate_image_path: Path to original plate crop
            original_text: OCR detected text
            corrected_text: Human-corrected text
            corrected_province: Human-corrected province
            confidence: Original OCR confidence
            verified_by: Operator ID
        
        Returns:
            Success status
        """
        try:
            src_path = Path(plate_image_path)
            
            if not src_path.exists():
                logger.error(f"Source image not found: {src_path}")
                return False
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_id = f"{timestamp}_{self.metadata['samples_count']:04d}"
            
            # Copy image to training set
            dst_image = self.train_images / f"{sample_id}.jpg"
            shutil.copy2(src_path, dst_image)
            
            # Create label file
            label_data = {
                "plate_text": corrected_text,
                "province": corrected_province,
                "original_text": original_text,
                "confidence": confidence,
                "verified_by": verified_by,
                "timestamp": timestamp
            }
            
            dst_label = self.train_labels / f"{sample_id}.json"
            with open(dst_label, 'w', encoding='utf-8') as f:
                json.dump(label_data, f, ensure_ascii=False, indent=2)
            
            # Update metadata
            self.metadata["samples_count"] += 1
            self.metadata["samples"].append({
                "id": sample_id,
                "plate_text": corrected_text,
                "province": corrected_province,
                "timestamp": timestamp
            })
            
            # Track class distribution
            if corrected_text not in self.metadata["classes"]:
                self.metadata["classes"][corrected_text] = 0
            self.metadata["classes"][corrected_text] += 1
            
            self._save_metadata()
            
            logger.info(
                f"Added training sample: {sample_id} - {corrected_text} "
                f"(Total: {self.metadata['samples_count']})"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add correction sample: {e}")
            return False
    
    async def add_low_confidence_sample(
        self,
        plate_image_path: str,
        detected_text: str,
        confidence: float,
        needs_review: bool = True
    ) -> bool:
        """
        Add low confidence sample for future review
        
        Args:
            plate_image_path: Path to plate image
            detected_text: Detected text
            confidence: Detection confidence
            needs_review: Mark for manual review
        
        Returns:
            Success status
        """
        try:
            # Store in separate folder for review
            review_path = self.dataset_path / "review"
            review_path.mkdir(exist_ok=True)
            
            src_path = Path(plate_image_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sample_id = f"review_{timestamp}"
            
            dst_image = review_path / f"{sample_id}.jpg"
            shutil.copy2(src_path, dst_image)
            
            # Save metadata
            meta_file = review_path / f"{sample_id}.json"
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "detected_text": detected_text,
                    "confidence": confidence,
                    "needs_review": needs_review,
                    "timestamp": timestamp
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Added low confidence sample for review: {sample_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add low confidence sample: {e}")
            return False
    
    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
        train_count = len(list(self.train_images.glob("*.jpg")))
        val_count = len(list(self.val_images.glob("*.jpg")))
        review_count = len(list((self.dataset_path / "review").glob("*.jpg"))) if (self.dataset_path / "review").exists() else 0
        
        return {
            "total_samples": self.metadata["samples_count"],
            "train_samples": train_count,
            "val_samples": val_count,
            "review_samples": review_count,
            "unique_plates": len(self.metadata["classes"]),
            "ready_for_training": train_count >= self.min_samples,
            "min_samples_required": self.min_samples,
            "last_updated": self.metadata.get("last_updated"),
            "class_distribution": self.metadata["classes"]
        }
    
    async def split_train_val(self, val_ratio: float = 0.2) -> Tuple[int, int]:
        """
        Split training data into train/val sets
        
        Args:
            val_ratio: Validation set ratio (0.0 - 1.0)
        
        Returns:
            (train_count, val_count)
        """
        try:
            # Get all training images
            train_images = list(self.train_images.glob("*.jpg"))
            
            if not train_images:
                logger.warning("No training images found")
                return (0, 0)
            
            # Shuffle
            import random
            random.shuffle(train_images)
            
            # Split
            val_count = int(len(train_images) * val_ratio)
            val_samples = train_images[:val_count]
            
            # Move to validation set
            moved = 0
            for img_path in val_samples:
                # Move image
                dst_img = self.val_images / img_path.name
                shutil.move(str(img_path), str(dst_img))
                
                # Move label
                label_path = self.train_labels / f"{img_path.stem}.json"
                if label_path.exists():
                    dst_label = self.val_labels / label_path.name
                    shutil.move(str(label_path), str(dst_label))
                
                moved += 1
            
            train_remaining = len(train_images) - moved
            
            logger.info(f"Split dataset: {train_remaining} train, {moved} val")
            return (train_remaining, moved)
            
        except Exception as e:
            logger.error(f"Failed to split dataset: {e}")
            return (0, 0)
    
    async def export_for_training(
        self,
        output_path: Optional[str] = None,
        format: str = "yolo"
    ) -> Optional[str]:
        """
        Export dataset in specific format for training
        
        Args:
            output_path: Export directory
            format: Dataset format (yolo, coco, etc.)
        
        Returns:
            Path to exported dataset
        """
        try:
            if output_path is None:
                output_path = self.dataset_path / "export" / datetime.now().strftime("%Y%m%d_%H%M%S")
            
            export_dir = Path(output_path)
            export_dir.mkdir(parents=True, exist_ok=True)
            
            if format == "yolo":
                # Create YOLO format structure
                (export_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
                (export_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
                (export_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
                (export_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
                
                # Copy training images
                for img in self.train_images.glob("*.jpg"):
                    shutil.copy2(img, export_dir / "images" / "train" / img.name)
                
                # Copy validation images
                for img in self.val_images.glob("*.jpg"):
                    shutil.copy2(img, export_dir / "images" / "val" / img.name)
                
                # Create data.yaml
                yaml_content = f"""
train: {export_dir}/images/train
val: {export_dir}/images/val

nc: 1
names: ['license_plate']
"""
                with open(export_dir / "data.yaml", 'w') as f:
                    f.write(yaml_content)
                
                logger.info(f"Exported dataset to: {export_dir}")
                return str(export_dir)
            
            else:
                logger.error(f"Unsupported format: {format}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to export dataset: {e}")
            return None
    
    def get_sample_preview(
        self,
        sample_id: str
    ) -> Optional[Dict]:
        """
        Get preview of a training sample
        
        Args:
            sample_id: Sample identifier
        
        Returns:
            Sample data with image path
        """
        try:
            img_path = self.train_images / f"{sample_id}.jpg"
            label_path = self.train_labels / f"{sample_id}.json"
            
            if not img_path.exists():
                return None
            
            label_data = {}
            if label_path.exists():
                with open(label_path, 'r', encoding='utf-8') as f:
                    label_data = json.load(f)
            
            return {
                "sample_id": sample_id,
                "image_path": str(img_path),
                "label_data": label_data
            }
            
        except Exception as e:
            logger.error(f"Failed to get sample preview: {e}")
            return None
    
    async def clear_dataset(self, confirm: bool = False) -> bool:
        """
        Clear entire dataset (use with caution!)
        
        Args:
            confirm: Must be True to proceed
        
        Returns:
            Success status
        """
        if not confirm:
            logger.warning("Clear dataset requires confirmation")
            return False
        
        try:
            # Clear directories
            for path in [self.train_images, self.train_labels, self.val_images, self.val_labels]:
                for file in path.glob("*"):
                    file.unlink()
            
            # Reset metadata
            self.metadata = {
                "samples_count": 0,
                "last_updated": None,
                "classes": {},
                "samples": []
            }
            self._save_metadata()
            
            logger.warning("Dataset cleared!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear dataset: {e}")
            return False


# Global instance
_active_learning_service: Optional[ActiveLearningService] = None


def get_active_learning() -> ActiveLearningService:
    """Get active learning service instance"""
    global _active_learning_service
    
    if _active_learning_service is None:
        raise RuntimeError("Active learning service not initialized")
    
    return _active_learning_service


def init_active_learning(
    dataset_path: str = "./storage/dataset",
    min_samples: int = 100
) -> ActiveLearningService:
    """Initialize active learning service"""
    global _active_learning_service
    
    _active_learning_service = ActiveLearningService(
        dataset_path=dataset_path,
        min_samples_for_training=min_samples
    )
    
    return _active_learning_service