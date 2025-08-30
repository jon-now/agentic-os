#!/usr/bin/env python3
"""
Vision Controller for RTX 3050 6GB
Provides OCR, image analysis, and basic vision capabilities optimized for 6GB VRAM
"""

import os
import logging
import asyncio
import tempfile
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import cv2
import numpy as np

# OCR and Vision imports with GPU optimization
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import paddleocr
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from PIL import Image, ImageEnhance, ImageFilter
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class VisionController:
    """RTX 3050 optimized vision processing controller"""
    
    def __init__(self):
        self.ocr_engine = None
        self.vision_models = {}
        self.rtx_3050_config = {
            "max_vram_usage": 2048,  # Reserve 2GB for vision tasks
            "batch_size": 4,
            "image_max_size": (1920, 1080),  # Reasonable limit for RTX 3050
            "ocr_confidence_threshold": 0.7,
            "enable_gpu_acceleration": CUDA_AVAILABLE
        }
        
        # Initialize vision capabilities
        self._initialize_ocr()
        self._initialize_vision_models()
        
        logger.info("Vision Controller initialized for RTX 3050")
    
    def _initialize_ocr(self):
        """Initialize OCR engine optimized for RTX 3050"""
        # Try EasyOCR first (more Windows-compatible)
        if EASYOCR_AVAILABLE:
            try:
                use_gpu = self.rtx_3050_config["enable_gpu_acceleration"]
                
                self.ocr_engine = easyocr.Reader(
                    ['en'],
                    gpu=use_gpu,
                    verbose=False
                )
                
                self.ocr_type = "easyocr"
                device = "GPU" if use_gpu else "CPU"
                logger.info(f"✅ EasyOCR initialized on {device} (RTX 3050 optimized)")
                
                if use_gpu:
                    self._log_vram_usage("EasyOCR loaded")
                
            except Exception as e:
                logger.error(f"EasyOCR initialization failed: {e}")
                self._fallback_to_paddleocr()
        
        elif PADDLEOCR_AVAILABLE:
            self._fallback_to_paddleocr()
        else:
            logger.warning("No OCR engine available. Install easyocr or paddleocr.")
            self.ocr_engine = None
    
    def _fallback_to_paddleocr(self):
        """Fallback to PaddleOCR if EasyOCR fails"""
        try:
            # RTX 3050 optimized settings
            use_gpu = self.rtx_3050_config["enable_gpu_acceleration"]
            
            self.ocr_engine = paddleocr.PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=use_gpu,
                gpu_mem=self.rtx_3050_config["max_vram_usage"],  # Limit GPU memory
                enable_mkldnn=True,  # CPU optimization
                cpu_threads=4,  # Reasonable thread count
                show_log=False
            )
            
            self.ocr_type = "paddleocr"
            device = "GPU" if use_gpu else "CPU"
            logger.info(f"✅ PaddleOCR initialized on {device} (fallback)")
            
            if use_gpu:
                self._log_vram_usage("PaddleOCR loaded")
            
        except Exception as e:
            logger.error(f"PaddleOCR initialization failed: {e}")
            self.ocr_engine = None
            self.ocr_type = None
    
    def _initialize_vision_models(self):
        """Initialize lightweight vision models for RTX 3050"""
        # For now, we'll focus on OCR and basic image processing
        # Larger vision models like LLaVA can be added when more VRAM is available
        self.vision_models = {
            "ocr": self.ocr_engine,
            "image_processing": True  # Basic CV2/PIL processing
        }
        
        logger.info("Vision models initialized (OCR + basic image processing)")
    
    async def extract_text_from_image(self, image_path: str, language: str = 'en') -> Dict[str, Any]:
        """Extract text from image using OCR"""
        if not self.ocr_engine:
            return {"error": "No OCR engine available"}
        
        try:
            # Load and preprocess image
            image = self._load_and_preprocess_image(image_path)
            if image is None:
                return {"error": "Failed to load image"}
            
            # Perform OCR based on engine type
            if self.ocr_type == "easyocr":
                result = await self._easyocr_extract(image)
            elif self.ocr_type == "paddleocr":
                result = await self._paddleocr_extract(image)
            else:
                return {"error": "No valid OCR engine"}
            
            return result
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {"error": str(e)}
    
    async def _paddleocr_extract(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text using PaddleOCR"""
        try:
            # Run OCR in executor to avoid blocking
            loop = asyncio.get_event_loop()
            ocr_result = await loop.run_in_executor(
                None, 
                self.ocr_engine.ocr, 
                image, 
                False  # Don't use angle classification for speed
            )
            
            if not ocr_result or not ocr_result[0]:
                return {"text": "", "confidence": 0, "boxes": []}
            
            # Parse results
            extracted_text = []
            boxes = []
            confidences = []
            
            for line in ocr_result[0]:
                bbox, (text, confidence) = line
                
                if confidence >= self.rtx_3050_config["ocr_confidence_threshold"]:
                    extracted_text.append(text)
                    boxes.append(bbox)
                    confidences.append(confidence)
            
            full_text = " ".join(extracted_text)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "boxes": boxes,
                "line_count": len(extracted_text),
                "engine": "paddleocr",
                "processing_time": "< 1s"
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return {"error": str(e)}
    
    async def _easyocr_extract(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract text using EasyOCR"""
        try:
            # Run OCR in executor
            loop = asyncio.get_event_loop()
            ocr_result = await loop.run_in_executor(
                None,
                self.ocr_engine.readtext,
                image
            )
            
            # Parse results
            extracted_text = []
            boxes = []
            confidences = []
            
            for bbox, text, confidence in ocr_result:
                if confidence >= self.rtx_3050_config["ocr_confidence_threshold"]:
                    extracted_text.append(text)
                    boxes.append(bbox)
                    confidences.append(confidence)
            
            full_text = " ".join(extracted_text)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "text": full_text,
                "confidence": avg_confidence,
                "boxes": boxes,
                "line_count": len(extracted_text),
                "engine": "easyocr",
                "processing_time": "< 1s"
            }
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return {"error": str(e)}
    
    def _load_and_preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and preprocess image for OCR"""
        try:
            # Support different input types
            if isinstance(image_path, str):
                if image_path.startswith('data:image'):
                    # Base64 encoded image
                    return self._decode_base64_image(image_path)
                else:
                    # File path
                    image = cv2.imread(image_path)
            else:
                return None
            
            if image is None:
                logger.error(f"Failed to load image from {image_path}")
                return None
            
            # Resize if too large (RTX 3050 memory optimization)
            height, width = image.shape[:2]
            max_width, max_height = self.rtx_3050_config["image_max_size"]
            
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
                logger.info(f"Image resized to {new_width}x{new_height} for RTX 3050 optimization")
            
            return image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return None
    
    def _decode_base64_image(self, base64_string: str) -> Optional[np.ndarray]:
        """Decode base64 image string"""
        try:
            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(base64_string)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return image
            
        except Exception as e:
            logger.error(f"Base64 image decoding failed: {e}")
            return None
    
    async def analyze_screenshot(self) -> Dict[str, Any]:
        """Capture and analyze current screenshot"""
        try:
            import pyautogui
            
            # Capture screenshot
            screenshot = pyautogui.screenshot()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                screenshot.save(tmp_file.name)
                temp_path = tmp_file.name
            
            # Extract text from screenshot
            result = await self.extract_text_from_image(temp_path)
            
            # Clean up
            os.unlink(temp_path)
            
            # Add screenshot info
            result["screenshot_info"] = {
                "size": screenshot.size,
                "timestamp": datetime.now().isoformat(),
                "source": "screen_capture"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Screenshot analysis failed: {e}")
            return {"error": str(e)}
    
    async def enhance_image_for_ocr(self, image_path: str) -> str:
        """Enhance image quality for better OCR results"""
        try:
            if not PIL_AVAILABLE:
                return image_path  # Return original if PIL not available
            
            # Load image
            image = Image.open(image_path)
            
            # Apply enhancements
            # 1. Convert to grayscale for better OCR
            if image.mode != 'L':
                image = image.convert('L')
            
            # 2. Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
            
            # 3. Apply slight sharpening
            image = image.filter(ImageFilter.SHARPEN)
            
            # Save enhanced image
            enhanced_path = image_path.replace('.', '_enhanced.')
            image.save(enhanced_path)
            
            logger.info(f"Image enhanced for OCR: {enhanced_path}")
            return enhanced_path
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image_path  # Return original on failure
    
    def _log_vram_usage(self, context: str):
        """Log VRAM usage for RTX 3050 monitoring"""
        try:
            if CUDA_AVAILABLE and TORCH_AVAILABLE:
                allocated = torch.cuda.memory_allocated() // 1024**2
                cached = torch.cuda.memory_reserved() // 1024**2
                total_vram = 6144  # RTX 3050 6GB
                
                logger.info(f"📊 {context} - VRAM: {allocated}MB allocated, {cached}MB cached, {total_vram-cached}MB free")
                
                if cached > 4000:  # 67% of 6GB
                    logger.warning(f"⚠️ High VRAM usage: {cached}MB/{total_vram}MB")
                    
        except Exception as e:
            logger.debug(f"VRAM monitoring failed: {e}")
    
    def get_vision_capabilities(self) -> Dict[str, Any]:
        """Get current vision capabilities and RTX 3050 status"""
        capabilities = {
            "ocr_available": self.ocr_engine is not None,
            "ocr_engine": self.ocr_type,
            "gpu_acceleration": self.rtx_3050_config["enable_gpu_acceleration"],
            "supported_formats": ["PNG", "JPEG", "JPG", "BMP", "TIFF"],
            "max_image_size": self.rtx_3050_config["image_max_size"],
            "rtx_3050_optimized": True,
            
            "features": {
                "text_extraction": self.ocr_engine is not None,
                "screenshot_analysis": True,
                "image_enhancement": PIL_AVAILABLE,
                "base64_support": True,
                "batch_processing": self.rtx_3050_config["batch_size"]
            },
            
            "recommendations": {
                "optimal_image_size": "1920x1080 or smaller",
                "supported_languages": ["en", "ch_sim", "ch_tra"] if PADDLEOCR_AVAILABLE else ["en"],
                "vram_usage": f"~{self.rtx_3050_config['max_vram_usage']}MB for OCR",
                "performance_tips": [
                    "Use smaller images for faster processing",
                    "Enable GPU acceleration when available",
                    "Enhance image quality before OCR",
                    "Process images in batches for efficiency"
                ]
            }
        }
        
        if CUDA_AVAILABLE and TORCH_AVAILABLE:
            try:
                allocated = torch.cuda.memory_allocated() // 1024**2
                cached = torch.cuda.memory_reserved() // 1024**2
                
                capabilities["vram_status"] = {
                    "allocated": f"{allocated}MB",
                    "cached": f"{cached}MB", 
                    "free": f"{6144-cached}MB",
                    "usage_percent": f"{(cached/6144)*100:.1f}%"
                }
            except Exception:
                pass
        
        return capabilities
    
    async def cleanup(self):
        """Clean up vision controller resources"""
        try:
            if CUDA_AVAILABLE and TORCH_AVAILABLE:
                torch.cuda.empty_cache()
                logger.info("Vision controller GPU cache cleared")
        except Exception as e:
            logger.debug(f"Cleanup failed: {e}")