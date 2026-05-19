"""
DINOv2 embedding model handling
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Union, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DINOv2Embedder:
    """
    Handles DINOv2 model loading and embedding generation
    """
    
    def __init__(self, model_name: str = "dinov2_vits14", device: str = "cuda"):
        """
        Initialize the DINOv2 embedder
        
        Args:
            model_name: Name of the DINOv2 model to use
            device: Device to run the model on ("cuda" or "cpu")
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading DINOv2 model: {model_name} on {self.device}")
        
        # Load pre-trained model from torch hub
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Define image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
        
        logger.info("DINOv2 model loaded successfully")
    
    def _load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load and validate image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        try:
            image = Image.open(image_path).convert("RGB")
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def get_embedding(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Generate embedding for a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Embedding as numpy array (1D, shape: (384,))
        """
        image = self._load_image(image_path)
        if image is None:
            return None
        
        # Preprocess image
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = self.model(image_tensor)
        
        # Normalize embedding and convert to numpy
        embedding = F.normalize(embedding, p=2, dim=1)
        embedding = embedding.cpu().numpy().astype(np.float32)
        
        return embedding[0]  # Return 1D array
    
    def get_embeddings_batch(self, image_paths: List[Union[str, Path]], 
                            batch_size: int = 8) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of images
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing
            
        Returns:
            List of embeddings as numpy arrays
        """
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_paths = []
            
            # Load images
            for path in batch_paths:
                image = self._load_image(path)
                if image is not None:
                    batch_images.append(self.transform(image))
                    valid_paths.append(path)
            
            if not batch_images:
                logger.warning(f"No valid images in batch starting at index {i}")
                continue
            
            # Stack into batch tensor
            batch_tensor = torch.stack(batch_images).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                batch_embeddings = self.model(batch_tensor)
            
            # Normalize and convert to numpy
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            batch_embeddings = batch_embeddings.cpu().numpy().astype(np.float32)
            
            embeddings.extend(batch_embeddings)
            
            logger.info(f"Processed batch {i//batch_size + 1}: {len(valid_paths)} images")
        
        return embeddings

    def get_embedding_from_pil(self, pil_image: Image.Image) -> np.ndarray:
        """In-memory variant of get_embedding — accepts a PIL Image directly."""
        image = pil_image.convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(image_tensor)
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy().astype(np.float32)[0]
