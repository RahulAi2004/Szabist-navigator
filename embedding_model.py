"""
DINOv2 embedding model handling
"""

import torch
import torch.nn as nn
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


class FinetunedDINOv2Classifier:
    """
    Fine-tuned DINOv2 + classification head loaded from checkpoints/.
    Architecture: DINOv2 ViT-S/14 backbone (frozen) -> BatchNorm1d(384) -> Linear(384, 12)
    """

    def __init__(self, checkpoint_path: Union[str, Path]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint_path = Path(checkpoint_path)

        logger.info(f"Loading fine-tuned DINOv2 classifier from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")

        self.idx_to_class: dict = ckpt["idx_to_class"]
        self.class_names: list  = ckpt["class_names"]
        self.num_classes: int   = ckpt["num_classes"]

        # Rebuild backbone
        self.backbone = torch.hub.load("facebookresearch/dinov2", ckpt["backbone_name"])
        backbone_sd = {k[len("backbone."):]: v
                       for k, v in ckpt["model_state_dict"].items()
                       if k.startswith("backbone.")}
        self.backbone.load_state_dict(backbone_sd)
        self.backbone.eval().to(self.device)

        # Rebuild classifier: LayerNorm(384) + Linear(384, num_classes)
        embed_dim = ckpt["embedding_dim"]
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, self.num_classes),
        )
        classifier_sd = {k[len("classifier."):]: v
                         for k, v in ckpt["model_state_dict"].items()
                         if k.startswith("classifier.")}
        self.classifier.load_state_dict(classifier_sd)
        self.classifier.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(ckpt["image_size"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

        logger.info(f"Fine-tuned classifier ready — {self.num_classes} classes, "
                    f"val_acc={ckpt.get('best_val_acc', '?'):.3f}")

    def predict_from_pil(self, pil_image: Image.Image) -> dict:
        """Return predicted location, confidence, and full probability dict."""
        tensor = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.backbone(tensor)          # (1, 384)
            logits    = self.classifier(embedding)     # (1, 12)
            probs     = F.softmax(logits, dim=1)[0]   # (12,)

        top_idx    = int(probs.argmax())
        location   = self.idx_to_class[top_idx]
        confidence = float(probs[top_idx])

        all_probs = {self.idx_to_class[i]: float(probs[i])
                     for i in range(self.num_classes)}

        return {
            "location":   location,
            "confidence": confidence,
            "all_probs":  all_probs,
        }
