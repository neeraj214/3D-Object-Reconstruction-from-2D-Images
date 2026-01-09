"""
Data Curation Helper for Hybrid 2D-to-3D Point Cloud Reconstruction

This module provides utilities for curating high-quality 2D image and 3D model pairs
from professional datasets like Pix3D and ShapeNetCore.
"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path


def create_placeholder_metadata() -> List[Dict[str, Any]]:
    """
    Create placeholder metadata demonstrating the required structure for 2D-3D pairs.
    
    This function returns a small mock dataset with 5 items (2 chairs, 3 tables)
    showing the expected path and label structure for the hybrid 2D-to-3D project.
    
    Returns:
        List of dictionaries containing metadata for each 2D-3D pair with keys:
        - img_path: Path to 2D image file
        - model_path: Path to 3D model file  
        - label_id: Integer class label (0-9)
        - label_name: Human-readable class name
        - dataset_source: Origin dataset (Pix3D/ShapeNetCore)
        - quality_score: Subjective quality rating (1-5)
    """
    
    # Define class mapping for clarity
    class_names = {
        0: "chair",
        1: "table", 
        2: "sofa",
        3: "car",
        4: "lamp",
        5: "bottle",
        6: "display",
        7: "cabinet",
        8: "bench",
        9: "bookshelf"
    }
    
    # Create mock metadata for 5 items (2 chairs, 3 tables)
    placeholder_data = [
        {
            "img_path": "data/curated_dataset/images/chair_001.jpg",
            "model_path": "data/curated_dataset/models/chair_001.obj",
            "label_id": 0,
            "label_name": "chair",
            "dataset_source": "Pix3D",
            "quality_score": 4.5,
            "image_resolution": [640, 480],
            "model_vertices": 2500,
            "viewpoint": "front",
            "lighting": "natural",
            "notes": "High-quality real photo with clean 3D model"
        },
        {
            "img_path": "data/curated_dataset/images/chair_002.jpg",
            "model_path": "data/curated_dataset/models/chair_002.ply",
            "label_id": 0,
            "label_name": "chair",
            "dataset_source": "ShapeNetCore",
            "quality_score": 4.2,
            "image_resolution": [512, 512],
            "model_vertices": 1800,
            "viewpoint": "side",
            "lighting": "synthetic",
            "notes": "Synthetic render with detailed mesh"
        },
        {
            "img_path": "data/curated_dataset/images/table_001.jpg",
            "model_path": "data/curated_dataset/models/table_001.obj",
            "label_id": 1,
            "label_name": "table",
            "dataset_source": "Pix3D",
            "quality_score": 4.8,
            "image_resolution": [800, 600],
            "model_vertices": 3200,
            "viewpoint": "angled",
            "lighting": "natural",
            "notes": "Excellent alignment between image and 3D model"
        },
        {
            "img_path": "data/curated_dataset/images/table_002.jpg",
            "model_path": "data/curated_dataset/models/table_002.obj",
            "label_id": 1,
            "label_name": "table",
            "dataset_source": "Pix3D",
            "quality_score": 4.3,
            "image_resolution": [640, 480],
            "model_vertices": 2800,
            "viewpoint": "top",
            "lighting": "natural",
            "notes": "Good top-down view showing table surface"
        },
        {
            "img_path": "data/curated_dataset/images/table_003.jpg",
            "model_path": "data/curated_dataset/models/table_003.ply",
            "label_id": 1,
            "label_name": "table",
            "dataset_source": "ShapeNetCore",
            "quality_score": 4.0,
            "image_resolution": [512, 512],
            "model_vertices": 2100,
            "viewpoint": "front",
            "lighting": "synthetic",
            "notes": "Simple table design, good for baseline testing"
        }
    ]
    
    return placeholder_data


def validate_metadata_structure(metadata: List[Dict[str, Any]]) -> bool:
    """
    Validate that metadata contains all required fields.
    
    Args:
        metadata: List of metadata dictionaries
        
    Returns:
        True if all required fields are present, False otherwise
    """
    required_fields = {"img_path", "model_path", "label_id"}
    
    if not metadata:
        return False
        
    for item in metadata:
        if not all(field in item for field in required_fields):
            return False
            
        # Validate field types
        if not isinstance(item["img_path"], str) or not item["img_path"]:
            return False
        if not isinstance(item["model_path"], str) or not item["model_path"]:
            return False
        if not isinstance(item["label_id"], int) or item["label_id"] < 0 or item["label_id"] > 9:
            return False
            
    return True


def create_dataset_splits(metadata: List[Dict[str, Any]], 
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15,
                         seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    """
    Split metadata into train/validation/test sets while maintaining class balance.
    
    Args:
        metadata: Complete metadata list
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with keys 'train', 'val', 'test' containing metadata splits
    """
    import random
    random.seed(seed)
    
    # Group by class
    class_groups = {}
    for item in metadata:
        label_id = item["label_id"]
        if label_id not in class_groups:
            class_groups[label_id] = []
        class_groups[label_id].append(item)
    
    splits = {"train": [], "val": [], "test": []}
    
    # Split each class group
    for class_items in class_groups.values():
        n = len(class_items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        # Shuffle class items
        shuffled = class_items.copy()
        random.shuffle(shuffled)
        
        # Split
        splits["train"].extend(shuffled[:n_train])
        splits["val"].extend(shuffled[n_train:n_train + n_val])
        splits["test"].extend(shuffled[n_train + n_val:])
    
    return splits


def save_metadata_splits(splits: Dict[str, List[Dict[str, Any]]], 
                        output_dir: str = "data/curated_dataset/splits"):
    """
    Save metadata splits to JSON files.
    
    Args:
        splits: Dictionary containing train/val/test splits
        output_dir: Directory to save split files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        output_path = os.path.join(output_dir, f"{split_name}.json")
        with open(output_path, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"Saved {len(split_data)} items to {output_path}")


def analyze_metadata_distribution(metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the distribution of metadata across different dimensions.
    
    Args:
        metadata: List of metadata dictionaries
        
    Returns:
        Dictionary containing analysis results
    """
    analysis = {
        "total_items": len(metadata),
        "class_distribution": {},
        "dataset_source_distribution": {},
        "quality_score_stats": {
            "mean": 0.0,
            "min": float('inf'),
            "max": 0.0
        }
    }
    
    # Class distribution
    for item in metadata:
        label_id = item["label_id"]
        label_name = item["label_name"]
        key = f"{label_id}_{label_name}"
        analysis["class_distribution"][key] = analysis["class_distribution"].get(key, 0) + 1
    
    # Dataset source distribution
    for item in metadata:
        source = item["dataset_source"]
        analysis["dataset_source_distribution"][source] = analysis["dataset_source_distribution"].get(source, 0) + 1
    
    # Quality score statistics
    quality_scores = [item["quality_score"] for item in metadata]
    if quality_scores:
        analysis["quality_score_stats"]["mean"] = sum(quality_scores) / len(quality_scores)
        analysis["quality_score_stats"]["min"] = min(quality_scores)
        analysis["quality_score_stats"]["max"] = max(quality_scores)
    
    return analysis


if __name__ == "__main__":
    # Test the placeholder metadata creation
    print("Creating placeholder metadata...")
    metadata = create_placeholder_metadata()
    
    print(f"\nCreated {len(metadata)} placeholder items")
    
    # Validate structure
    is_valid = validate_metadata_structure(metadata)
    print(f"Metadata structure valid: {is_valid}")
    
    # Analyze distribution
    analysis = analyze_metadata_distribution(metadata)
    print(f"\nAnalysis Results:")
    print(f"Total items: {analysis['total_items']}")
    print(f"Class distribution: {analysis['class_distribution']}")
    print(f"Dataset source distribution: {analysis['dataset_source_distribution']}")
    print(f"Quality score stats: {analysis['quality_score_stats']}")
    
    # Create and save splits
    print(f"\nCreating dataset splits...")
    splits = create_dataset_splits(metadata)
    
    for split_name, split_data in splits.items():
        print(f"{split_name}: {len(split_data)} items")
    
    # Save to files (optional - uncomment to save)
    # save_metadata_splits(splits)
    
    # Display first item as example
    print(f"\nExample metadata item:")
    print(json.dumps(metadata[0], indent=2))
    
    print("\nData curation helper test completed successfully!")