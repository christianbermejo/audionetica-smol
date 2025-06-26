import os
import json
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoConfig

class CacheManager:
    def __init__(self, cache_dir="./model_cache"):
        """Initialize cache manager with specified cache directory"""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_index_path = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_cache_index()
        
    def _load_cache_index(self):
        """Load the cache index from disk or create a new one"""
        if self.cache_index_path.exists():
            with open(self.cache_index_path, 'r') as f:
                return json.load(f)
        else:
            return {"models": {}}
            
    def _save_cache_index(self):
        """Save the cache index to disk"""
        with open(self.cache_index_path, 'w') as f:
            json.dump(self.cache_index, f, indent=2)
            
    def is_model_cached(self, model_id):
        """Check if a model is already cached"""
        return model_id in self.cache_index["models"]
        
    def get_model_path(self, model_id):
        """Get the path to a cached model"""
        if self.is_model_cached(model_id):
            return str(self.cache_dir / self.cache_index["models"][model_id]["path"])
        return None
        
    def cache_model(self, model_id, model_type):
        """Download and cache a model"""
        if self.is_model_cached(model_id):
            print(f"Model {model_id} already cached")
            return self.get_model_path(model_id)
            
        print(f"Downloading and caching model {model_id}...")
        
        # Create a directory for this model
        model_dir = self.cache_dir / model_id.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Download the model files
        try:
            # Use snapshot_download to get all model files
            local_dir = snapshot_download(
                repo_id=model_id,
                cache_dir=str(model_dir),
                local_dir=str(model_dir),
                token=True
            )
            
            # Get model info for the cache index
            try:
                config = AutoConfig.from_pretrained(model_id)
                model_info = {
                    "name": model_id,
                    "type": model_type,
                    "path": model_id.replace("/", "_"),
                    "config": config.to_dict() if hasattr(config, "to_dict") else {}
                }
            except Exception:
                model_info = {
                    "name": model_id,
                    "type": model_type,
                    "path": model_id.replace("/", "_"),
                    "config": {}
                }
                
            # Update cache index
            self.cache_index["models"][model_id] = model_info
            self._save_cache_index()
            
            print(f"Model {model_id} cached successfully")
            return str(model_dir)
            
        except Exception as e:
            print(f"Error caching model {model_id}: {e}")
            # Clean up failed download
            if model_dir.exists():
                shutil.rmtree(model_dir)
            return None
            
    def clear_cache(self, model_id=None):
        """Clear the cache for a specific model or all models"""
        if model_id is None:
            # Clear all models
            for model_id in list(self.cache_index["models"].keys()):
                model_path = self.cache_dir / self.cache_index["models"][model_id]["path"]
                if model_path.exists():
                    shutil.rmtree(model_path)
                del self.cache_index["models"][model_id]
            self._save_cache_index()
            print("All models cleared from cache")
        else:
            # Clear specific model
            if model_id in self.cache_index["models"]:
                model_path = self.cache_dir / self.cache_index["models"][model_id]["path"]
                if model_path.exists():
                    shutil.rmtree(model_path)
                del self.cache_index["models"][model_id]
                self._save_cache_index()
                print(f"Model {model_id} cleared from cache")
            else:
                print(f"Model {model_id} not found in cache")
                
    def get_cache_info(self):
        """Get information about the cache"""
        total_size = 0
        model_sizes = {}
        
        for model_id, info in self.cache_index["models"].items():
            model_path = self.cache_dir / info["path"]
            size = sum(f.stat().st_size for f in model_path.glob('**/*') if f.is_file())
            model_sizes[model_id] = size
            total_size += size
            
        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "model_count": len(self.cache_index["models"]),
            "models": model_sizes
        }
