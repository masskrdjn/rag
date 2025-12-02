# cache_manager.py
from datetime import datetime, timedelta
from typing import Any, Optional, Dict
import json
import hashlib
import os

class CacheManager:
    """
    Cache multi-niveaux pour optimiser les requêtes répétées.
    
    Niveaux:
    1. In-memory (rapide, volatile)
    2. Disk cache (persistant, après redémarrage)
    """
    
    def __init__(self, 
                 max_memory_entries: int = 1000,
                 cache_dir: str = "./cache",
                 ttl_hours: int = 24):
        """
        Initialiser le cache manager
        
        Args:
            max_memory_entries: Max entrées en mémoire
            cache_dir: Répertoire pour cache disque
            ttl_hours: Durée de vie (heures)
        """
        self.memory_cache = {}
        self.max_memory = max_memory_entries
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_key(self, question: str, top_k: int) -> str:
        """Générer une clé cache unique"""
        query_str = f"{question.lower()}:{top_k}"
        hash_obj = hashlib.md5(query_str.encode())
        return hash_obj.hexdigest()
    
    def get(self, question: str, top_k: int) -> Optional[Dict]:
        """Récupérer du cache (memory first, puis disk)"""
        
        key = self._get_cache_key(question, top_k)
        
        # 1. Vérifier le cache mémoire
        if key in self.memory_cache:
            cached_data, timestamp = self.memory_cache[key]
            
            if datetime.now() - timestamp < self.ttl:
                print(f"✓ Cache HIT (memory) pour: '{question[:50]}...'")
                return cached_data
            else:
                # Entrée expirée, supprimer
                del self.memory_cache[key]
        
        # 2. Vérifier le cache disque
        try:
            disk_cache_file = os.path.join(self.cache_dir, f"{key}.json")
            
            if os.path.exists(disk_cache_file):
                with open(disk_cache_file, 'r', encoding='utf-8') as f:
                    cached_entry = json.load(f)
                
                file_time = datetime.fromtimestamp(
                    os.path.getmtime(disk_cache_file)
                )
                
                if datetime.now() - file_time < self.ttl:
                    print(f"✓ Cache HIT (disk) pour: '{question[:50]}...'")
                    
                    # Re-charger en mémoire
                    self.memory_cache[key] = (
                        cached_entry, 
                        datetime.now()
                    )
                    return cached_entry
                else:
                    # Fichier expiré, supprimer
                    os.remove(disk_cache_file)
        
        except Exception as e:
            print(f"⚠️ Erreur cache disque: {e}")
        
        return None
    
    def set(self, question: str, top_k: int, data: Dict):
        """Sauvegarder en cache (memory + disk)"""
        
        key = self._get_cache_key(question, top_k)
        timestamp = datetime.now()
        
        # 1. Sauvegarder en mémoire
        if len(self.memory_cache) >= self.max_memory:
            # Supprimer l'entrée la plus ancienne
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k][1]
            )
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = (data, timestamp)
        
        # 2. Sauvegarder sur disque
        try:
            disk_cache_file = os.path.join(self.cache_dir, f"{key}.json")
            with open(disk_cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Erreur sauvegarde cache: {e}")
    
    def clear_expired(self):
        """Nettoyer les entrées expirées"""
        
        now = datetime.now()
        
        # Mémoire
        expired_keys = [
            k for k, (_, ts) in self.memory_cache.items()
            if now - ts > self.ttl
        ]
        for k in expired_keys:
            del self.memory_cache[k]
        
        # Disque
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            try:
                file_time = datetime.fromtimestamp(
                    os.path.getmtime(filepath)
                )
                if now - file_time > self.ttl:
                    os.remove(filepath)
            except OSError:
                pass
        
        print(f"✓ Cache nettoyé: {len(expired_keys)} entrées mémoire supprimées")
    
    def get_stats(self) -> Dict:
        """Obtenir les statistiques du cache"""
        
        try:
            disk_entries = len(os.listdir(self.cache_dir))
        except FileNotFoundError:
            disk_entries = 0
            
        return {
            'memory_entries': len(self.memory_cache),
            'memory_max': self.max_memory,
            'disk_entries': disk_entries,
            'memory_usage_pct': len(self.memory_cache) / self.max_memory * 100 if self.max_memory > 0 else 0,
            'ttl_hours': self.ttl.total_seconds() / 3600
        }
