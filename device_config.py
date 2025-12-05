# device_config.py
"""
Configuration centralisée pour la gestion CPU/GPU dans le système RAG.

Modes disponibles via RAG_USE_GPU:
- "0" ou "false" : Force CPU
- "1" ou "true"  : Force GPU (erreur si indisponible)
- "auto" (défaut): Détection automatique

Exemple d'utilisation:
    from device_config import get_device, USE_GPU, is_gpu_available
    
    device = get_device()  # "cuda" ou "cpu"
    if USE_GPU:
        # Charger modèle GPU
"""

import os
import sys

# =============================================================================
# CONFIGURATION PRINCIPALE
# =============================================================================

def _parse_gpu_env() -> str:
    """Parse la variable d'environnement RAG_USE_GPU."""
    value = os.environ.get("RAG_USE_GPU", "auto").lower().strip()
    if value in ("0", "false", "no", "off", "cpu"):
        return "cpu"
    elif value in ("1", "true", "yes", "on", "gpu", "cuda"):
        return "gpu"
    else:
        return "auto"

def _check_cuda_available() -> bool:
    """Vérifie si CUDA est disponible via PyTorch."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
    except Exception as e:
        print(f"⚠️  Erreur lors de la vérification CUDA: {e}")
        return False

def _get_cuda_device_name() -> str:
    """Récupère le nom du GPU si disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "Unknown"

# =============================================================================
# VARIABLES GLOBALES (calculées une seule fois au chargement)
# =============================================================================

_GPU_MODE = _parse_gpu_env()
_CUDA_AVAILABLE = _check_cuda_available()

# Déterminer le device effectif
if _GPU_MODE == "cpu":
    USE_GPU = False
    DEVICE = "cpu"
elif _GPU_MODE == "gpu":
    if _CUDA_AVAILABLE:
        USE_GPU = True
        DEVICE = "cuda"
    else:
        print("⚠️  RAG_USE_GPU=1 mais CUDA non disponible. Forçage CPU.")
        USE_GPU = False
        DEVICE = "cpu"
else:  # auto
    USE_GPU = _CUDA_AVAILABLE
    DEVICE = "cuda" if _CUDA_AVAILABLE else "cpu"

# =============================================================================
# API PUBLIQUE
# =============================================================================

def get_device() -> str:
    """
    Retourne le device à utiliser : "cuda" ou "cpu".
    
    Returns:
        str: "cuda" si GPU activé et disponible, sinon "cpu"
    """
    return DEVICE

def is_gpu_available() -> bool:
    """
    Vérifie si un GPU CUDA est disponible.
    
    Returns:
        bool: True si CUDA disponible
    """
    return _CUDA_AVAILABLE

def is_gpu_enabled() -> bool:
    """
    Vérifie si le mode GPU est activé (et disponible).
    
    Returns:
        bool: True si USE_GPU est True
    """
    return USE_GPU

def get_device_info() -> dict:
    """
    Retourne les informations complètes sur la configuration device.
    
    Returns:
        dict: Informations sur GPU/CPU
    """
    info = {
        "device": DEVICE,
        "use_gpu": USE_GPU,
        "cuda_available": _CUDA_AVAILABLE,
        "gpu_mode_env": _GPU_MODE,
        "env_var": os.environ.get("RAG_USE_GPU", "auto"),
    }
    
    if _CUDA_AVAILABLE:
        try:
            import torch
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
            info["cuda_version"] = torch.version.cuda
            info["pytorch_version"] = torch.__version__
        except Exception as e:
            info["gpu_info_error"] = str(e)
    
    return info

def print_device_status():
    """Affiche le statut du device dans la console."""
    info = get_device_info()
    
    if USE_GPU:
        gpu_name = info.get("gpu_name", "Unknown")
        gpu_mem = info.get("gpu_memory_total", "?")
        print(f"🚀 GPU activé: {gpu_name} ({gpu_mem})")
    else:
        reason = "forcé par RAG_USE_GPU=0" if _GPU_MODE == "cpu" else "CUDA non disponible"
        print(f"💻 Mode CPU ({reason})")

# =============================================================================
# HELPER POUR MODÈLES
# =============================================================================

def get_torch_device():
    """
    Retourne un objet torch.device pour les modèles PyTorch.
    
    Returns:
        torch.device: Device PyTorch
    """
    try:
        import torch
        return torch.device(DEVICE)
    except ImportError:
        raise ImportError("PyTorch requis pour get_torch_device(). Installez torch.")

# =============================================================================
# AFFICHAGE AU CHARGEMENT DU MODULE
# =============================================================================

if __name__ != "__main__":
    # Afficher le statut au premier import
    print_device_status()
