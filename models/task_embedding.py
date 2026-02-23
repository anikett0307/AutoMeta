# models/task_embedding.py
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root / "src"))

from maml_model import TaskSetEncoder as TaskEmbedding
__all__ = ["TaskEmbedding"]
