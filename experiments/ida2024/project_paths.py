"""
    This script ensures that the project root is in the path.
    This makes the following import possible:

    >>> import project_path # Import this first!
    >>> from slisemap.slisemap import Slisemap
    >>> from experiment.data import get_autompg
"""
from pathlib import Path
import sys

PPROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
if str(PPROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PPROJECT_DIR))

DATA_DIR = PPROJECT_DIR / "experiments" / "data"
RESULTS_DIR = PPROJECT_DIR / "experiments" / "results"
MANUSCRIPT_DIR = PPROJECT_DIR / "manuscript" / "ida2024"
