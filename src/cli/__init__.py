"""xBD Pipeline CLI package."""

from pathlib import Path
import click
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging(log_file: Optional[Path] = None):
    """Configure logging with optional file output."""
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

def validate_path(path: Path, must_exist: bool = True) -> Path:
    """Validate and normalize a path."""
    path = Path(path).resolve()
    if must_exist and not path.exists():
        raise click.BadParameter(f"Path does not exist: {path}")
    return path 