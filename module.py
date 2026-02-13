import sys
from pathlib import Path


def sync_source_module() -> Path:
    """
    Synchronize the source module directory with the current working directory.
    This also used for notebook environments to allow imports from the source modules.

    Args:
        None

    Returns:
        None

    Raises:
        None
    """
    root = Path(__file__).parent.resolve()

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    return root


# Allow automatic synchronization when imported
if __name__ != "__main__":
    sync_source_module()