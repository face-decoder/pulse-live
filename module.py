import sys
from pathlib import Path


def sync_source_module() -> Path:
    root = Path(__file__).parent.resolve()

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    return root


if __name__ != "__main__":
    sync_source_module()
