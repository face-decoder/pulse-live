import os


class Window:
    SECONDS: float = float(os.getenv("WINDOW_SECONDS", "1.5"))
    MIN_FRAMES: int = int(os.getenv("MIN_FRAMES", "20"))


class Stream:
    FPS: int = int(os.getenv("TARGET_FPS", "15"))
    HEARTBEAT_SECONDS: float = 30.0
