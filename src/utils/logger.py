import logging
import os
from datetime import datetime

def setup_logger(log_dir: str = "logs") -> None:
    """Sets up a centralized logger that outputs to both terminal and a log file."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"restaurant_analysis_{timestamp}.log")
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s", # Keeping it simple for the terminal output like the old print
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    # Create a specific formatter for the file handler to include timestamps
    file_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger().handlers[1].setFormatter(file_formatter)
    
    # Silence third-party verbose loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
