import torch
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ANSI color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

def test_gpu_access():
    """Test if a GPU is accessible."""
    logging.info("Starting GPU accessibility test...")
    try:
        if torch.cuda.is_available():
            logging.info(f"{Colors.GREEN}GPU is accessible: {Colors.RESET}")
            for idx in range(torch.cuda.device_count()):
                logging.info(f"{Colors.GREEN}GPU {idx}: {torch.cuda.get_device_name(idx)}{Colors.RESET}")
            return True
        else:
            logging.warning(f"{Colors.YELLOW}No GPU is accessible.{Colors.RESET}")
            return False
    except Exception as e:
        logging.error(f"{Colors.RED}An error occurred: {e}{Colors.RESET}")
        return False

if __name__ == "__main__":
    test_gpu_access()
