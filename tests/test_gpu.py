import torch
import logging
import warnings

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
            gpu_name = torch.cuda.get_device_name(0)
            print(f"{Colors.GREEN}GPU is accessible: {gpu_name}{Colors.RESET}")
            logging.info(f"GPU detected: {gpu_name}")
            return True
        else:
            print(f"{Colors.RED}No GPU is accessible.{Colors.RESET}")
            logging.warning("No GPU is accessible.")
            return False
    except Exception as e:
        print(f"{Colors.YELLOW}An error occurred: {e}{Colors.RESET}")
        logging.error(f"Error during GPU test: {e}")
        warnings.warn(f"Test failed due to an exception: {e}")
        return False

if __name__ == "__main__":
    result = test_gpu_access()
    if result:
        logging.info(f"{Colors.GREEN}Test passed: GPU is accessible.{Colors.RESET}")
    else:
        logging.info(f"{Colors.RED}Test failed: GPU is not accessible.{Colors.RESET}")
