# Setup script for Introduction to Deep Learning course for Windows
# RUN IT LINE BY LINE!

# Create the env
py -m venv iDL_venv

# Activate the env
.\iDL_venv\Scripts\Activate

# Install the packages and PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
