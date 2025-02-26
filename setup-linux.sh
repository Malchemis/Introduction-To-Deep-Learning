# Setup script for Introduction to Deep Learning course
# RUN IT LINE BY LINE!

# Create conda environment
conda create -n iDL python=3.12 -y

# Activate the environment
conda activate iDL

# Install PyTorch and Lightning
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# Install using pip the other packages
pip install -r requirements.txt

echo "Setup complete. Activate the environment with 'conda activate iDL'."
