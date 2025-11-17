# atomsurf

## :construction_worker: Installation

This implementation requires Python >= 3.7. Install the remaining dependencies using conda and pip:

```bash
conda create -n atomsurf -y
conda activate atomsurf
conda install python=3.8
conda install boost=1.73.0 dssp -c conda-forge -c salilab # if this fails, it can be ignored and preprocessing should be adapted

# Now let's install all of the torch/pyg dependencies !
# For GPU support, conda is better
conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.3.0 pytorch-scatter pytorch-sparse pytorch-spline-conv pytorch-cluster -c pyg
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
python -c "import torch; print(torch.cuda.is_available())"

# Otherwise, pip is simpler
pip install torch
pip install torch_geometric==2.3.0 torch_scatter torch_sparse torch_spline_conv torch_cluster pyg_lib -f https://data.pyg.org/whl/torch-1.13.1+cpu.html

# Finally, let's install other dependencies
# Install diffusion-net
pip install git+https://github.com/pvnieo/diffusion-net-plus.git
pip install -r requirements.txt
```
