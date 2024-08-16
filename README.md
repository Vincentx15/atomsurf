# atomsurf


## :construction_worker: Installation
This implementation requires Python >= 3.7. Install the remaining dependencies using conda and pip:

```bash
conda create -n atomsurf -y
conda activate atomsurf
conda install python=3.8
conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg=2.3.0 pytorch-scatter pytorch-sparse -c pyg
conda install boost=1.73.0 dssp -c conda-forge -c salilab
pip install -r requirements.txt

pip install torch_geometric==2.3.0 pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
python -c "import torch; print(torch.cuda.is_available())"
# install diffusion-net
#pip install git+https://github.com/pvnieo/diffusion-net-plus.git
```


