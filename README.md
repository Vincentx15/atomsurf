# atomsurf


## :construction_worker: Installation
This implementation requires Python >= 3.7.

We have a dependency on pymesh, which can be installed following their homepage.
```bash
git clone https://github.com/PyMesh/PyMesh.git
cd PyMesh
git submodule update --init
export PYMESH_PATH=`pwd`

# Install Pymesh dependencies with apt-get
apt-get install libeigen3-dev libgmp-dev libgmpxx4ldbl libmpfr-dev libboost-dev libboost-thread-dev libtbb-dev python3-dev
# Or in jean zay by loading modules
module load gmp/6.1.2 eigen/3.3.7-mpi cmake/3.21.3 mpfr/4.0.2 boost/1.70.0

./setup.py build
./setup.py install

# Check everything works ok :
python -c "import pymesh; pymesh.test()"
```

Install the remaining dependencies using conda and pip:

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


