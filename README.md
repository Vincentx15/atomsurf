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
conda create -n atom2d -y
conda activate atom2d
conda install python=3.8
conda install pytorch=1.13 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt
```


