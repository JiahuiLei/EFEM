conda remove -n efem --all -y
conda create -n efem python=3.8 -y
source activate efem

# install pytorch
echo ====INSTALLING PyTorch======
which python
which pip
conda install pytorch=1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y

# # install pytorch3d
echo ====INSTALLING=PYTORCH3D======
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d=0.6.1 -c pytorch3d -y

# # Install Pytorch Geometry
conda install pyg -c pyg -y

# install requirements
pip install cython
pip install -r requirements.txt
# pip install pyopengl==3.1.5 # for some bugs in the nvidia driver on our cluster

# build ONet Tools
cd lib_shape_prior
python setup.py build_ext --inplace
python setup_c.py build_ext --inplace # for kdtree in cuda 11