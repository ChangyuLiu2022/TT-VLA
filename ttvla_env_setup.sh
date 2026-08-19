conda create -n ttvla  python==3.10
conda activate ttvla

pip install torch==2.2.0 
pip install numpy==1.26.4

conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia

cd openvla && pip install -e . && cd ..
pip install -U tyro
pip install datasets==3.3.2

# special install for flash attention
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# install other dependencies
cd ManiSkill && pip install -e . && cd ..
cd SimplerEnv && pip install -e . && cd ..


cd VLAC && pip install -e . && cd ..
pip install tokenizers==0.21.4

#for video flash
pip install av==15.1.0
pip install decord==0.6.0




##############no need to install#########
#############followings are for data collection and evaluation, which are not necessary for training the VLA model. You can skip them if you only want to train the VLA model.#############
# following is for collecting data.
conda create -n octo_env -y python=3.10
conda activate octo_env

git clone https://github.com/octo-models/octo.git

cd ManiSkill && pip install -e . && cd ..

cd octo && pip install -e . && pip install -r requirements.txt && cd ..
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 "nvidia-cudnn-cu11>=8.7,<9.0" --index-url https://download.pytorch.org/whl/cu118
pip install -U tyro
pip install scipy==1.12.0

cd SimplerEnv && pip install -e . && cd ..