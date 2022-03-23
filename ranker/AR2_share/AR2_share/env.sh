source /quantus-nfs/zh/.bashrc

# default torch 1.7.1 cuda 110

sudo conda uninstall -y pip
sudo conda install -y pip
sudo conda install -y -c pytorch faiss-gpu
sudo pip install sklearn
sudo pip install wget
sudo pip install pytrec-eval
sudo pip install transformers==4.7.0
sudo pip install tensorboardX
sudo apt update
sudo apt install -y byobu

# source /vc_data/users/v-zhhang/.bashrc