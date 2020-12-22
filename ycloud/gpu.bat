#!/bin/bash          

if [ -n "$1" ]
then

DATA_FILE="/home/sergey/mnt/st1500/Usr/Sergey/TheJob/clientsProjects/data/Webvork/real_2019-12-01-2019-12-31_confirmed_not-system-wm_5e89eecc02efc.csv"
SCHEMA_FILE="/home/sergey/mnt/st1500/Usr/Sergey/TheJob/clientsProjectsNew/wvproject/configurations/schema.yaml"
JUPYTER_NOTEBOOK_CONFIG="/home/sergey/mnt/st1500/Usr/Sergey/TheJob/clientsProjectsNew/wvproject/ycloud/gpu/jupyter_notebook_config.py"

echo "*****************************"
echo "*                           *"
echo "*    local configurations   *"
echo "*                           *"
echo "*****************************"

# ssh $1
rm ~/.ssh/known_hosts
# ssh -N -f -tt -L localhost:6001:localhost:7000 $1

echo   git credentials ----
scp ~/.ssh/gitHubSergeyBridge_rsa $1:~/.ssh
scp /etc/locale.gen $1:/etc


echo "*****************************"
echo "*                           *"
echo "*    remote configurations  *"
echo "*                           *"
echo "*****************************"

ssh $1 -tt << EOF

  mkdir "data"
  mkdir "Downloads"

  echo "*****************  locale settings  **************"
    sudo localedef ru_RU.UTF-8 -i ru_RU -f UTF-8

    sudo localectl set-locale LANG=en_US.UTF-8
    sudo localectl set-locale LANGUAGE=en_US.UTF-8

    sudo locale-gen
    sudo locale


  echo "*****************  apt update  **************"
    sudo apt-get update
    sudo apt install docker.io
    echo "sudo apt-get --yes upgrade------------------------"

    echo '* libraries/restart-without-asking boolean true' | sudo debconf-set-selections
    sudo apt-get --yes upgrade

    echo PRESS CTRL-C

EOF
ssh $1   "echo PRESS CTRL-C after EOF"

ssh $1 -tt << EOF

  echo "****************** 1. NVIDIA drivers installing *******************************"
    cd $HOME/Downloads

    wget http://ru.download.nvidia.com/tesla/440.64.00/NVIDIA-Linux-x86_64-440.64.00.run
    sudo apt-get --yes install ./NVIDIA-Linux-x86_64-440.64.00.run

  # echo "====== CUDA installing ======================"
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
    sudo apt-get update
    sudo apt-get -y install cuda
    echo PRESS CTRL-C

EOF

ssh $1 -tt << EOF

  echo "====== anaconda fetches installing ======================"

  sudo apt-get --yes install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 \
         libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

  echo PRESS CTRL-C

EOF
ssh $1 -tt << EOF
  echo "====== anaconda installing ======================"
    wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
    bash  ./Anaconda3-2020.02-Linux-x86_64.sh -b -p ~/anaconda3

  echo PRESS CTRL-C

EOF
ssh $1 -tt << EOF

  echo "***********  conda init *********************************"

    ~/anaconda3/bin/conda init
    source ~/.bashrc


  echo "====== pandas installing ======================"
    conda config --add channels conda-forge
    conda update -n base -c defaults conda

   echo PRESS CTRL-C

EOF
ssh $1 -tt << EOF

   # conda install --yes jupyter
   conda activate
    pip install  catboost
    pip install  tensorflow
    pip install  chart_studio
    pip install  cufflinks


    mkdir ~/wvproj &&  cd ~/wvproj

  echo PRESS CTRL-C

EOF
ssh $1 -tt << EOF

  echo "========== git cloning ==========================="
  git config --global user.name "SergeyBridge"
  git config --global user.email "sergey.novozhilov@yandex.ru"

  git init
  GIT_SSH_COMMAND='ssh -i ~/.ssh/gitHubSergeyBridge_rsa \
    -o UserKnownHostsFile=/dev/null \
    -o StrictHostKeyChecking=no' \
    git clone git@github.com:SergeyBridge/wvproj.git

  eval "$(ssh-agent -s)"
  ssh-add ~/.ssh/gitHubSergeyBridge_rsa
  # git stash
  # git pull git@github.com:SergeyBridge/wvproj.git

  echo PRESS CTRL-C

EOF

scp  $DATA_FILE SCHEMA_FILE $1:data
ssh $1 mkdir ".jupyter"
scp  $JUPYTER_NOTEBOOK_CONFIG $1:.jupyter

else
  echo "gpu.bat gpu@host_address: expected 1 parameter - remote host address "
fi
