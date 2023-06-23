# 해당 모델 검색 (ex: Geforce 750TI)
## Compute Capability 값을 기억 (ex: Geforce 750TI -> 5.0)
https://ko.wikipedia.org/wiki/CUDA

# 해당 cuda 버전을 현재 GPU가 지원하는지 확인
https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#id4



# 윈도우
# 드라이버
## 528.49 버전이 cuda 11.8과 호환
https://www.nvidia.com/download/driverResults.aspx/199654/en-us/

# cuda
https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local

# cudnn 다운로드 (해당 cuda 버전 다운로드)
# 압축해제 -> bin, include, lib 3개의 폴더를
-> C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8 
경로에 복사




# 우분투

# 설치 전 확인 (NVIDIA 드라이버)
$ nvidia-smi
$ cat /proc/driver/nvidia/version

# GPU 모델과 정보 확인 (ex: Geforce 750TI)
$ lspci -k

# 해당 메뉴로 이동
-> CUDA-Enabled GeForce and TITAN Products






# pytorch 호환성 확인
참조: https://pytorch.org/get-started/locally/

# pytorch 의 최대 지원 cuda 버전을 확인

참조: https://hwk0702.github.io/python/tips/2021/04/23/NVIDEA/

# 우분투에 설치가능한 드라이버 검색
## recommended 가 달린 버전을 기억
$ ubuntu-drivers devices

# 저장소 추가
$ sudo add-apt-repository ppa:graphics-drivers/ppa
$ sudo apt update

# 드라이버 설치
$ sudo apt-get install nvidia-driver-[recommended 버전]
재부팅

$ sudo apt-get install nvidia-driver-530
$ sudo reboot now

#cuda toolkit
https://en.wikipedia.org/wiki/CUDA   #호환성
https://developer.nvidia.com/cuda-toolkit-archive

# cuda 11.8 을 설치해야하기에 toolkit 11.8 을 설치
# 해당 아카이브 페이지에서 운영체제 및 기타 환경을 선택 후 나오는 명령창 그대로 입력하여 설치

$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
$ sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
$ sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
$ sudo apt-get update
$ sudo apt-get -y install cuda


# 환경변수 설정
$ nano ~/.bashrc
# 가장 아래 추가

## CUDA and cuDNN paths
export PATH=/usr/local/cuda-[VERSION]/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-[VERSION]/lib64:${LD_LIBRARY_PATH}

## 예:
export PATH=/usr/local/cuda-11.8/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}

# 추가한 내용 적용
$ source ~/.bashrc

#CUDA 버전 확인
$ nvcc -V

#CUDNN 설치

#해당 경로에서 현재 OS종류,버전 및 CUDA 버전과 일치하는 항목을 받아 설치 
https://developer.nvidia.com/rdp/cudnn-archive

## .deb 는 바로 실행하여 설치



## tgz 압축으로 받았다면

## 압축해제
$ tar -xzvf cudnn-linux-x86_64-8.9.1.23_cuda11-archive.tgz.xz

## 붙여넣고 권한설정
$ sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
$ sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*

$ sudo cp cudnn-linux-x86_64-8.9.1.23_cuda11-archive/include/cudnn*.h /usr/local/cuda/include
$ sudo cp cudnn-linux-x86_64-8.9.1.23_cuda11-archive/lib/libcudnn* /usr/local/cuda/lib64
$ sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*



## 확인(압축해제를 직접하지 않았다면 사용안됨)
$ cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2


# pytorch gpu 버전 설치
다운로드: https://pytorch.org/get-started/locally/

# 사용할 (가상)환경을 활성화하고 해당 환경에 설치 (conda or pip 둘중하나)
(env)$ conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
(env)& pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# https://velog.io/@somnode/gpu-cuda-driver-tensorflow-pytorch-version-compatibility
>>> import torch
>>> torch.cuda.is_available()
