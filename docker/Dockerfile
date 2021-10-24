FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

RUN apt-get update -y

RUN pip install numpy scipy matplotlib librosa==0.6.0 tensorflow tensorboardX inflect==0.2.5 Unidecode==1.0.22 pillow jupyter numba==0.48 g2p_en pandas transforms3d
