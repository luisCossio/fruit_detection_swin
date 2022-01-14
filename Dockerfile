ARG PYTORCH="1.8.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install mmcv-full==1.3.5+torch1.8.0+cu111 -f https://openmmlab.oss-accelerate.aliyuncs.com/mmcv/dist/index.html \
    pip install pip install mmdet==2.18.0
# RUN pip install mmcv-full==1.4.0 && pip install mmdet==2.20.0

# Install MMDetection
RUN conda clean --all

RUN git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git /swin_detection

# # Install specific code
RUN git clone https://github.com/luisCossio/fruit_detection_swin.git /fruit_detection_swin


ENV FORCE_CUDA="1"
# RUN pip install -r requirements/build.txt
# RUN pip install --no-cache-dir -e .

WORKDIR /swin_detection
RUN git clone https://github.com/NVIDIA/apex && cd apex \
    && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

RUN python setup.py develop

RUN mv /fruit_detection_swin/train2.py /swin_detection/tools/train2.py \
    && mv /fruit_detection_swin/minneapple_instance.py /swin_detection/configs/_base_/datasets/ \
    && mv /fruit_detection_swin/htc_swin.py /swin_detection/configs/swin/htc_swin.py
