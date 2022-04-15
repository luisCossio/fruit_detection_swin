ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Install MMDetection
RUN conda clean --all

# # Install specific repositories
RUN git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git /swin_detection \
    && git clone https://github.com/luisCossio/fruit_detection_swin.git /fruit_detection_swin \
    && pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.0/index.html

WORKDIR /swin_detection

ENV FORCE_CUDA="1"
RUN pip install -r /fruit_detection_swin/requirements.txt && pip install --no-cache-dir -e .

RUN git clone https://github.com/NVIDIA/apex \
    && cd apex \
    && pip install -v --disable-pip-version-check --no-cache-dir ./

RUN mv /fruit_detection_swin/train2.py /swin_detection/tools/train2.py \
    && mv /fruit_detection_swin/minneapple_instance.py /swin_detection/configs/_base_/datasets/ \
    && mv /fruit_detection_swin/cascade_mask_small.py /swin_detection/configs/swin/cascade_mask_small.py \
    && mv /fruit_detection_swin/cascade_mask_tiny.py /swin_detection/configs/swin/cascade_mask_tiny.py \
    && mv /fruit_detection_swin/coco.py /swin_detection/mmdet/datasets/coco.py \
    && mv /fruit_detection_swin/cascade_mask_rcnn_swin_fpn.py /swin_detection/configs/_base_/models/cascade_mask_rcnn_swin_fpn.py

RUN python setup.py develop
# RUN pip uninstall mmpycocotools pycocotools && pip install mmpycocotools pycocotools