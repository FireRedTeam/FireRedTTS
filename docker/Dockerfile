FROM docker-reg.devops.xiaohongshu.com/media/red_serving_test:xformer_tts.gpt2

ENV DEBIAN_FRONTEND=noninteractive 
ARG COS_PATH="https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/ruoli/third_party/"

RUN apt update && apt install -y libsndfile1 htop libibverbs-dev libwayland-dev libglib2.0-dev libhwloc-dev ffmpeg python3-soundfile  rustc cargo espeak-ng libsox-dev language-pack-en zip
RUN pip3 install torchaudio==2.5.0a0+56bc006 diffusers==0.31.0 sentencex==0.6.1 -i http://pypi.devops.xiaohongshu.com/simple --trusted-host pypi.devops.xiaohongshu.com --no-deps
RUN cd /workspace && wget -q ${COS_PATH}flash_attn-2.7.3+cu12torch2.5cxx11abiTRUE-cp310-cp310-linux_x86_64.whl && pip3 install flash_attn-2.7.3+cu12torch2.5cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

ARG repo_name
ADD $repo_name.tar.gz /workspace/$repo_name
RUN cd /workspace/$repo_name && pip3 install -e .
RUN cd /workspace/$repo_name/fireredtts/modules/text_normalizer && cmake . -DCMAKE_BUILD_TYPE=Release && make -j8 && cd /workspace

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libopenblas.so.0

WORKDIR /workspace/
