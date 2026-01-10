FROM harbor.z01.b-lab.uniondynamic.com/vllm/vllm-rocm:v0.11.0-mi100

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone