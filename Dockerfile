FROM ubuntu:20.04
COPY . .


RUN apt-get update
RUN apt-get install -y apt-utils
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y gnupg2
RUN apt-get install -y wget
RUN apt-get install -y git
RUN apt-get install -y python3.9
RUN apt-get install -y curl
RUN apt-get install -y python3-distutils
RUN apt-get install -y python3-apt
RUN apt-get update
RUN apt-get install -y python3-dev
RUN apt-get install -y libosmesa6-dev libgl1-mesa-glx libglfw3
RUN apt-get install -y python3-pip
RUN apt-get install -y patchelf

ADD https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz .
ADD https://roboti.us/file/mjkey.txt .

RUN mkdir /root/.mujoco

## mujoco 210
RUN tar -xf mujoco210-linux-x86_64.tar.gz -C /root/.mujoco/

RUN mv mjkey.txt /root/.mujoco/

RUN cp -r /root/.mujoco/mujoco210/bin/* /usr/lib/


RUN pip install -r requirements.txt


RUN export LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV D4RL_SUPPRESS_IMPORT_ERROR=1

ENTRYPOINT ['sh','srcipts/start.sh']