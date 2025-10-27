FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

SHELL ["/bin/bash", "-c"]
RUN apt-get update -qq && apt-get upgrade -qq &&\
    apt-get install -qq man wget sudo vim tmux

RUN apt update

RUN apt install -y cudnn9

RUN yes | pip install --upgrade pip

RUN yes | pip install numpy==1.26.4 transformers accelerate

RUN pip cache purge

RUN pip install notebook

WORKDIR /home

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip", "0.0.0.0","--allow-root", "--no-browser"]
