FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Install base utilities and also the dependencies for faiss
RUN apt-get update \
    && apt-get install -y build-essential gfortran libblas-dev libopenblas-base zlib1g zlib1g-dev \
    && apt-get install -y wget nginx ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# install all the faiss dependencies
#RUN conda install -c pytorch -c nvidia -c rapidsai -c conda-forge python=3.11 faiss-gpu-raft=1.9.0 -y
RUN conda install -c pytorch -c rapidsai -c conda-forge -c nvidia python=3.11  pytorch/label/nightly::faiss-gpu-cuvs 'cuda-version>=12.0,<=12.5'

WORKDIR /opt/program

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"


COPY ./model_server/ .
RUN pip install -r requirements.txt
RUN chmod +x serve

ENTRYPOINT ["serve"]
