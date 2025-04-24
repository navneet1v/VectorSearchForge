FROM opensearchstaging/remote-vector-index-builder:faiss-base-latest

USER root
RUN apt-get update && apt-get upgrade -y && apt-get install libjemalloc2
RUN conda install -c conda-forge -c nvidia -c rapidsai cuvs=24.12 
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/libjemalloc.so.2
USER appuser
RUN pip install h5py cupy-cuda12x tqdm

WORKDIR /cuvs_benchmarks
COPY . /cuvs_benchmarks

ENV PYTHONPATH=${PYTHONPATH}:/tmp/faiss/build/faiss/python:/
RUN ["python", "/cuvs_benchmarks/test_imports.py"]
CMD ["python", "main.py"]