from concurrent.futures.thread import ThreadPoolExecutor

import flask
import faiss
import math
import numpy as np
from timeit import default_timer as timer
from flask import request, jsonify
import uuid
import time
import traceback
import os
import sys
import gc

app = flask.Flask(__name__)

executor = ThreadPoolExecutor(max_workers=5)

d = 768
print(f"Creating dataset for jobid ....")
t1 = timer()
xb = np.random.rand(1_000_000, d).astype(dtype= np.float32)
t2 = timer()
print(f"Dataset created for jobid in : {t2 - t1} sec")

@app.route('/invocations', methods=["POST"])
def invoke():
    # model() is a hypothetical function that gets the inference output:
    job_id = str(uuid.uuid4())
    is_async = int(os.getenv('IS_ASYNC', 0))
    if is_async == 1:
        print("Running async innovacation of model endpoint")
        _run_job(job_id=job_id)
    else:
        print("Running sync innovacation of model endpoint")
        executor.submit(_run_job, job_id)
    
    return jsonify({"job_id": job_id}), 200
    


@app.route('/ping', methods=["GET"])
def ping():
    return flask.Response(response="\n", status=200, mimetype="application/json")


def _run_job(job_id):
    try:
        print(f"Job {job_id} is running")
        ids = None
        indexData(d, xb, ids, {}, "l2", f"testgpuIndex-{job_id}.cagra.graph")
        print(f"Indexing done for jobid: {job_id}")
        gc.collect()

    except Exception as e:
        print(f"An error occurred for job id : {job_id}: {e}")
        traceback.print_exc()


def indexData(d:int, xb:np.ndarray, ids:np.ndarray, indexingParams:dict, space_type:str, file_to_write:str="gpuIndex.cagra.graph"):
    #num_of_parallel_threads = max(math.floor(os.cpu_count() - 1), 1)
    num_of_parallel_threads = 8
    print(f"Setting number of parallel threads for graph build: {num_of_parallel_threads}")
    faiss.omp_set_num_threads(num_of_parallel_threads)
    res = faiss.StandardGpuResources()
    metric = faiss.METRIC_L2
    if space_type == "innerproduct":
        metric = faiss.METRIC_INNER_PRODUCT
    cagraIndexConfig = faiss.GpuIndexCagraConfig()
    cagraIndexConfig.intermediate_graph_degree = 64 if indexingParams.get('intermediate_graph_degree') is None else indexingParams['intermediate_graph_degree']
    cagraIndexConfig.graph_degree = 32 if indexingParams.get('graph_degree') == None else indexingParams['graph_degree']
    cagraIndexConfig.device = faiss.get_num_gpus() - 1
    cagraIndexConfig.store_dataset = False

    cagraIndexConfig.build_algo = faiss.graph_build_algo_IVF_PQ
    cagraIndexIVFPQConfig = faiss.IVFPQBuildCagraConfig()
    cagraIndexIVFPQConfig.kmeans_n_iters = 10 if indexingParams.get('kmeans_n_iters') == None else indexingParams['kmeans_n_iters']
    cagraIndexIVFPQConfig.pq_bits = 8 if indexingParams.get('pq_bits') == None else indexingParams['pq_bits']
    cagraIndexIVFPQConfig.pq_dim = 32 if indexingParams.get('pq_dim') == None else indexingParams['pq_dim']
    cagraIndexIVFPQConfig.n_lists = int(math.sqrt(len(xb))) if indexingParams.get('n_lists') == None else indexingParams['n_lists']
    cagraIndexIVFPQConfig.kmeans_trainset_fraction = 0.1 if indexingParams.get('kmeans_trainset_fraction') == None else indexingParams['kmeans_trainset_fraction']
    cagraIndexIVFPQConfig.conservative_memory_allocation = True
    cagraIndexConfig.ivf_pq_params = cagraIndexIVFPQConfig

    cagraIndexSearchIVFPQConfig = faiss.IVFPQSearchCagraConfig()
    cagraIndexSearchIVFPQConfig.n_probes = 30 if indexingParams.get('n_probes') == None else indexingParams['n_probes']
    cagraIndexConfig.ivf_pq_search_params = cagraIndexSearchIVFPQConfig

    print("Creating GPU Index.. with IVF_PQ")
    t1 = timer()
    cagraIVFPQIndex = faiss.GpuIndexCagra(res, d, metric, cagraIndexConfig)
    idMapIVFPQIndex = faiss.IndexIDMap(cagraIVFPQIndex)
    indexDataInIndex(idMapIVFPQIndex, ids, xb)
    t2 = timer()
    
    print(f"Indexing time: {t2 - t1} sec")

def indexDataInIndex(index: faiss.Index, ids, xb):
    if ids is None:
        index.train(xb)
    else:
        index.add_with_ids(xb, ids)