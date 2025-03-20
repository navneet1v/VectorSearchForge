import faiss
from benchmarking.decorators.timer import timer_func
from benchmarking.utils.common_utils import get_omp_num_threads
import logging
from timeit import default_timer as timer
import math
import numpy as np

def indexData(d:int, xb:np.ndarray, ids:np.ndarray, indexingParams:dict, space_type:str, file_to_write:str="gpuIndex.cagra.graph"):
    num_of_parallel_threads = get_omp_num_threads()
    logging.info(f"Setting number of parallel threads for graph build: {num_of_parallel_threads}")
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
    cagraIndexConfig.refine_rate = 2.0 if indexingParams.get('refine_rate') == None else indexingParams.get('refine_rate')

    cagraIndexConfig.build_algo = faiss.graph_build_algo_IVF_PQ
    cagraIndexIVFPQConfig = faiss.IVFPQBuildCagraConfig()
    cagraIndexIVFPQConfig.kmeans_n_iters = 20 if indexingParams.get('kmeans_n_iters') == None else indexingParams['kmeans_n_iters']
    cagraIndexIVFPQConfig.pq_bits = 8 if indexingParams.get('pq_bits') == None else indexingParams['pq_bits']
    cagraIndexIVFPQConfig.pq_dim = 0 if indexingParams.get('pq_dim') == None else indexingParams['pq_dim']
    cagraIndexIVFPQConfig.n_lists = int(math.sqrt(len(xb))) if indexingParams.get('n_lists') == None else indexingParams['n_lists']
    cagraIndexIVFPQConfig.kmeans_trainset_fraction = 0.5 if indexingParams.get('kmeans_trainset_fraction') == None else indexingParams['kmeans_trainset_fraction']
    cagraIndexIVFPQConfig.force_random_rotation = True
    cagraIndexIVFPQConfig.conservative_memory_allocation = True
    cagraIndexConfig.ivf_pq_params = cagraIndexIVFPQConfig

    cagraIndexSearchIVFPQConfig = faiss.IVFPQSearchCagraConfig()
    cagraIndexSearchIVFPQConfig.n_probes = 20 if indexingParams.get('n_probes') == None else indexingParams['n_probes']
    cagraIndexConfig.ivf_pq_search_params = cagraIndexSearchIVFPQConfig

    print("Creating GPU Index.. with IVF_PQ")
    cagraIVFPQIndex = faiss.GpuIndexCagra(res, d, metric, cagraIndexConfig)
    idMapIVFPQIndex = faiss.IndexIDMap(cagraIVFPQIndex)

    t1 = timer()
    indexDataInIndex(idMapIVFPQIndex, ids, xb)
    t2 = timer()
    indexTime = t2 - t1
    t1 = timer()
    writeIndexMetrics = writeCagraIndexOnFile(idMapIVFPQIndex, cagraIVFPQIndex, file_to_write)
    t2 = timer()
    writeIndexTime = t2 - t1
    # This will ensure that when destructors of the index is called the internal indexes are deleted too.
    cagraIVFPQIndex.thisown = True
    idMapIVFPQIndex.own_fields = True
    del cagraIVFPQIndex
    del idMapIVFPQIndex
    return {
        "indexTime": indexTime, "writeIndexTime": writeIndexTime, "totalTime": indexTime + writeIndexTime, "unit": "seconds", 
        "gpu_to_cpu_index_conversion_time": writeIndexMetrics["gpu_to_cpu_index_conversion_time"] ,
        "write_to_file_time": writeIndexMetrics["write_to_file_time"]
    }


@timer_func
def indexDataInIndex(index: faiss.Index, ids, xb):
    index.add_with_ids(xb, ids)


@timer_func
def writeCagraIndexOnFile(idMapIndex: faiss.Index, cagraIndex: faiss.GpuIndexCagra, outputFileName: str):
    t1 = timer()
    cpuIndex = faiss.IndexHNSWCagra()
    # This will ensure that we have faster conversion time
    cpuIndex.base_level_only = True
    # This will ensure that when destructors of the index is called the internal indexes are deleted too.
    cpuIndex.own_fields = True
    cagraIndex.copyTo(cpuIndex)
    idMapIndex.index = cpuIndex
    t2 = timer()
    conversion_time = t2 - t1
    
    t1 = timer()
    faiss.write_index(idMapIndex, outputFileName)
    t2 = timer()
    write_to_file_time = t2 - t1
    del cpuIndex
    return {
        "gpu_to_cpu_index_conversion_time": conversion_time,
        "write_to_file_time": write_to_file_time
    }
