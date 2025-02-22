import sys
# TODO: Find a better way to import faiss package rather than this.
sys.path.append('/tmp/faiss/build/faiss/python')
import faiss
from timeit import default_timer as timer
import math

from utils.common import get_omp_num_threads
from utils.decorators.timer import timer_func
from vector_data_accessor.accessor import VectorsDataset
import logging

logger = logging.getLogger(__name__)

def create_index(vectorsDataset:VectorsDataset, indexingParams:dict, space_type:str, file_to_write:str= "gpuIndex.cagra.graph"):
    num_of_parallel_threads = get_omp_num_threads()
    logging.info(f"Setting number of parallel threads for gpu based graph build: {num_of_parallel_threads}")
    faiss.omp_set_num_threads(num_of_parallel_threads)
    res = faiss.StandardGpuResources()
    metric = faiss.METRIC_L2
    if space_type == "innerproduct":
        metric = faiss.METRIC_INNER_PRODUCT
    cagraIndexConfig = faiss.GpuIndexCagraConfig()
    cagraIndexConfig.intermediate_graph_degree = 64 if indexingParams.get('intermediate_graph_degree') is None else indexingParams['intermediate_graph_degree']
    cagraIndexConfig.graph_degree = 32 if indexingParams.get('graph_degree') == None else indexingParams['graph_degree']
    # get the index of the GPU device. We assume that we are using only 1 GPU machine. If we are using more than 1 GPU
    # then we need to update this.
    cagraIndexConfig.device = faiss.get_num_gpus() - 1
    # This is to ensure that full dataset copy is not happening on the GPU machine during quantization
    cagraIndexConfig.store_dataset = False

    dataset_size = len(vectorsDataset.ids)

    cagraIndexConfig.build_algo = faiss.graph_build_algo_IVF_PQ
    cagraIndexIVFPQConfig = faiss.IVFPQBuildCagraConfig()
    cagraIndexIVFPQConfig.kmeans_n_iters = 0.5 if indexingParams.get('kmeans_n_iters') == None else indexingParams['kmeans_n_iters']
    # instead of 32 bits per dim, you are using only 8 bits per dim
    cagraIndexIVFPQConfig.pq_bits = 8 if indexingParams.get('pq_bits') == None else indexingParams['pq_bits']
    #cagraIndexIVFPQConfig.pq_dim = 16 if indexingParams.get('pq_dim') == None else indexingParams['pq_dim']
    compression_factor = 8
    # instead of using d dimension you use d/compression_factor dimensions.
    cagraIndexIVFPQConfig.pq_dim = int(vectorsDataset.dimensions / compression_factor)

    # In total instead of 128 * 32 bits(where dimension = 128) in total for 1 vector you are using ((128 / 8) * 8) = 128 bits in total
    # So this is a 32x compression.

    cagraIndexIVFPQConfig.n_lists = int(math.sqrt(dataset_size)) if indexingParams.get('n_lists') == None else indexingParams['n_lists']
    cagraIndexIVFPQConfig.kmeans_trainset_fraction = 10 if indexingParams.get('kmeans_trainset_fraction') == None else indexingParams['kmeans_trainset_fraction']
    cagraIndexIVFPQConfig.conservative_memory_allocation = True
    cagraIndexConfig.ivf_pq_params = cagraIndexIVFPQConfig

    cagraIndexSearchIVFPQConfig = faiss.IVFPQSearchCagraConfig()
    cagraIndexSearchIVFPQConfig.n_probes = 30 if indexingParams.get('n_probes') == None else indexingParams['n_probes']
    cagraIndexConfig.ivf_pq_search_params = cagraIndexSearchIVFPQConfig

    logger.info("Creating GPU Index.. with IVF_PQ")
    cagraIVFPQIndex = faiss.GpuIndexCagra(res, vectorsDataset.dimensions, metric, cagraIndexConfig)
    idMapIVFPQIndex = faiss.IndexIDMap(cagraIVFPQIndex)

    t1 = timer()
    indexDataInIndex(idMapIVFPQIndex, vectorsDataset.ids, vectorsDataset.vectors)
    t2 = timer()
    indexTime = t2 - t1

    t1 = timer()
    writeIndexMetrics = writeCagraIndexOnFile(idMapIVFPQIndex, cagraIVFPQIndex, file_to_write)
    logger.info("Write completed")
    t2 = timer()
    writeIndexTime = t2 - t1
    # This will ensure that when destructors of the index is called the internal indexes are deleted too.
    # Be very careful if you are making changes around this.
    cagraIVFPQIndex.thisown = True
    idMapIVFPQIndex.own_fields = True
    del cagraIVFPQIndex
    del idMapIVFPQIndex
    # Let's free up the Vector dataset. We should free up the space to ensure that we can free up some RAM
    vectorsDataset.free_vectors_space()
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
    # This value will come from input keeping it 256 as this is what we have in our benchmarks
    cpuIndex.hnsw.efSearch = 256
    logging.info(f"HNSW value of ef search is {cpuIndex.hnsw.efSearch}")
    # This will ensure that we have faster conversion time, but make the graph immutable
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
