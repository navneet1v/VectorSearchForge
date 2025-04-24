import struct
import os
from numpy import float32
import numpy as np
import time

from abc import ABC, ABCMeta, abstractmethod
from enum import Enum

import h5py

from typing import cast

import logging

import faiss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/files/vector_search.log'),
        logging.StreamHandler()
    ]
)

class Context(Enum):
    """DataSet context enum. Can be used to add additional context for how a
    data-set should be interpreted.
    """
    INDEX = 1
    QUERY = 2
    NEIGHBORS = 3
    CUSTOM = 4


class DataSet(ABC):
    """DataSet interface. Used for reading data-sets from files.

    Methods:
        read: Read a chunk of data from the data-set
        size: Gets the number of items in the data-set
        reset: Resets internal state of data-set to beginning
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def read(self, chunk_size: int):
        pass

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def reset(self):
        pass


class HDF5DataSet(DataSet):
    """ Data-set format corresponding to `ANN Benchmarks
    <https://github.com/erikbern/ann-benchmarks#data-sets>`_
    """

    def __init__(self, dataset_path: str, context: Context, custom_context=None):
        file = h5py.File(dataset_path)
        self.data = cast(h5py.Dataset, file[self._parse_context(context, custom_context)])
        self.current = 0

    def read(self, chunk_size: int):
        if self.current >= self.size():
            return None

        end_i = self.current + chunk_size
        if end_i > self.size():
            end_i = self.size()

        v = cast(np.ndarray, self.data[self.current:end_i])
        self.current = end_i
        return v

    def size(self):
        return self.data.len()

    def reset(self):
        self.current = 0

    @staticmethod
    def _parse_context(context: Context, custom_context=None) -> str:
        if context == Context.NEIGHBORS:
            return "neighbors"

        if context == Context.INDEX:
            return "train"

        if context == Context.QUERY:
            return "test"

        if context == Context.CUSTOM:
            return custom_context

        raise Exception("Unsupported context")


class BigANNNeighborDataSet(DataSet):
    """ Data-set format for neighbor data-sets for `Big ANN Benchmarks
    <https://big-ann-benchmarks.com/index.html#bench-datasets>`_"""

    def __init__(self, dataset_path: str):
        self.file = open(dataset_path, 'rb')
        self.file.seek(0, os.SEEK_END)
        num_bytes = self.file.tell()
        self.file.seek(0)

        if num_bytes < 8:
            raise Exception("File is invalid")

        self.num_queries = int.from_bytes(self.file.read(4), "little")
        self.k = int.from_bytes(self.file.read(4), "little")

        # According to the website, the number of bytes that will follow will
        # be:  num_queries X K x sizeof(uint32_t) bytes + num_queries X K x
        # sizeof(float)
        if (num_bytes - 8) != 2 * (self.num_queries * self.k * 4):
            raise Exception("File is invalid")

        self.current = 0

    def read(self, chunk_size: int):
        if self.current >= self.size():
            return None

        end_i = self.current + chunk_size
        if end_i > self.size():
            end_i = self.size()

        v = [[int.from_bytes(self.file.read(4), "little") for _ in
              range(self.k)] for _ in range(end_i - self.current)]

        self.current = end_i
        return v

    def size(self):
        return self.num_queries

    def reset(self):
        self.file.seek(8)
        self.current = 0

import logging
import os
import shutil
from urllib.request import urlretrieve

import numpy as np
import bz2
import sys

def recall_at_r(results, neighbor_dataset:HDF5DataSet, r, k, query_count):
    """
    Calculates the recall@R for a set of queries against a ground truth nearest
    neighbor set
    Args:
        results: 2D list containing ids of results returned by OpenSearch.
        results[i][j] i refers to query, j refers to
            result in the query
        neighbor_dataset: 2D dataset containing ids of the true nearest
        neighbors for a set of queries
        r: number of top results to check if they are in the ground truth k-NN
        set.
        k: k value for the query
        query_count: number of queries
    Returns:
        Recall at R
    """
    correct = 0.0
    total_num_of_results = 0
    for query in range(query_count):
        true_neighbors = neighbor_dataset.read(1)
        if true_neighbors is None:
            break
        true_neighbors_set = set(true_neighbors[0][:k])
        true_neighbors_set.discard(-1)
        min_r = min(r, len(true_neighbors_set))
        total_num_of_results += min_r
        for j in range(min_r):
            if results[query][j] in true_neighbors_set:
                correct += 1.0
    neighbor_dataset.reset()
    return correct / total_num_of_results

def get_omp_num_threads():
    import math
    return max(math.floor(os.cpu_count()/4), 1)

def ensureDir(dirPath:str) -> str:
    os.makedirs(f"/tmp/files/{dirPath}", exist_ok=True)
    return f"/tmp/files/{dirPath}"



def downloadDataSetForWorkload(workloadToExecute: dict) -> str:
    download_url = workloadToExecute["download_url"]
    dataset_name = workloadToExecute["dataset_name"]
    isCompressed = False if workloadToExecute.get("compressed") is None else True
    compressionType = workloadToExecute.get("compression-type")
    return downloadDataSet(download_url, dataset_name, isCompressed, compressionType)


def downloadDataSet(download_url: str, dataset_name: str, isCompressed:bool, compressionType:str | None) -> str:
    logging.info("Downloading dataset...")
    destination_path_compressed = None
    dir_path = ensureDir("dataset")
    if compressionType is not None:
        destination_path_compressed = os.path.join(dir_path, f"{dataset_name}.hdf5.{compressionType}")
    destination_path = os.path.join(dir_path, f"{dataset_name}.hdf5")

    if not os.path.exists(destination_path):
        if isCompressed:
            logging.info(f"downloading {download_url} -> {destination_path_compressed} ...")
            urlretrieve(download_url, destination_path_compressed)
            decompress_dataset(destination_path_compressed, compressionType, destination_path)
        else:
            logging.info(f"downloading {download_url} -> {destination_path} ...")
            urlretrieve(download_url, destination_path)
        logging.info(f"downloaded {download_url} -> {destination_path}...")
    return destination_path


def decompress_dataset(filePath:str, compressionType:str, outputFile:str):
    logging.info(f"Decompression {filePath} having compression type: {compressionType}")
    if compressionType == "bz2":
        with bz2.BZ2File(filePath) as fr, open(outputFile,"wb") as fw:
            shutil.copyfileobj(fr, fw, length = 1024 * 1024 * 10)  # read by 100MB chunks
        logging.info("Completed decompression... ")
    else:
        logging.error(f"Compression type : {compressionType} is not supported for decompression")
        sys.exit()



def prepare_indexing_dataset(datasetFile: str, normalize: bool = None, docToRead:int = -1) -> tuple[int, np.ndarray, list]:
    logging.info(f"Reading data set from file: {datasetFile}")
    index_dataset: HDF5DataSet = HDF5DataSet(datasetFile, Context.INDEX)

    logging.info(f"Total number of docs that we will read for indexing: {index_dataset.size() if docToRead == -1 else docToRead}")
    xb: np.ndarray = index_dataset.read(index_dataset.size() if docToRead == -1 or docToRead is None else docToRead).astype(dtype = np.float32)
    d: int = len(xb[0])
    ids = [i for i in range(len(xb))]
    if normalize:
        logging.info("Doing normalization...")
        xb = xb / np.linalg.norm(xb)
        logging.info("Completed normalization...")

    logging.info("Dataset info : ")
    logging.info(f"Dimensions: {d}")
    logging.info(f"Total Vectors: {len(xb)}")
    logging.info(f"Total Ids: {len(ids)}")
    logging.info(f"Normalized: {normalize}")

    return d, xb, ids


def prepare_search_dataset(datasetFile: str, normalize: bool = None) -> tuple[int, np.ndarray, HDF5DataSet]:
    logging.info(f"Reading data set from file: {datasetFile}")
    search_dataset: HDF5DataSet = HDF5DataSet(datasetFile, Context.QUERY)
    xq: np.ndarray = search_dataset.read(search_dataset.size()).astype(dtype= np.float32)
    gt:HDF5DataSet = HDF5DataSet(datasetFile, Context.NEIGHBORS)
    d: int = len(xq[0])
    logging.info("Dataset info : ")
    logging.info(f"Dimensions: {d}")
    logging.info(f"Total Vectors: {len(xq)}")
    logging.info(f"Normalized: {normalize}")
    if normalize:
        logging.info("Doing normalization...")
        xq = xq / np.linalg.norm(xq)
        logging.info("Completed normalization...")
    return d, xq, gt

def custom_excepthook(exc_type, exc_value, exc_traceback):
    try:
        # First log the original error
        logging.error("An unhandled exception occurred:", 
                     exc_info=(exc_type, exc_value, exc_traceback))
        
        # If working with CUDA/GPU operations, log memory info
        try:
            import cupy as cp
            mem_info = cp.get_default_memory_pool().used_bytes()
            logging.info(f"GPU Memory Usage: {mem_info / 1024**2:.2f} MB")
        except:
            pass

    except Exception as hook_error:
        # If our custom handler fails, fall back to the default handler
        print("Error occurred in exception handler:")
        print(hook_error)
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Install the custom exception handler
sys.excepthook = custom_excepthook

def indexAndSearchUsingCuvs():
    from cuvs.neighbors import cagra
    import cupy as cp

    workloads = [
        {
            "download_url": "https://huggingface.co/datasets/navneet1v/datasets/resolve/main/coherev2-dbpedia.hdf5?download=true",
            "dataset_name": "coherev2-dbpedia",
            "normalize": False
        },
        {
            "download_url": "https://huggingface.co/datasets/navneet1v/datasets/resolve/main/FlickrImagesTextQueries.hdf5?download=true",
            "dataset_name": "FlickrImagesTextQueries",
            "normalize": True
        },
        {
            "download_url": "https://huggingface.co/datasets/navneet1v/datasets/resolve/main/marco_tasb.hdf5?download=true",
            "dataset_name": "marco_tasb",
            "normalize": False
        },
        {
            "download_url": "https://dbyiw3u3rf9yr.cloudfront.net/corpora/vectorsearch/cohere-wikipedia-22-12-en-embeddings/documents-1m.hdf5.bz2",
            "dataset_name": "cohere-768-ip",
            "compressed": True,
            "compression-type": "bz2",
            "normalize": False
        }
    ]

    for workload in workloads:

        logging.info(f"Running for workload {workload['dataset_name']}")
        file = downloadDataSetForWorkload(workload)
        d, xb, ids = prepare_indexing_dataset(file, workload["normalize"])
        index_params = cagra.IndexParams(intermediate_graph_degree=64,graph_degree=32,build_algo='ivf_pq', metric="inner_product")

        index = cagra.build(index_params, xb)

        d, xq, gt = prepare_search_dataset(file, workload["normalize"])

        xq = cp.asarray(xq)

        search_params = cagra.SearchParams(itopk_size = 200)
        distances, neighbors = cagra.search(search_params, index, xq, 100)

        logging.info("Search is done")
        neighbors = cp.asnumpy(neighbors)

        logging.info(f"Recall at k=100 is : {recall_at_r(neighbors, gt, 100, 100, len(xq))}")
        logging.info("Sleeping for 5 seconds")
        time.sleep(5)


def indexAndSearchUsingFaiss(file, indexingParams={}):
    d, xb, ids = prepare_indexing_dataset(file)
    res = faiss.StandardGpuResources()
    metric = faiss.METRIC_L2
    cagraIndexConfig = faiss.GpuIndexCagraConfig()
    cagraIndexConfig.intermediate_graph_degree = 64 if indexingParams.get('intermediate_graph_degree') is None else indexingParams['intermediate_graph_degree']
    cagraIndexConfig.graph_degree = 32 if indexingParams.get('graph_degree') == None else indexingParams['graph_degree']
    cagraIndexConfig.device = faiss.get_num_gpus() - 1
    cagraIndexConfig.store_dataset = False
    cagraIndexConfig.refine_rate = 2.0 if indexingParams.get('refine_rate') == None else indexingParams.get('refine_rate')
    import math
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
    
    cagraIndexConfig.build_algo = faiss.graph_build_algo_IVF_PQ
    print("Creating GPU Index.. with IVF_PQ")
    cagraIVFPQIndex = faiss.GpuIndexCagra(res, d, metric, cagraIndexConfig)
    idMapIVFPQIndex = faiss.IndexIDMap(cagraIVFPQIndex)

    idMapIVFPQIndex.add_with_ids(xb, ids)

    cpuIndex = faiss.IndexHNSWCagra()
    cpuIndex.base_level_only = True

    cagraIVFPQIndex.copyTo(cpuIndex)
    idMapIVFPQIndex.index = cpuIndex

    graph_file = "/tmp/files/open-ai.graph"
    faiss.write_index(idMapIVFPQIndex, graph_file)
    del xb
    del idMapIVFPQIndex
    import gc
    gc.collect()
    cagraHNSWIndex:faiss.IndexIDMap = faiss.read_index(graph_file)
    cagraHNSWIndex.index.base_level_only = True
    
    hnswParameters = faiss.SearchParametersHNSW()
    hnswParameters.efSearch = 256

    def search(q, k, params):
        D, ids = cagraHNSWIndex.search(x=q, k=k, params=params)
        return ids

    d, xq, gt = prepare_search_dataset(file)
    I = []
    from tqdm import tqdm
    for query in tqdm(xq, total=len(xq), desc=f"Running queries for ef_search: {hnswParameters.efSearch}"):
        result = search(np.array([query]), 100, hnswParameters)
        I.append(result[0])
    recall_at_k = recall_at_r(I, gt, 100, 100, len(xq))
    
    logging.info(f"Recall at 100 using faiss : is {recall_at_k}")

if __name__ == "__main__":
    try:
        # workloadToExecute = {
        #     "download_url": "https://huggingface.co/datasets/navneet1v/datasets/resolve/main/open-ai-1536-temp.hdf5?download=true",
        #     "dataset_name": "open-ai-1536"
        # }
        # file = downloadDataSetForWorkload(workloadToExecute)

        # indexAndSearchUsingFaiss(file)
        indexAndSearchUsingCuvs()
    except Exception as e:
        logging.error("Error in main execution:", exc_info=True)
    
    