import csv
import getopt
import json
import logging
import os
import sys
from benchmarking.data_types.data_types import IndexTypes, WorkloadTypes
from benchmarking.utils.common_utils import ensureDir, formatTimingMetricsValue, readAllWorkloads

logging.basicConfig(level=logging.INFO)

def persistMetricsAsCSV(workloadType: WorkloadTypes, allMetrics: dict, workloadName: str, indexType: IndexTypes):
    ensureDir(f"results/{workloadName}")
    fields = ["workload-name", "indexType", "dataset-name", "dimensions", "vectors-count", "queries-count", "indexing-params", "index-creation-time", "gpu-to-cpu-index-conversion-time", "write-to-file-time", "write-index-time", "total-build-time", "search-parameter", "search-time", "unit", "search-throughput", "recall@100", "recall@1"]
    rows = []
    if workloadType == WorkloadTypes.INDEX:
        logging.error("This type of workload is not supported for writing data in csv")
        sys.exit()
    else:
        workloadDetails = allMetrics[workloadName]["workload-details"]
        searchParamItr = 0
        indexingParamItr = 0
        for searchMetric in allMetrics[workloadName]["searchMetrics"]:
            searchParamItr = searchParamItr + 1
            searchTimingMetrics = searchMetric["search-timing-metrics"]
            row = {
                "workload-name": workloadName,
                "indexType": indexType.value,
                "dataset-name": workloadDetails["dataset_name"],
                "dimensions": workloadDetails["dimension"],
                "queries-count": workloadDetails.get("queriesCount"),
                "vectors-count": workloadDetails.get("vectorsCount"),
                "indexing-params": searchMetric["indexing-params"],
                "search-time": formatTimingMetricsValue(searchTimingMetrics['searchTime']),
                "unit": searchTimingMetrics["units"],
                "recall@100": searchTimingMetrics["recall_at_100"],
                "recall@1": searchTimingMetrics["recall_at_1"],
                "search-parameter": searchMetric["search-params"],
                "search-throughput": searchTimingMetrics["search_throughput"],
            }

            if allMetrics[workloadName].get("indexingMetrics") is not None:
                row["vectors-count"] = workloadDetails["vectorsCount"]
                indexingMetrics = allMetrics[workloadName]["indexingMetrics"]
                row["index-creation-time"] = formatTimingMetricsValue(indexingMetrics[indexingParamItr]["indexing-timingMetrics"]["indexTime"])
                row["write-index-time"] = formatTimingMetricsValue(indexingMetrics[indexingParamItr]["indexing-timingMetrics"]["writeIndexTime"])
                row["gpu-to-cpu-index-conversion-time"] = formatTimingMetricsValue(indexingMetrics[indexingParamItr]["indexing-timingMetrics"].get("gpu_to_cpu_index_conversion_time"))
                row["write-to-file-time"] = formatTimingMetricsValue(indexingMetrics[indexingParamItr]["indexing-timingMetrics"].get("write_to_file_time"))
                row["total-build-time"] = formatTimingMetricsValue(indexingMetrics[indexingParamItr]["indexing-timingMetrics"]["totalTime"])
                if searchParamItr % len(workloadDetails["search-parameters"]) == 0:
                    indexingParamItr = indexingParamItr + 1
            rows.append(row)
            

    with open(f"results/{workloadName}/{workloadType.value}_{indexType.value}.csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        # writing headers (field names)
        writer.writeheader()
        # writing data rows
        writer.writerows(rows)
    
    logging.info(f"Results are stored at location: results/{workloadName}/{workloadType.value}_{indexType.value}.csv")
    return f"results/{workloadName}/{workloadType.value}_{indexType.value}.csv"


def writeDataInCSV(workloadName:str, indexType:str, workloadType:WorkloadTypes):
    if workloadType == WorkloadTypes.INDEX:
        logging.error("This type of workload is not supported for writing data in csv")
        sys.exit()
    
    indexTypesList = []

    if indexType == "all":
        indexTypesList = IndexTypes.enumList()
    else:
        indexTypesList.append(IndexTypes.from_str(indexType))

    workloadCSVFiles = []
    for indexTypeEnum in indexTypesList:
        if workloadName == "all":
            allWorkloads = readAllWorkloads()
            
            for currentWorkloadName in allWorkloads[indexTypeEnum.value]:
                csvFile = writeDataInCSVPerWorkload(currentWorkloadName, indexTypeEnum, workloadType)
                if csvFile is not None:
                    workloadCSVFiles.append(csvFile)
        else:
            csvFile = writeDataInCSVPerWorkload(workloadName, indexTypeEnum, workloadType)
            if csvFile is not None:
                workloadCSVFiles.append(csvFile)

    writeDataInSingleCSVFile(workloadCSVFiles, "all_results.csv")

def writeDataInCSVPerWorkload(workloadName:str, indexType:IndexTypes, workloadType:WorkloadTypes)-> str:
    jsonFile = f"results/{workloadName}/{workloadType.value}_{indexType.value}.json"
    if os.path.exists(jsonFile) is False:
        logging.warn(f"No result file exist for {workloadName} , indexType: {indexType.value} workloadType {workloadType.value} at {jsonFile}")
        return None

    f = open(jsonFile)
    allMetrics = json.load(f)
    f.close
    return persistMetricsAsCSV(workloadType, allMetrics, workloadName, indexType)

def writeDataInSingleCSVFile(workloadCSVFiles: list, outfileName:str):
    if len(workloadCSVFiles) == 0:
        logging.warn("No CSV files to combine to a single result file")
        return
    ensureDir("results/all/")

    if os.path.exists(f"results/all/{outfileName}"):
        logging.info(f"Deleting the file results/all/{outfileName}, as it exist")
        os.remove(f"results/all/{outfileName}")


    outputFile = open(f"results/all/{outfileName}",'w')
    # This will add header and other all the data from first file in output file.
    with open(workloadCSVFiles[0]) as f:
        logging.info(f"Writing file: {workloadCSVFiles[0]}")
        for line in f:
            outputFile.write(line)
    
    # Now we call add all other files by skipping their headers
    for resultFiles in workloadCSVFiles[1:]:
        logging.info(f"Writing file: {resultFiles}")
        with open(resultFiles) as f:
            next(f)
            for line in f:
                outputFile.write(line)
    logging.info(f"All data is written in the file results/all/{outfileName}")


def main(argv):
    opts, args = getopt.getopt(argv, "", ["workload=", "index_type=", "workload_type=", "h"])
    workloadName = "all"
    indexType = "all"
    workloadType = WorkloadTypes.INDEX_AND_SEARCH
    for opt, arg in opts:
        if opt == '--h':
            print('--dataset_file <dataset file path>')
            print(f'--index_type should have a value {IndexTypes.list()}')
            print(f'--workload_type(optional) should have a value {WorkloadTypes.list()} default is : {WorkloadTypes.INDEX_AND_SEARCH.value}')
            sys.exit()
        elif opt in "--workload":
            workloadName = arg
        elif opt == '--index_type':
            indexType = arg
        elif opt == "--workload_type":
            workloadType = WorkloadTypes.from_str(arg)

    writeDataInCSV(workloadName, indexType, workloadType)

if __name__ == "__main__":
    main(sys.argv[1:])