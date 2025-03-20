import getopt
import sys
from benchmarking.workload.workload import runWorkload
from benchmarking.data_types.data_types import IndexTypes, WorkloadTypes
import config

def main(argv):
    opts, args = getopt.getopt(argv, "", ["workload=", "index_type=", "workload_type=", "run_id=", "help"])
    workloadName = "all"
    indexType = "all"
    workloadType = WorkloadTypes.INDEX_AND_SEARCH

    for opt, arg in opts:
        if opt == '--help':
            print('--workload <dataset file name>')
            indexTypeValues = IndexTypes.list()
            indexTypeValues.append("all")
            print(f'--index_type(optional) should have a value {indexTypeValues}, default is {indexType}')
            print(f'--workload_type(optional) should have a value {WorkloadTypes.list()} default is : {WorkloadTypes.INDEX_AND_SEARCH.value}')
            print(f'--run_id(optional) can have any value, default is : {config.run_id}')
            sys.exit()
        elif opt in "--workload":
            workloadName = arg
        elif opt == '--index_type':
            indexType = arg
        elif opt == "--workload_type":
            workloadType = WorkloadTypes.from_str(arg)
        elif opt == "--run_id":
            config.run_id = arg

    runWorkload(workloadName, indexType, workloadType)


if __name__ == "__main__":
    main(sys.argv[1:])
