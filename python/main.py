import getopt
import sys
from python.workload.workload import runWorkload
from python.data_types.data_types import IndexTypes


def main(argv):
    opts, args = getopt.getopt(argv, "", ["workload=", "index_type="])
    workloadName = None
    indexType = None
    for opt, arg in opts:
        if opt == '-h':
            print('--dataset_file <dataset file path>')
            print(f'--index_type should have a value {IndexTypes}')
            sys.exit()
        elif opt in "--workload":
            workloadName = arg
        elif opt == '--index_type':
            indexType = IndexTypes.from_str(arg)
    print(indexType.value)
    runWorkload(workloadName, indexType)


if __name__ == "__main__":
    main(sys.argv[1:])
