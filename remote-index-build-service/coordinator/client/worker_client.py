import concurrent.futures
import logging
from dataclasses import dataclass

from urllib3 import HTTPConnectionPool
import json

from util.common import ThreadSafeRoundRobinIterator

logger = logging.getLogger(__name__)

@dataclass
class Worker:
    host:str
    port: int

@dataclass
class RegisterWorkerRequest:
    workerURL: str
    workerPort: int
    workerProtocol: str = 'http'

    @staticmethod
    def build_register_worker_request(data: dict) -> list['RegisterWorkerRequest']:
        workerList = data.get('workerList', [])
        if not workerList:
            raise ValueError(f"Missing required fields in the input: {data}")
        if not workerList:
            raise ValueError(f"At-least 1 worker should be present: {data}")
        register_worker_request_list = []
        for worker in workerList:
            if not all(key in worker for key in ['workerURL', 'workerPort']):
                raise ValueError(f"Missing required fields in the input: {data}")
            register_worker_request_list.append(RegisterWorkerRequest(
                workerURL=worker['workerURL'],
                workerPort=worker['workerPort'],
                workerProtocol=worker.get('workerProtocol', 'http')
            ))
        return register_worker_request_list

class WorkerClient:

    def __init__(self, worker):
        self.logger = logging.getLogger(__name__)
        self.client_pool = HTTPConnectionPool(host=worker.host, port=worker.port, maxsize=10, timeout=1)
        self.worker = worker

    def get_job(self, job_id: str):
        return self.client_pool.request("GET", f"/_status/{job_id}", headers={'Content-Type': 'application/json'})

    def create_index(self, createIndexRequest):
        self.logger.info(f"createIndexRequest is : {createIndexRequest}")
        response = self.client_pool.request("POST", "/_build", body=json.dumps(createIndexRequest), headers={'Content-Type': 'application/json'})
        if response.status == 200 or response.status == 201:
            return response.json()
        return None

    def get_jobs(self):
        jobs = self.client_pool.request("GET", "/jobs", headers={'Content-Type': 'application/json'})
        jobs = jobs.json()
        self.logger.info(f"Jobs are : {jobs}")
        return jobs

    def heart_beat(self):
        try:
            response = self.client_pool.request(method="GET", url="/_heart_beat", headers={'Content-Type': 'application/json'})
            return response.status == 200
        except Exception as e:
            self.logger.error(f"Error in heart_beat for {self.worker.host}:{self.worker.port} : {e}")
        return False

    def __str__(self):
        return f"WorkerClient(host={self.worker.host}, port={self.worker.port})"

    def __repr__(self):
        return self.__str__()


class WorkerService:

    def __init__(self, workers: list):
        self.round_robin_iterator = None
        self.logger = logging.getLogger(__name__)
        self.workers = workers
        self._build_worker_client()

    def _build_worker_client(self):
        self.worker_clients: list[WorkerClient] = []
        for worker in self.workers:
            self.worker_clients.append(WorkerClient(worker))
        if len(self.worker_clients) == 0:
            self.round_robin_iterator = None
        else:
            self.round_robin_iterator = ThreadSafeRoundRobinIterator(self.worker_clients)


    def get_job(self, job_id: str):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.worker_clients)) as executor:
            futures = []
            for worker_client in self.worker_clients:
                future = executor.submit(worker_client.get_job, job_id)
                future.add_done_callback(lambda x: x.result().release_conn())
                futures.append(future)

            for future in futures:
                response = future.result()
                if response.status == 200:
                    return response.json()
                else:
                    self.logger.info(f"No job found for the {job_id} : {response.status} {response.reason}")
            raise Exception(f"Error in get_job for job_id {job_id}")


    def create_index(self, createIndexRequest):
        worker_client = self.round_robin_iterator.get_next()
        self.logger.debug("in create_index call")
        response = worker_client.create_index(createIndexRequest)
        self.logger.info(f"response is : {response}")
        return response

    def get_jobs(self):
        jobs = {}
        logging.info("in get_jobs call")
        for worker_client in self.worker_clients:
            client_jobs = worker_client.get_jobs()
            for job in client_jobs:
                jobs[job] = client_jobs[job]
        self.logger.info(f"jobs are : {jobs}")
        return jobs

    def register_worker(self, register_worker_request_list: list[RegisterWorkerRequest]):
        self.logger.info(f"register_worker_request is : {register_worker_request_list}")
        for register_worker_request in register_worker_request_list:
            worker = Worker(register_worker_request.workerURL, register_worker_request.workerPort)
            self.worker_clients.append(WorkerClient(worker))
            if self.round_robin_iterator is None:
                self.round_robin_iterator = ThreadSafeRoundRobinIterator(self.worker_clients)
            else:
                self.round_robin_iterator.add_item(WorkerClient(worker))

    def get_all_worker(self):
        worker_list = []
        for worker in self.worker_clients:
            w = {
                "workerURL": worker.worker.host,
                "workerPort": worker.worker.port
            }
            worker_list.append(w)
        return worker_list