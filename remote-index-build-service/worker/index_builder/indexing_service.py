import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, Any
import threading
import logging

from index_builder.vector_index_builder import build_index_and_upload_index
from models.data_model import CreateIndexRequest, CreateIndexResponse

logger = logging.getLogger(__name__)
# Keeping number of threads 1 here to ensure that we are not creating more than 1 indexing job
executor = ThreadPoolExecutor(max_workers=6)

@dataclass
class JobDetails:
    id: str
    status: str
    error: str = None
    result: Dict[str, Any] = None
    request: CreateIndexRequest = None

class IndexingService:
    def __init__(self):
        self.jobs: Dict[str, JobDetails] = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str, create_index_request: CreateIndexRequest) -> JobDetails:
        with self._lock:
            job = JobDetails(id=job_id, status="submitted", request= create_index_request)
            self.jobs[job_id] = job
            return job

    def update_job_status(self, job_id: str, **kwargs):
        with self._lock:
            if job_id in self.jobs:
                for key, value in kwargs.items():
                    setattr(self.jobs[job_id], key, value)

    def get_job_status(self, job_id: str) -> JobDetails:
        job = self.jobs.get(job_id)
        return job

    def get_jobs(self) -> Dict[str, JobDetails]:
        return self.jobs

    def start_job(self, job_id:str, create_index_request):
        # submit the job
        executor.submit(self._run_job, job_id, create_index_request)
        logger.info(f"Job started {job_id}")

    def _run_job(self, job_id, create_index_request:CreateIndexRequest):
        try:
            logger.info(f"Starting index creation for job {job_id}")
            self.update_job_status(
                job_id,
                status="running"
            )
            # create the index
            graph_file, stats = build_index_and_upload_index(create_index_request)

            result = CreateIndexResponse(bucketName=create_index_request.bucketName, graphFileLocation=graph_file, stats=stats)

            self.update_job_status(
                job_id,
                status="completed",
                result=result
            )
            logger.info(f"Index creation completed for job {job_id}")

        except Exception as e:
            logger.error(f"Error creating index for job {job_id}: {str(e)}")
            logger.error(traceback.format_exc())
            self.update_job_status(
                job_id,
                status="failed",
                error=str(e)
            )