from flask import Flask, jsonify, request
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
from waitress import serve
from client.worker_client import WorkerService, Worker, RegisterWorkerRequest
import traceback
import os
import math
import threading
import time

from util.common import is_dev_env

# Create logger
logger = logging.getLogger()
if is_dev_env():
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Console handler (stdout)
console_handler = logging.StreamHandler()
if is_dev_env():
    console_handler.setLevel(logging.DEBUG)
else:
    console_handler.setLevel(logging.INFO)

# File handler
file_handler = RotatingFileHandler(
    '/app/logs/remote-index-build-service.log',
    maxBytes=1024 * 1024,
    backupCount=5
)
if is_dev_env():
    file_handler.setLevel(logging.DEBUG)
else:
    file_handler.setLevel(logging.INFO)

# Format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

logger = logging.getLogger(__name__)

app = Flask(__name__)

def get_worker_from_seed_file(seed_file):
    with open(seed_file, 'r') as f:
        data = json.load(f)
        workers = []
        for worker in data:
            workers.append(Worker(worker['host'], worker['port']))
        logging.info(f"Workers are: {workers}")
        return workers

if is_dev_env():
    logger.info("Running in dev mode")
    workers = get_worker_from_seed_file("workers_seed.json")
else:
    logger.info("Running in prod mode, workers will be added by user")
    workers = []

workerservice = WorkerService(workers=workers)

def heart_beat(workerService):
    logger.info(f"Running the heart beat thread")
    show_message = True
    while True:
        worker_clients = workerService.worker_clients
        round_robin_iterator = workerService.round_robin_iterator
        if show_message:
            logger.info(f"Workers before heart beat is : {worker_clients} {round_robin_iterator}")
            show_message = False
        for worker_client in worker_clients[:]:
            try:
                if show_message:
                    logger.info(f"Checking client: {worker_client.worker.host}, port: {worker_client.worker.port}")
                if not worker_client.heart_beat():
                    show_message = True
                    if round_robin_iterator is not None:
                        logger.info(f"Removing client: {worker_client.worker.host}, port: {worker_client.worker.port}")
                        if round_robin_iterator.has_item(worker_client):
                            round_robin_iterator.remove_item(worker_client)
                            logger.info(f"Removed client: {worker_client.worker.host}, port: {worker_client.worker.port}")
                        else:
                            logger.info(f"iterator round_robin_iterator doesn't have: {worker_client.worker.host}, port: {worker_client.worker.port}")

                    else:
                        logger.info(f"iterator round_robin_iterator was null: {worker_client.worker.host}, port: {worker_client.worker.port}")

                    if worker_client in worker_clients:
                        worker_clients.remove(worker_client)

            except Exception as e:
                logger.error(f"Error in heart beat: {e}")
        if show_message:
            logger.info(f"Workers after heart beat is : {worker_clients}, {round_robin_iterator}")
        # Sleeping for 5 sec
        time.sleep(5)


def startup_task():
    with app.app_context():
        logger.info("Server is up and running, executing startup task")
        # Your startup code here
        logger.info(f"Thread Heartbeat")
        global workerservice
        # Create thread with arguments
        thread = threading.Thread(target=heart_beat, args=(workerservice,))
        thread.start()


@app.route('/')
def hello():
    return jsonify({
        "message": "Hello from Vector Index Build Service Coordinator!",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/_build', methods=['POST'])
def create_index():
    try:
        input = request.json
        response = workerservice.create_index(input)
        logger.info(f"Response is: {response}")
        return json.dumps(response, default=lambda o: o.__dict__, indent=4)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/jobs', methods=['GET'])
def get_jobs():
    try:
        return workerservice.get_jobs()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/_status/<string:job_id>')
def job(job_id: str):
    try:
        jobs = workerservice.get_job(job_id)
        return json.dumps(jobs, default=lambda o: o.__dict__, indent=4), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/register_worker', methods=['POST'])
def register_worker():
    try:
        logger.info(f"Received request: %s ", request.json)
        register_worker_request = RegisterWorkerRequest.build_register_worker_request(request.json)
        workerservice.register_worker(register_worker_request)
    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Invalid request {request.json}"}), 400
    return jsonify({"message": "Worker registered successfully"}), 201

@app.route('/workers', methods=['GET'])
def get_all_worker():
    workers_list = {
        "workerList": workerservice.get_all_worker()
    }
    return json.dumps(workers_list, default=lambda o: o.__dict__, indent=4), 200


if __name__ == '__main__':
    if is_dev_env():
        logger.info("Running in dev mode so not going to start the heart beat process")
    else:
        logger.info("Starting heart beat process")
        threading.Thread(target=startup_task).start()
    serve(app, host="0.0.0.0", port=6006, threads=max(math.floor(os.cpu_count() * 4), 4))