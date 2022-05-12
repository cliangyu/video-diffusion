"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
NO_MPI = ('NO_MPI' in os.environ)
if not NO_MPI:
    from mpi4py import MPI
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
#GPUS_PER_NODE = 8
GPUS_PER_NODE = th.cuda.device_count()

SETUP_RETRY_COUNT = 3


def setup_dist():
    """
    Setup a distributed process group.
    """
    if NO_MPI:
        return
    if dist.is_initialized():
        return

    backend = "gloo" if not th.cuda.is_available() else "nccl"
    comm = MPI.COMM_WORLD

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    print('setting up world')
    print("RANK", comm.rank)
    print("SIZE", comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device("cuda") if NO_MPI else th.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    def read_data():
        with bf.BlobFile(path, "rb") as f:
            return f.read()
    def load_data(data):
        return th.load(io.BytesIO(data), **kwargs)
    if NO_MPI:
        return load_data(read_data())
    else:
        data = read_data() if MPI.COMM_WORLD.Get_rank() == 0 else None
        data = MPI.COMM_WORLD.bcast(data)
        return load_data(data)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
