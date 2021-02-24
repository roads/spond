# Utilities
import os
import socket
import torch

# Must be done before any other use of torch.
# Sets device to be CPU if the hostname is pals.ucl.ac.uk
# and limits the CPUs being used to 25%
def setup_torch():

    hostname = socket.gethostname()

    if hostname.endswith("pals.ucl.ac.uk"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
        torch.set_num_threads(int(os.cpu_count() / 4))
    else:
        # Detect if GPUs are available
        GPU = torch.cuda.is_available()

        # If you have a problem with your GPU, set this to "cpu" manually
        device = torch.device("cuda:0" if GPU else "cpu")
    return device
