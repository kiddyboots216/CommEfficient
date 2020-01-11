# CommEfficient
This branch uses Python multiprocessing in a functional implementation of federated learning. 

It comes with two experimental setups; a ResNet9 on CIFAR10 (`cv_train.py`) and GPT2 on ConvAI (`gpt2_train.py`). 

There are a variety of command-line args which are best examined by looking at `utils.py`

The server is contained in `fed_aggregator.py` and the worker is contained in `fed_worker.py`

Other relevant branches: `dp` contains `dp_functions.py` which provides a hook for computing DP queries for DP-SGD. `attacks` contains an implementation of a malicious model poisoning adversary.

To use sketching, you need to install https://github.com/nikitaivkin/csh
