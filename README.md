# CommEfficient
This repo uses PyTorch's native distributed module in an implementation of federated learning.

It comes with a few experimental setups; various Residual Networks on CIFAR10, CIFAR100, FEMNIST, ImageNet (`cv_train.py`) and GPT2 on PersonaChat (`gpt2_train.py`). 

There are a variety of command-line args which are best examined by looking at `utils.py`

The server is contained in `fed_aggregator.py` and the worker is contained in `fed_worker.py`

Other relevant branches: `attacks` contains an implementation of a malicious model poisoning adversary.

To use sketching, you need to install https://github.com/nikitaivkin/csh
