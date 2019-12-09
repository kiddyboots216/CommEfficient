# CommEfficient
This branch uses Python multiprocessing in a functional implementation of federated learning. 

It comes with two experimental setups; a ResNet9 on CIFAR10 (`fed_train.py`) and GPT2 on ConvAI (`gpt2_train.py`). 

There are a variety of command-line args which are best examined by looking at `utils.py`

All functions are in `functions.py`

To use sketching, you need to install https://github.com/nikitaivkin/csh

To use DP, you need to install https://github.com/kiddyboots216/pytorch_privacy
