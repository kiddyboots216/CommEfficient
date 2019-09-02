import ray
import time
ray.init(object_store_memory=int(1e8))
import torch
from collections import Counter
import numpy as np
n_clients = 200
n_clients_to_select = 4
@ray.remote(num_gpus=1)
def test(big_vec):
    return big_vec + torch.randn(int(1e5))
x = torch.randn(int(1e5))
x = ray.put(x,
        #weakref=True
        )
#x = ray.put(torch.randn(int(1e5))
       #, weakref=True
        #)
data = [x for _ in range(n_clients)]
start_idx = 0
while True:
    start_idx = start_idx % n_clients
    end_idx = start_idx + n_clients_to_select
    indices = np.random.choice(n_clients, n_clients_to_select, replace=False)
    #indices = np.arange(start_idx, end_idx)
    new_data = [test.remote(data[idx]) for idx in indices]
    for i, idx in enumerate(indices):
        data[idx] = new_data[i]
    start_idx = end_idx


    obj_store = ray.objects()
    c = Counter()
    for objid, info in obj_store.items():
        if "DataSize" in info:
            c[info["DataSize"]] += 1
        else:
            c["nosize"] += 1
    print(c)
    print(f"Object store size: {sum([val['DataSize'] if 'DataSize' in val else 0 for key, val in obj_store.items()])}")
    print()
