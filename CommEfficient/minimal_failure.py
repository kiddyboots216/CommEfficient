"""
import ray
import torch
import time
from collections import Counter
"""
import time
import ray
"""
import numpy as np
ray.init(object_store_memory=int(250e6))

@ray.remote
def identity(vectors):
    return [ray.put(ray.get(vec)) for vec in vectors]

vectors = [ray.put(np.zeros(int(1e5), dtype=np.int32)) for _ in range(200)]
i = 0
while True:
    vectors = ray.get(identity.remote(vectors))
    i += 1
    print(i)
    print(ray.worker.global_worker.plasma_client.debug_string())
"""
FREE_DELAY_S = 10.0
MAX_FREE_QUEUE_SIZE = 100
_last_free_time = 0.0
_to_free = []
def ray_free(object_ids):
    """Call ray.get and then queue the object ids for deletion.
    This function should be used whenever possible in RLlib, to optimize
    memory usage. The only exception is when an object_id is shared among
    multiple readers.
    Args:
        object_ids (ObjectID|List[ObjectID]): Object ids to fetch and free.
    Returns:
        The result of ray.get(object_ids).
    """

    global _last_free_time
    global _to_free

    if type(object_ids) is not list:
        object_ids = [object_ids]
    _to_free.extend(object_ids)

    # batch calls to free to reduce overheads
    now = time.time()
    if (len(_to_free) > MAX_FREE_QUEUE_SIZE
            or now - _last_free_time > FREE_DELAY_S):
        ray.internal.free(_to_free)
        _to_free = []
        _last_free_time = now

def ray_get_and_free(object_ids):
    """Call ray.get and then queue the object ids for deletion.
    This function should be used whenever possible in RLlib, to optimize
    memory usage. The only exception is when an object_id is shared among
    multiple readers.
    Args:
        object_ids (ObjectID|List[ObjectID]): Object ids to fetch and free.
    Returns:
        The result of ray.get(object_ids).
    """

    global _last_free_time
    global _to_free

    result = ray.get(object_ids)
    if type(object_ids) is not list:
        object_ids = [object_ids]
    _to_free.extend(object_ids)

    # batch calls to free to reduce overheads
    now = time.time()
    if (len(_to_free) > MAX_FREE_QUEUE_SIZE
            or now - _last_free_time > FREE_DELAY_S):
        ray.internal.free(_to_free)
        _to_free = []
        _last_free_time = now

    return result
"""
@ray.remote
def update(vector):
    return ray_get_and_free(vector)

@ray.remote
def identity(vectors):
    vecs = ray_get_and_free(vectors)
    return [ray.put(vec) for vec in vecs]

if __name__=="__main__":
    ray.init(object_store_memory=int(200e6))
    obj_id = ray.put(torch.randn(int(1e5)))
    vectors = [obj_id for _ in range(200)]
    while True:
        vectors = [update.remote([vec]) for vec in vectors]
        #vectors = ray.get(identity.remote(vectors))
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
"""
