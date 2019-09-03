import ray
import torch
ray.init(object_store_memory=int(100e6))

@ray.remote
def identity(vectors):
    return [ray.put(ray.get(vec)) for vec in vectors]

obj_id = ray.put(torch.randn(int(1e5)))
vectors = [obj_id for _ in range(200)]
while True:
    vectors = ray.get(identity.remote(vectors))
