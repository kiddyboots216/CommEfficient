import torch
import nose
from fed_aggregator import FedModel, FedOptimizer, FedCriterion, FedAccuracy
def makeSketchers(nWeights, nWorkers, k, r, c, p2, device):
    lr = 0.005
    #model = torch.nn.Linear(nWeights, 1, bias=False).to(device)
    #list(model.parameters())[0].data.zero_()
    # initialize weights to zero
    sketched_params = {'k': k, 'p2': p2, 'num_cols': c,
            'num_rows': r, 'num_blocks': 1, 'momentum': 0.0, 'weight_decay': 0.0,
        'nesterov': False, 'dampening': 0, 'n_clients': nWorkers, 'lr': lr,
        'sketch': True, 'sketch_down': False, 'device': device, 'unit_test': True, 
        'n_clients_per_round': nWorkers, 'virtual_momentum': False, 
        'momentum_sketch': False, 'topk_down': False, 'local_topk': False, 'true_topk': False,
        'local_momentum': False} 
    model_cls = torch.nn.Linear
    model_config = {'in_features': nWeights, 'out_features': 1, 'bias': False}
    fed_model = FedModel(model_cls, model_config, sketched_params)
    opt = torch.optim.SGD(fed_model.parameters(), lr=lr)
    fed_opt = FedOptimizer(opt, sketched_params)
    return fed_model, fed_opt

def makeData(nRows, nDims, device):
    X = torch.arange(nRows * nDims).view(nRows, nDims).float().to(device)
    y = torch.arange(nRows).view(nRows, 1).float().to(device)
    return X, y

def checkW(model, expectedWs):
    w = list(model.parameters())[0]
    inExpected = False
    for expectedW in expectedWs:
        if torch.allclose(w.cpu(), expectedW.cpu()):
            inExpected = True
    msg = "Got w={}, expected one of ("
    msg += ",".join(["{}" for _ in expectedWs])
    assert inExpected, msg.format(w, *expectedWs)

def runTest(nData, nWeights, nWorkers, k, r, c, p2,
            expectedW1s, expectedW2s, device, doSlowSketching):

    model, opt = makeSketchers(nWeights, nWorkers, k, r, c, p2,
                                       device)

    # setting this flag to True uses a faster sketching calculation
    # that would be cheating in the real distributed setting
    #summer._doSlowSketching = doSlowSketching

    X, y = makeData(nData, nWeights, device)
    #minibatch = [X,y]
    idx = [i for i in range(nWorkers)]
    #minibatches = [minibatch for _ in range(nWorkers)]
    minibatches = []
    #batch_size = int(nData/nWorkers)
    batch_size = 4
    for i in range(nWorkers):
        start = i * batch_size // nWorkers
        end = (i+1) * batch_size // nWorkers
        in_batch = X[start:end]
        target_batch = y[start:end]
        minibatch = [in_batch, target_batch]
        minibatches.append(minibatch)
    criterion = torch.nn.MSELoss(reduction='sum')
    fed_criterion = FedCriterion(criterion)
    fake_criterion = FedAccuracy(criterion)
    model.train(True)
    outs, loss, acc, grads = model(minibatches, idx)
    opt.step(grads, idx)
    #loss = summer((y - model(X))**2)

    yield checkW, model, expectedW1s

    opt.zero_grad()
    outs, loss, acc, grads = model(minibatches, idx)
    opt.step(grads, idx)

    if expectedW2s is not None:
        yield checkW, model, expectedW2s

"""
Learning Rate: 0.005
Momentum: 0.9

One Parameter:
    Data:
        (0) => 0
        (1) => 1
        (2) => 2
        (3) => 3

    Model:
        y = wx

    Loss:
        L = \sum_{i=1}^4 (wx - x)^2

    Full Gradient
        dL/dw = 2(w - 1)(x0^2 + x1^2 + x2^2 + x3^2)
              = 28(w-1)

    Mini-Batch Gradient (BS=2)
        dL1/dw = 2(w - 1)(x0^2 + x1^2) = 2(w - 1)
        dL2/dw = 2(w - 1)(x2^2 + x3^2) = 26(w - 1)

    One Worker:
        w0 = 0
        g^1 = u^1 = v^1 = -28
        w1 = 0.14
        g^1 = u^1 = v^1 = -24.08
        w2 = 0.3808
    Two Workers:
        w0 = 0
        g^1 = u^1 = v^1 = -2
        g^2 = u^2 = v^2 = -26
        w1 = 0.14
        g^1 = u^1 = v^1 = -1.72
        g^2 = u^2 = v^2 = -22.36
        w2 = 0.3808

Two Parameters:
    Data:
        (0, 1) => 0
        (2, 3) => 1
        (4, 5) => 2
        (6, 7) => 3

    Model:
        y = w.x, w \in R^2

    Loss:
        L = \sum_{i=0}^3 (w.x_i - i)^2

    Full Gradient
        dL/dw0 = \sum_{i=0}^3 2(w.x_i - i) x_{i,0}
               = 8(14w_0 + 17w_1 - 7)
        dL/dw1 = \sum_{i=0}^3 2(w.x_i - i) x_{i,1}
               = 4(34w_0 + 42w_1 - 17)

    Mini-Batch Gradient (BS=2)
        dL0/dw0 = 8w_0 + 12w_1 - 4
        dL0/dw1 = 12w_0 + 20w_1 - 6
        dL1/dw0 = 4(26w_0 + 31w_1 - 13)
        dL1/dw1 = 2(62w_0 + 74w_1 - 31)

    One Worker, k=2:
        Large Sketch, p2=0:
            w0 = (0, 0)
            g^1 = u^1 = v^1 = (-56, -68)
            w1 = (0.28, 0.34)
            g^1 = u^1 = v^1 = (21.6, 27.2)
            w2 = (0.172, 0.204)
        Sketch is 1x1, p2=0:
            w0 = (0, 0)
            g^1 = u^1 = v^1 = (-56, -68)
            S^1 = [[\pm 124]] or [[\pm 12]]
            w1 = 0.62 or \pm 0.06 (not -0.62)
            too much work to figure out all possible w2...
        Sketch is 1x1, p2=0:
            recover behavior of large sketch
    Two Workers:
        Large Sketch, p2=0:
            k=2:
                w0 = (0, 0)
                g^1 = u^1 = v^1 = (-4, -6)
                g^2 = u^2 = v^2 = (-52, -62)
                w1 = (0.28, 0.34)
                same as before
                w2 = (0.172, 0.204)
            k=1:
                w0 = (0, 0)
                g^1 = u^1 = v^1 = (-4, -6)
                g^2 = u^2 = v^2 = (-52, -62)
                w1 = (0, 0.34)
                u^1 = v^1 = (-4, 0)
                u^2 = v^2 = (-52, 0)
                g^1 = (0.08, 0.8)
                g^2 = (-9.84, -11.68)
                u^1 = (-3.52, 0.8)
                u^2 = (-56.64, -11.68)
                w2 = (-0.3008, 0.34)

"""

testParams = [
    #N, d, W, k, r, c,    p2, expectedW1s,     expectedW2s
    (4, 1, 1, 1, 1, 1,    0,  ([0.14],),       ([0.3808],)),
    #(4, 1, 2, 1, 1, 1,    0,  ([0.14],),       ([0.3808],)),
    #(4, 2, 1, 2, 9, 1000, 0,  ([0.28, 0.34],), ([0.172, 0.204],)),
    #(4, 2, 1, 2, 1, 1,    0,  ([0.62, 0.62],
    #                            [0.06, -0.06],
    #                            [-0.06, 0.06]), None),
    #(4, 2, 1, 2, 1, 1,    1,  ([0.28, 0.34],), ([0.172, 0.204],)),
    #(4, 2, 2, 2, 9, 1000, 0,  ([0.28, 0.34],), ([0.172, 0.204],)),
    #(4, 2, 2, 1, 9, 1000, 0,  ([0, 0.34],), ([-0.3008, 0.34],))
]

def testAll():
    import ray
    ray.init()
    doSlowSketching = False
    for device in ["cpu"]:
        for N, d, W, k, r, c, p2, w1s, w2s in testParams:
            w1s = tuple(torch.tensor(w) for w in w1s)
            if w2s is not None:
                w2s = tuple(torch.tensor(w) for w in w2s)
            for test in runTest(N, d, W, k, r, c, p2, w1s, w2s,
                                device, doSlowSketching):
                yield test

