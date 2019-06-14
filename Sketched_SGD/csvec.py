import math
import numpy as np
import copy
import torch

LARGEPRIME = 2**61-1

cache = {}

#import line_profiler
#import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

class CSVec(object):
    """ Simple Count Sketched Vector """
    def __init__(self, d, c, r, doInitialize=True, device=None,
                 nChunks=1, numBlocks=1):
        global cache

        self.r = r # num of rows
        self.c = c # num of columns
        # need int() here b/c annoying np returning np.int64...
        self.d = int(d) # vector dimensionality
        # how much to chunk up (on the GPU) any computation
        # that requires computing something along all tokens. Doing
        # so saves GPU RAM at the cost of having to transfer the chunks
        # of self.buckets and self.signs between host & device
        self.nChunks = nChunks

        # reduce memory consumption of signs & buckets by constraining
        # them to be repetitions of a single block
        self.numBlocks = numBlocks

#         if device is None:
#             device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         else:
#         assert("cuda" in device or device == "cpu")
        self.device = device
#         print(f"CSVec is using backend {self.device}")

        if not doInitialize:
            return

        # initialize the sketch
        self.table = torch.zeros((self.r, self.c), device=self.device)
#         print(f"Making table of dim{self.r,self.c} which is {self.table}")

        # if we already have these, don't do the same computation
        # again (wasting memory storing the same data several times)
        if (d, c, r) in cache:
            hashes = cache[(d, c, r)]["hashes"]
            self.signs = cache[(d, c, r)]["signs"]
            self.buckets = cache[(d, c, r)]["buckets"]
            if self.numBlocks > 1:
                self.blockSigns = cache[(d, c, r)]["blockSigns"]
                self.blockOffsets = cache[(d, c, r)]["blockOffsets"]
            return

        # initialize hashing functions for each row:
        # 2 random numbers for bucket hashes + 4 random numbers for
        # sign hashes
        # maintain existing random state so we don't mess with
        # the main module trying to set the random seed but still
        # get reproducible hashes for the same value of r

        # do all these computations on the CPU, since pytorch
        # is incapable of in-place mod, and without that, this
        # computation uses up too much GPU RAM
        rand_state = torch.random.get_rng_state()
        torch.random.manual_seed(42)
        hashes = torch.randint(0, LARGEPRIME, (r, 6),
                                    dtype=torch.int64, device="cpu")

        if self.numBlocks > 1:
            nTokens = self.d // numBlocks
            if self.d % numBlocks != 0:
                # so that we only need numBlocks repetitions
                nTokens += 1
            self.blockSigns = torch.randint(0, 2, size=(self.numBlocks,),
                                            device=self.device) * 2 - 1
            self.blockOffsets = torch.randint(0, self.c,
                                              size=(self.numBlocks,),
                                              device=self.device)
        else:
            assert(numBlocks == 1)
            nTokens = self.d

        torch.random.set_rng_state(rand_state)

        tokens = torch.arange(nTokens, dtype=torch.int64, device="cpu")
        tokens = tokens.reshape((1, nTokens))

        # computing sign hashes (4 wise independence)
        h1 = hashes[:,2:3]
        h2 = hashes[:,3:4]
        h3 = hashes[:,4:5]
        h4 = hashes[:,5:6]
        self.signs = (((h1 * tokens + h2) * tokens + h3) * tokens + h4)
        self.signs = ((self.signs % LARGEPRIME % 2) * 2 - 1).float()
        if self.nChunks == 1:
            # only move to device now, since this computation takes too
            # much memory if done on the GPU, and it can't be done
            # in-place because pytorch (1.0.1) has no in-place modulo
            # function that works on large numbers
            self.signs = self.signs.to(self.device)

        # computing bucket hashes  (2-wise independence)
        h1 = hashes[:,0:1]
        h2 = hashes[:,1:2]
        self.buckets = ((h1 * tokens) + h2) % LARGEPRIME % self.c
        if self.nChunks == 1:
            # only move to device now. See comment above.
            # can't cast this to int, unfortunately, since we index with
            # this below, and pytorch only lets us index with long
            # tensors
            self.buckets = self.buckets.to(self.device)

        cache[(d, c, r)] = {"hashes": hashes,
                            "signs": self.signs,
                            "buckets": self.buckets}
        if numBlocks > 1:
            cache[(d, c, r)].update({"blockSigns": self.blockSigns,
                                     "blockOffsets": self.blockOffsets})

    def zero(self):
        self.table.zero_()

    def __deepcopy__(self, memodict={}):
        # don't initialize new CSVec, since that will calculate bc,
        # which is slow, even though we can just copy it over
        # directly without recomputing it
        newCSVec = CSVec(d=self.d, c=self.c, r=self.r,
                         doInitialize=False, device=self.device,
                         nChunks=self.nChunks, numBlocks=self.numBlocks)
        newCSVec.table = copy.deepcopy(self.table)
        global cache
        cachedVals = cache[(self.d, self.c, self.r)]
        newCSVec.hashes = cachedVals["hashes"]
        newCSVec.signs = cachedVals["signs"]
        newCSVec.buckets = cachedVals["buckets"]
        if self.numBlocks > 1:
            newCSVec.blockSigns = cachedVals["blockSigns"]
            newCSVec.blockOffsets = cachedVals["blockOffsets"]
        return newCSVec

    def __add__(self, other):
        # a bit roundabout in order to avoid initializing a new CSVec
        returnCSVec = copy.deepcopy(self)
        returnCSVec += other
        return returnCSVec

    def __iadd__(self, other):
        if isinstance(other, CSVec):
            self.accumulateCSVec(other)
        elif isinstance(other, torch.Tensor):
#             self.accumulateTable(other)
            self.accumulateVec(other)
        else:
#             from IPython.core.debugger import set_trace; set_trace()
            raise ValueError(f"Can't add this to a CSVec: {other} because it is not a {CSVec}")
        return self

    def accumulateVec(self, vec):
        # updating the sketch
        try:
            assert(len(vec.size()) == 1 and vec.size()[0] == self.d), f"Len of {vec} was {len(vec.size())} instead of 1 or size was {vec.size()[0]} instead of {self.d}"
        except AssertionError:
            return self.accumulateTable(vec)
#             vec = torch.squeeze(vec, 0)
#             assert(len(vec.size()) == 1 and vec.size()[0] == self.d), f"After squeeze, Len was {len(vec.size())} instead of 1 or size was {vec.size()[0]} instead of {self.d}"
        for r in range(self.r):
            buckets = self.buckets[r,:].to(self.device)
            signs = self.signs[r,:].to(self.device)
            for blockId in range(self.numBlocks):
                start = blockId * buckets.size()[0]
                end = (blockId + 1) * buckets.size()[0]
                end = min(end, self.d)
                offsetBuckets = buckets[:end-start].clone()
                offsetSigns = signs[:end-start].clone()
                if self.numBlocks > 1:
                    offsetBuckets += self.blockOffsets[blockId]
                    offsetBuckets %= self.c
                    offsetSigns *= self.blockSigns[blockId]
                self.table[r,:] += torch.bincount(
                                    input=offsetBuckets,
                                    weights=offsetSigns * vec[start:end],
                                    minlength=self.c
                                   )
                #self.table[r,:] += torch.ones(self.c).to(self.device)

        """
        for i in range(self.nChunks):
            start = int(i / self.nChunks * self.d)
            end = int((i + 1) / self.nChunks * self.d)
            # this will be idempotent if nChunks == 1
            buckets = self.buckets[:,start:end].to(self.device)
            signs = self.signs[:,start:end].to(self.device)
            for r in range(self.r):
                self.table[r,:] += torch.bincount(
                                    input=buckets[r,:],
                                    weights=signs[r,:] * vec[start:end],
                                    minlength=self.c
                                   )
                #self.table[r,:] += torch.ones(self.c)
                #pass
        """
    def accumulateTable(self, table):
        assert self.table.size() == table.size(), f"This CSVec is {self.table.size()} but the table is {table.size()}"
        self.table += table
    
    def accumulateCSVec(self, csVec):
        # merges csh sketch into self
        assert(self.d == csVec.d)
        assert(self.c == csVec.c)
        assert(self.r == csVec.r)
        self.table += csVec.table

    #@profile
    def _findHHK(self, k):
        #return torch.arange(k).to(self.device), torch.arange(k).to(self.device).float()
        assert(k is not None)
        #tokens = torch.arange(self.d, device=self.device)
        #vals = self._findValues(tokens)
        vals = self._findAllValues()
        #vals = torch.arange(self.d).to(self.device).float()
        # sort is faster than torch.topk...
        HHs = torch.sort(vals**2)[1][-k:]
        #HHs = torch.topk(vals**2, k, sorted=False)[1]
        return HHs, vals[HHs]

    def _findHHThr(self, thr):
        assert(thr is not None)
        # to figure out which items are heavy hitters, check whether
        # self.table exceeds thr (in magnitude) in at least r/2 of
        # the rows. These elements are exactly those for which the median
        # exceeds thr, but computing the median is expensive, so only
        # calculate it after we identify which ones are heavy
        tablefiltered = (  (self.table >  thr).float()
                         - (self.table < -thr).float())
        est = torch.zeros(self.d, device=self.device)
        for r in range(self.r):
            est += tablefiltered[r,self.buckets[r,:]] * self.signs[r,:]
        est = (  (est >=  math.ceil(self.r/2.)).float()
               - (est <= -math.ceil(self.r/2.)).float())

        # HHs - heavy coordinates
        HHs = torch.nonzero(est)
        return HHs, self._findValues(HHs)

    def _findValues(self, coords):
        # estimating frequency of input coordinates
        assert(self.numBlocks == 1)
        chunks = []
        d = coords.size()[0]
        if self.nChunks == 1:
            vals = torch.zeros(self.r, self.d, device=self.device)
            for r in range(self.r):
                vals[r] = (self.table[r, self.buckets[r, coords]]
                           * self.signs[r, coords])
            return vals.median(dim=0)[0]

        # if we get here, nChunks > 1
        for i in range(self.nChunks):
            vals = torch.zeros(self.r, d // self.nChunks,
                               device=self.device)
            start = int(i / self.nChunks * d)
            end = int((i + 1) / self.nChunks * d)
            buckets = self.buckets[:,coords[start:end]].to(self.device)
            signs = self.signs[:,coords[start:end]].to(self.device)
            for r in range(self.r):
                vals[r] = self.table[r, buckets[r, :]] * signs[r, :]
            # take the median over rows in the sketch
            chunks.append(vals.median(dim=0)[0])

        vals = torch.cat(chunks, dim=0)
        return vals

    def _findAllValues(self):
#         from IPython.core.debugger import set_trace; set_trace()
        if self.nChunks == 1:
            if self.numBlocks == 1:
                vals = torch.zeros(self.r, self.d, device=self.device)
                for r in range(self.r):
                    vals[r] = (self.table[r, self.buckets[r,:]]
                               * self.signs[r,:])
#                 print(f"Table of size {self.r, self.d} is {self.table}")
                return vals.median(dim=0)[0]
            else:
                medians = torch.zeros(self.d, device=self.device)
                #ipdb.set_trace()
                for blockId in range(self.numBlocks):
                    start = blockId * self.buckets.size()[1]
                    end = (blockId + 1) * self.buckets.size()[1]
                    end = min(end, self.d)
                    vals = torch.zeros(self.r, end-start, device=self.device)
                    for r in range(self.r):
                        buckets = self.buckets[r, :end-start]
                        signs = self.signs[r, :end-start]
                        offsetBuckets = buckets + self.blockOffsets[blockId]
                        offsetBuckets %= self.c
                        offsetSigns = signs * self.blockSigns[blockId]
                        vals[r] = (self.table[r, offsetBuckets]
                                    * offsetSigns)
                    medians[start:end] = vals.median(dim=0)[0]
                return medians

    def findHHs(self, k=None, thr=None):
        assert((k is None) != (thr is None))
        if k is not None:
            return self._findHHK(k)
        else:
            return self._findHHThr(thr)

    def unSketch(self, k=None, epsilon=None):
        # either epsilon or k might be specified
        # (but not both). Act accordingly
        if epsilon is None:
            thr = None
        else:
            thr = epsilon * self.l2estimate()

        hhs = self.findHHs(k=k, thr=thr)

        if k is not None:
            assert(len(hhs[1]) == k), f"Should have found {k} hhs but only found {len(hhs[1])}"
        if epsilon is not None:
            assert((hhs[1] < thr).sum() == 0)

        # the unsketched vector is 0 everywhere except for HH
        # coordinates, which are set to the HH values
        unSketched = torch.zeros(self.d, device=self.device)
        unSketched[hhs[0]] = hhs[1]
        return unSketched

    def l2estimate(self):
        # l2 norm esimation from the sketch
        return np.sqrt(torch.median(torch.sum(self.table**2,1)).item())