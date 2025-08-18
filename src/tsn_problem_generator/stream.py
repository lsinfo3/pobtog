from math import inf

from typing import List


PREAMBLE = 8 * 8  # Bit
IPG = 12 * 8  # Bit


class Stream(object):
    LAST_ID = -1

    def __init__(self, label: str, path: List, priority: int, rate: float, burst: int, minFrameSize: int, maxFrameSize: int) -> None:
        """
        :param rate: in bits/s
        :param burst: in bit (including overheads PREAMBLE + IPG)
        :minFrameSize: in bit (excluding overhead)
        :maxFrameSize: in bit (excluding overhead)
        """
        Stream.LAST_ID += 1
        self._id = Stream.LAST_ID
        self._label = label
        self._path = path
        self._priority = priority
        self._rate = rate
        self._burst = burst
        self._minFrameSize = minFrameSize
        self._maxFrameSize = maxFrameSize
        self.init_local_streams()

        if burst < maxFrameSize + PREAMBLE + IPG:
            raise ValueError(f"{burst=} < {maxFrameSize=} + Preamble+IPG ({PREAMBLE+IPG})")

    @property
    def id(self): return self._id

    @property
    def label(self): return self._label

    @property
    def path(self): return self._path

    @property
    def priority(self): return self._priority

    @property
    def rate(self): return self._rate

    @property
    def burst(self): return self._burst

    @property
    def minFrameSize(self): return self._minFrameSize

    @property
    def maxFrameSize(self): return self._maxFrameSize

    def init_local_streams(self):
        self.localStreams = [LocalStream(self, i) for i in range(len(self.path))]

    def to_json_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "path": [self.path[0].n1.name] + [l.n2.name for l in self.path],
            "priority": self.priority,
            "rate": self.rate,
            "burst": self.burst,
            "minFrameSize": self.minFrameSize,
            "maxFrameSize": self.maxFrameSize
        }

    def clone(self):
        return Stream(self._label, self._path, self._priority, self._rate, self._burst, self._minFrameSize, self._maxFrameSize)

    def __key(self):
        return (self._id, self._priority, self._rate, self._burst, self._minFrameSize, self._maxFrameSize, self._path[-1])

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, Stream):
            return self.__key() == other.__key()
        return NotImplemented

    def __repr__(self):
        return f"Stream{self.id}{{label={self.label}, path={self.path[0].n1.name} -> {self.path[-1].n2.name} ({len(self.path)} hops), tspec={self.priority}/{self.burst}/{self.rate}}}"


class LocalStream(object):
    def __init__(self, parent_stream: Stream, pathIndex: int, accMaxLatency: int = inf, accMinLatency: int = 0, accMinLatencyCQF: int = 0, accMaxLatencyCQF: int = 0, maxIdleSlope: float = -1):
        self._s = parent_stream
        self._pathIndex = pathIndex
        self._accMaxLatency = accMaxLatency
        self._accMinLatency = accMinLatency
        self._accMinLatencyCQF = accMinLatencyCQF
        self._accMaxLatencyCQF = accMaxLatencyCQF
        self._maxIdleSlope = maxIdleSlope

    @property
    def s(self): return self._s

    @property
    def pathIndex(self): return self._pathIndex

    @property
    def accMaxLatency(self): return self._accMaxLatency

    @property
    def accMinLatency(self): return self._accMinLatency

    @property
    def accMaxLatencyCQF(self): return self._accMaxLatencyCQF

    @property
    def accMinLatencyCQF(self): return self._accMinLatencyCQF

    @property
    def maxIdleSlope(self): return self._maxIdleSlope

    @property
    def link(self): return self.s.path[self.pathIndex]

    @property
    def prev(self): return self.s.localStreams[self.pathIndex-1] if self.pathIndex > 0 else None

    @property
    def prevIngress(self): return self.prev.link.ingressPortN2 if self.pathIndex > 0 else None

    @property
    def id(self): return self._s.id

    @property
    def label(self): return self._s.label

    @property
    def path(self): return self._s.path

    @property
    def priority(self): return self._s.priority

    @property
    def cqf_prio(self): return self._s.cqf_prio

    @property
    def rate(self): return self._s.rate

    @property
    def burst(self): return self._s.burst

    @property
    def minFrameSize(self): return self._s.minFrameSize

    @property
    def maxFrameSize(self): return self._s.maxFrameSize

    def __key(self):
        return (self._s, self._pathIndex, self._accMaxLatency, self._accMinLatency)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, LocalStream):
            return self.__key() == other.__key()
        return NotImplemented

    def __repr__(self):
        return f"LocalStream{self.id}{{label={self.label}, pathIndex={self.pathIndex}, accMaxLatency={self.accMaxLatency}, accMinLatency={self.accMinLatency}}}"

