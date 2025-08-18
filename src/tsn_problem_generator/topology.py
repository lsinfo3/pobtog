from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy import cumsum
from queue import Queue
from typing import Dict, List, Tuple, Iterable, Set
from shapely.geometry.point import Point

import geopop.tsn_problem_generator.stream as s


@dataclass(eq=True, unsafe_hash=True, order=True)
class Node(object):
    name: str = field(hash=True)
    type: str = field(hash=True, compare=False)
    location: Point = field(default= Point(0,0), hash=True, compare=False)
    neighs: List[Link] = field(default_factory=list, compare=False, hash=False, repr=False)
    lastUsedPort: int = field(default=-1, compare=False, repr=False)
    joinPoint: bool = field(default=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if "-" in self.name: raise ValueError("Node name may not contain '-'")
        valid_types = ("switch", "host", "controller", "sensor")
        if self.type not in valid_types:
            raise ValueError("Node type must be one of %s, not %s" % (valid_types, self.type))

    def addNeigh(self, n2: Node, bw: float) -> None:
        if n2 not in [l.get_other(self) for l in self.neighs]:
            inport = self.setAndGetNextPort()
            outport = n2.setAndGetNextPort()
            self.neighs.append(Link(self, n2, bw, inport, outport))
            n2.neighs.append(Link(n2, self, bw, outport, inport))

    def setAndGetNextPort(self) -> int:
        self.lastUsedPort += 1
        return self.lastUsedPort

    def numNeighorSwitches(self) -> int:
        return len([l for l in self.neighs if l.get_other(self).type == "switch"])

    def to_json_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.type,
            "location": (self.location.x, self.location.y)
        }
    @classmethod
    def from_json_dict(cls, data: dict) -> Node:
        """
        Reconstruct a Node from a JSON dictionary.
        """
        name = data["name"]
        type = data["type"]
        location = Point(data["location"])
        return cls(name=name, type=type, location=location)
    @property
    def is_host(self) -> bool:
        return self.type in ["host", "controller", "sensor"]


def Switch(name: str, location= None) -> Node:
    return Node(name, "switch", location or Point(0,0))

def Host(name: str, location= None) -> Node:
    return Node(name, "host",location or Point(0,0))

def Controller(name: str, location= None) -> Node:
    return Node(name, "controller",location or Point(0,0))

def Sensor(name: str, location= None) -> Node:
    return Node(name, "sensor",location or Point(0,0))


@dataclass(eq=True, frozen=True, order=True)
class Link(object):
    n1: Node = field(hash=True)
    n2: Node = field(hash=True)
    bandwidth: float = field(hash=True)
    egressPortN1: int = field(hash=True, repr=False)
    ingressPortN2: int = field(hash=True, repr=False)

    @property
    def name(self) -> str:
        return self.n1.name + "-" + self.n2.name

    def get_other(self, n: Node) -> Node:
        return self.n2 if n == self.n1 else self.n1

    def mirror(self):
        return Link(self.n2, self.n1, self.bandwidth, self.ingressPortN2, self.egressPortN1)

    def __repr__(self):
        if self.bandwidth > 0:
            return f"{self.name}({self.short_bw})"
        return self.name

    @property
    def short_bw(self) -> str:
        if self.bandwidth >= 1e9:
            return f"{round(self.bandwidth / 1e9)}G"
        if self.bandwidth >= 1e6:
            return f"{round(self.bandwidth / 1e6)}M"
        if self.bandwidth >= 1e3:
            return f"{round(self.bandwidth / 1e3)}K"
        return f"{round(self.bandwidth)}"

    def to_json_dict(self) -> dict:
        return {
            "n1": self.n1.name,
            "n2": self.n2.name,
            "bandwidth": self.bandwidth
        }
    
    @classmethod
    def from_json_dict(cls, data: dict, nodes: List[Node]) -> Link:
        """
        Reconstruct a Link from a JSON dictionary.
        
        The `nodes` argument is a mapping of node names to Node objects.
        """
        n1 = next((node for node in nodes if node.name == data["n1"]), None)
        n2 = next((node for node in nodes if node.name == data["n2"]), None)
   
        if n1 is None or n2 is None:
            raise ValueError(f"Node with name {data['n1'] if n1 is None else data['n2']} not found.")
        bandwidth = data["bandwidth"]
        # Ports are not part of the JSON; initialize them to -1 as placeholders.
        return cls(n1=n1, n2=n2, bandwidth=bandwidth, egressPortN1=-1, ingressPortN2=-1)



class Topology(object):
    def __init__(self, max_delays: Dict[Link, Tuple] = None, max_bandwidths: Dict[Link, Tuple] = None, max_queues: Dict[Link, Tuple] = None) -> None:
        self.nodes: List[Node] = []
        self.streams_per_link: Dict[Link, Dict[int, s.LocalStream]] = {}
        self.max_delays: Dict[Link, Tuple] = max_delays
        """
        The max_delays (per_hop_guarantees) are defined per link and per priority.
        
        max_delays := {linkname -> (d0, d1, d2, d3, d4, d5, d6, d7)}
        
        Do not adjust this variable directly. Use update_guarantees() instead.
        """
        self.max_bandwidths: Dict[Link, Tuple] = max_bandwidths
        """
        The max_bandwidths (= max_idle_slops) are defined per link and per priority.

        max_bandwidths := {linkname -> (r0, r1, r2, r3, r4, r5, r6, r7)}
        """
        self.max_queue_sizes: Dict[Link, Tuple] = max_queues
        """
        The max_queue_sizes are defined per link and per priority.
        
        max_queue_sizes := {linkname -> (q0, q1, q2, q3, q4, q5, q6, q7)}
        """

    @property
    def links(self) -> Iterable[Link]:
        all_neighs = [n.neighs for n in self.nodes]
        return set().union(*all_neighs)

    @property
    def hosts(self) -> List[Node]:
        return [n for n in self.nodes if n.type in {"host", "controller", "sensor"}]

    @property
    def controllers(self) -> List[Node]:
        return [n for n in self.nodes if n.type == "controller"]

    @property
    def sensors(self) -> List[Node]:
        return [n for n in self.nodes if n.type == "sensor"]

    @property
    def switches(self) -> List[Node]:
        return [n for n in self.nodes if n.type == "switch"]

    @property
    def joinPoints(self) -> List[Node]:
        return [n for n in self.nodes if n.joinPoint]

    def getDanglingSwitches(self) -> List[Node]:
        return [n for n in self.nodes if n.type == "switch" and n.numNeighorSwitches() == 1]

    def reset_with_prefix(self, prefix: str) -> Topology:
        # Clear everything that might use the has of nodes internally
        self.streams_per_link.clear()
        if self.max_delays: self.max_delays.clear()
        if self.max_bandwidths: self.max_bandwidths.clear()
        if self.max_queue_sizes: self.max_queue_sizes.clear()

        for node in self.nodes:
            node.name = prefix + node.name

        return self

    def to_json_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "links": list(self.links),
            "streams": sorted(list(self.get_all_streams()), key=lambda x: x.id)
        }
    
    @classmethod
    def from_json_dict(cls, json_dict: dict) -> Topology:
        topology = cls()
        
        # Add nodes
        node_dict = {}
        nodes = [Node.from_json_dict(json_node) for json_node in json_dict["nodes"]]
        for n in nodes:
            topology.add_node(n)
        
        # Add links 
        links = [Link.from_json_dict(json_link, topology.nodes) for json_link in json_dict["links"]]
        for l in links:
            topology.create_and_add_links(l.n1,l.n2, l.bandwidth)
 
        

        # Add streams here, since they require the topology and links
        streams = []
        for stream_data in json_dict["streams"]:
            # Reconstruct the path using links and nodes
            path_node_names = stream_data["path"]
            path = []
            for i in range(len(path_node_names) - 1):
                n1_name = path_node_names[i]
                n2_name = path_node_names[i + 1]
                n1 = next((node for node in topology.nodes if node.name == n1_name), None)
                n2 = next((node for node in topology.nodes if node.name == n2_name), None)
                if n1 is None or n2 is None:
                    raise ValueError(f"Node with name {n1_name if n1 is None else n2_name} not found.")
                link= topology.get_link(n1,n2)
                path.append(link)
    
            # Create the Stream object
            stream = s.Stream(
                label=stream_data["label"],
                path=path,
                priority=stream_data["priority"],
                rate=stream_data["rate"],
                burst=stream_data["burst"],
                minFrameSize=stream_data["minFrameSize"],
                maxFrameSize=stream_data["maxFrameSize"]
            )
            streams.append(stream)
        topology.add_streams(streams)

        return topology


    def add_node(self, n: Node) -> Node:
        if n not in self.nodes:
            self.nodes.append(n)
        else:
            raise ValueError(f"Node {n.name} is already part of this topology.")
        return n

    def create_and_add_links(self, n1: Node, n2: Node, bandwidth: float) -> Node:
        """
        :param bandwidth: in Bit/s

        returns n2
        """
        if n1 not in self.nodes:
            self.add_node(n1)
        if n2 not in self.nodes:
            self.add_node(n2)
        n1.addNeigh(n2, bandwidth)
        #return (n1.neighs[-1], n2.neighs[-1])
        return n2

    def get_link(self, n1: Node, n2: Node) -> Link:
        for i,l in enumerate(n1.neighs):
            if l.n2 == n2:
                return l
        raise ValueError("link '%s-%s' not found" % (n1.name, n2.name))

    def get_link_by_name(self, linkname: str) -> Link:
        for l in self.links:
            if l.name == linkname:
                return l
        raise ValueError(f"link '{linkname}' not found")

    def add_stream(self, stream: s.Stream) -> None:
        for i, link in enumerate(stream.path):
            if link not in self.streams_per_link:
                self.streams_per_link[link] = {}
            self.streams_per_link[link][stream.id] = stream.localStreams[i]

        if self.max_delays != None:
            self.update_acc_latencies(stream)

        if self.max_bandwidths != None:
            self.update_idle_slope_stream(stream)

    def add_streams(self, streams: Iterable[s.Stream]) -> None:
        for s in streams:
            self.add_stream(s)

    def remove_stream(self, stream: s.Stream) -> None:
        for link in stream.path:
            del self.streams_per_link[link][stream.id]

    def remove_all_streams(self) -> None:
        self.streams_per_link = {}

    def get_streams_of_link(self, link: Link) -> Iterable[s.LocalStream]:
        return self.streams_per_link.get(link, {}).values()

    def get_all_streams(self) -> Set[s.Stream]:
        all_sets = [{x.s for x in innerdict.values()} for innerdict in self.streams_per_link.values()]
        return set().union(*all_sets)

    def update_acc_latencies(self, stream: s.Stream) -> None:
        accMaxLatencies = cumsum([self.max_delays[link][stream.priority] for link in stream.path])
        accMinLatencies = cumsum([stream.minFrameSize / (link.bandwidth / 1e9) for link in stream.path])

        accMaxLatencies = np.insert(accMaxLatencies, 0, 0)
        accMinLatencies = np.insert(accMinLatencies, 0, 0)

        for i, link in enumerate(stream.path):
            stream.localStreams[i]._accMaxLatency = accMaxLatencies[i]
            stream.localStreams[i]._accMinLatency = accMinLatencies[i]

    def update_guarantees_dict(self, guarantees_dict: Dict[Link, Tuple]) -> None:
        if not self.max_delays:
            self.max_delays = {}
        for link, tuple in guarantees_dict.items():
            self.max_delays[link] = tuple
        for stream in self.get_all_streams():
            self.update_acc_latencies(stream)

    def update_guarantees_all_links(self, link_guarantees: Tuple) -> None:
        guarantees_dict = {}
        for link in self.links:
            guarantees_dict[link] = link_guarantees
        self.update_guarantees_dict(guarantees_dict)

    def update_idle_slopes_all_links(self, max_idle_slopes: Tuple) -> None:
        self.max_bandwidths = {}
        for link in self.links:
            self.max_bandwidths[link] = max_idle_slopes
        for stream in self.get_all_streams():
            self.update_idle_slope_stream(stream)

    def update_idle_slope_stream(self, stream: s.Stream) -> None:
        for i, link in enumerate(stream.path):
            stream.localStreams[i]._maxIdleSlope = self.max_bandwidths[link][stream.priority]

    def update_queue_sizes_all_links(self, max_queue_sizes: Tuple) -> None:
        self.max_queue_sizes = {}
        for link in self.links:
            self.max_queue_sizes[link] = max_queue_sizes

    def shortest_path(self, n1: Node, n2: Node) -> List[Link]:
        pi = {}
        pi[n1] = None

        q = Queue()
        q.put(n1)

        while not q.empty():
            node = q.get()

            if node == n2:
                path = []
                while node != None:
                    path.insert(0, node)
                    node = pi[node]
                return self.nodes_to_links(path)

            for link in node.neighs:
                neigh = link.get_other(node)
                if neigh not in pi:
                    q.put(neigh)
                    pi[neigh] = node

        raise ValueError("No path from %s to %s exists" % (n1.name, n2.name))

    def nodes_to_links(self, nodelist: List[Node]) -> List[Link]:
        linklist: List[Link] = [None] * (len(nodelist) - 1)
        for i in range(1, len(nodelist)):
            n1: Node = nodelist[i-1]
            n2: Node = nodelist[i]

            link = None
            for l in n1.neighs:
                if l.n2 == n2:
                    link = l
                    break
            if link == None: raise ValueError("link '%s-%s' not found" % (n1.name, n2.name))
            linklist[i-1] = link

        return linklist

    def get_other_devices_within_distance(self, start_node: Node, min_dist: int, max_dist: int) -> List[Node]:
        devices = []
        seen = set()
        q = Queue()

        q.put((start_node, 0))  # origin, distance

        while not q.empty():
            origin, inbound_dist = q.get()
            if inbound_dist < max_dist:
                for link in origin.neighs:
                    neigh = link.get_other(origin)
                    if neigh != start_node and neigh not in seen:
                        seen.add(neigh)
                        if inbound_dist + 1 >= min_dist:
                            devices.append(neigh)
                        q.put((neigh, inbound_dist + 1))
        return devices
