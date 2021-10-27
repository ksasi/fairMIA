import random
import networkx as nx
from networkx.generators.random_graphs import _random_subset
from networkx.utils import py_random_state


# dependancy index
class choiceAttribute():
    def __init__(self, name, choices, weights, d_index):
        self.name = name
        self.choices = choices
        self.weights = weights
        self.d_index = d_index
        self.type = 0


class rangeAttribute():
    def __init__(self, name, start, end, step, weights, d_index):
        self.name = name
        self.start = start
        self.end = end
        self.step = step
        self.weights = weights
        self.d_index = d_index
        self.type = 1


class barabasiAlbertAttributeModel():
    @py_random_state(3)
    def __init__(self, n, m, seed=None, initial_graph=None):
        # self.graph = barabasi_albert_graph(n,m)
        self.n = n
        self.m = m
        self.seed = seed
        self.initial_graph = initial_graph
        self.attributes = []
        if self.m < 1 or self.m >= self.n:
            raise nx.NetworkXError(
                f"Barabási–Albert network must have m >= 1 and m < n, m = {m}, n = {n}"
            )

        if self.initial_graph is None:
            # Default initial graph : star graph on (m + 1) nodes
            self.G = nx.star_graph(self.m)
        else:
            if len(self.initial_graph) < self.m or len(self.initial_graph) > self.n:
                raise nx.NetworkXError(
                    f"Barabási–Albert initial graph needs between m={self.m} and n={self.n} nodes"
                )
            self.G = self.initial_graph.copy()

    def show_attributes(self):
        for attribute in self.attributes:
            print(attribute.name, end=' : ')
            if attribute.type == 0:
                for choice in attribute.choices:
                    print(choice, end=', ')
                print()
            elif attribute.type == 1:
                print(f'values from {attribute.start} to {attribute.end} with stepsize {attribute.step}')

    def add_choice_attribute(self, name, choices, weights, d_index=1):
        self.attributes.append(choiceAttribute(name, choices, weights=weights, d_index=d_index))

    def add_range_attribute(self, name, start, end, step, d_index=1):
        self.attributes.append(rangeAttribute(name, start, end, step, weights=None, d_index=d_index))

    def generate_graph(self):
        graph_attributes = {}
        for attribute in self.attributes:

            choices = attribute.choices if attribute.type==0 else range(attribute.start,attribute.end,attribute.step)

            for node in self.G.nodes():
                if node not in graph_attributes:
                    graph_attributes[node] = {}
                graph_attributes[node][attribute.name] = \
                random.choices(choices, weights=attribute.weights)[0]

            # nx.set_node_attributes(self.G, random.random(), attribute.name)
        # List of existing nodes, with nodes repeated once for each adjacent edge
        repeated_nodes = [n for n, d in self.G.degree() for _ in range(d)]

        # Start adding the other n - m0 nodes.
        source = len(self.G)
        while source < self.n:

            # Now choose m unique nodes from the existing nodes
            # Pick uniformly from repeated_nodes (preferential attachment)

            ################################################################B
            graph_attributes[source] = {}
            repeated_items=[]
            for attribute in self.attributes:
                #random attribute generated
                choices = attribute.choices if attribute.type == 0 else range(attribute.start, attribute.end,
                                                                              attribute.step)
                random_attribute = random.choices(choices, weights=attribute.weights, k=1)[0]
                graph_attributes[source][attribute.name]=random_attribute
                for i in range (attribute.d_index):
                    repeated_items.extend([k for k, v in graph_attributes.items() if v[attribute.name] == random_attribute ])
            repeated_items=list(filter(lambda a: a != source, repeated_items))
            total_items=len(repeated_items)
            repeated_nodes.extend(repeated_items)
            ################################################################A
            targets = _random_subset(repeated_nodes, self.m, self.seed)
            ################################################################B
            repeated_nodes=repeated_nodes[:-total_items]
            ################################################################A
            # Add edges to m nodes from the source.
            self.G.add_edges_from(zip([source] * self.m, targets))
            # Add one node to the list for each new edge just created.
            repeated_nodes.extend(targets)

            # And the new node "source" has m edges to add to the list.
            repeated_nodes.extend([source] * self.m)

            source += 1
        #############################################B
        nx.set_node_attributes(self.G, graph_attributes)
        #############################################A

        return self.G

