# DEPENDENCIES
import networkx as nx


# CLASS
class Routes:
    """ GClass Routes generates a graph with the specifications called to the Graph function.
    It also provides a series of methods in order to compute different properties of the graph. """

    def __init__(self, graph):
        # nodes, edges and weights for graph model
        # self.graph_model, self.fixed_positions = graph
        self.graph_model = graph

        # instantiate an empty graph object
        self.graph = nx.Graph()

        # each edge is a tuple of the form (node1, node2, {'weight': weight})
        self.edges = [(k[0], k[1], {'weight': v}) for k, v in self.graph_model.items()]

        self.graph.add_edges_from(self.edges)

    # METHODS

    def shortest_path(self, source, target, method='dijkstra', verbose=False):
        """
        Parameters:
        GNetworkX graph
        source: node, optional
        Starting node for path. If not specified, compute the shortest paths for each possible starting node.

        target: node, optional
        Ending node for path. If not specified, compute the shortest paths to all possible nodes.

        weight: None, string or function, optional (default = None)
        If None, every edge has weight/distance/cost 1. If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1. If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three positional arguments: the two endpoints of
        an edge and the dictionary of edge attributes for that edge. The function must return a number.

        method: string, optional (default = ‘dijkstra’)
        The algorithm to use to compute the path. Supported options: ‘dijkstra’, ‘bellman-ford’. Other inputs produce a
        ValueError. If weight is None, unweighted graph methods are used, and this suggestion is ignored.

        Returns:
        One posible dhortest path: list or dictionary
        All returned paths include both the source and target in the path.

        If the source and target are both specified, return a single list of nodes in a shortest path from the source
        to the target.

        If only the source is specified, return a dictionary keyed by targets with a list of nodes in a shortest path
        from the source to one of the targets.

        If only the target is specified, return a dictionary keyed by sources with a list of nodes in a shortest path
        from one of the sources to the target.

        If neither the source nor target are specified return a dictionary of dictionaries with
        path[source][target]=[list of nodes in path].

        Raises: NodeNotFound
        If source is not in G.
        ValueError
        """
        if verbose:
            print(f'\nThe shortest path between node {source} and node {target} is:')
        short_path = nx.shortest_path(self.graph, source, target, weight="weight", method=method)
        return short_path

    def all_shortest_paths(self, source, target):
        """ Compute all shortest simple paths in the graph.
        Parameters:
            G: NetworkX graph
            source: node
            Starting node for path.

            target: node
            Ending node for path.

            weight None, string or function, optional (default = None)
            If None, every edge has weight/distance/cost 1. If a string, use this edge attribute as the edge weight.
            Any edge attribute not present defaults to 1.
            If this is a function, the weight of an edge is the value returned by the function. The function must
            accept exactly three positional arguments: the two endpoints of an edge and the dictionary of edge
            attributes for that edge. The function must return a number.

            method string, optional (default = ‘dijkstra’) The algorithm to use to compute the path lengths.
            Supported options: ‘dijkstra’, ‘bellman-ford’. Other inputs produce a ValueError. If weight is None,
            unweighted graph methods are used, and this suggestion is ignored."""
        return list(nx.all_shortest_paths(self.graph, source, target, weight='weight'))

    def all_simple_paths(self, source, target):
        """
        Generate all simple paths in the graph G from source to target.
        A simple path is a path with no repeated nodes.
        Parameters:
                    self.graph (NetworkX graph) –
                    source (node) – Starting node for path
                    target (node) – Ending node for path
                    cutoff (integer, optional) – Depth to stop the search. Only paths of length <= cutoff are returned.
        Returns:
                path_generator – A generator that produces lists of simple paths. If there are no paths between
                the source and target within the given cutoff the generator produces no output.

        Return type:
                    generator
        """
        return list(nx.all_simple_paths(self.graph, source, target))

    def number_of_nodes(self):
        """Returns the number of nodes in the graph."""
        return self.graph.number_of_nodes()

    def all_possible_simple_paths(self, verbose=False):
        """ Returns a list with all possible simple paths from each node to each other node.
        Paths to node itself are filtered """
        total_nodes = list(nx.nodes(self.graph))
        total_nodes.sort()
        num_paths = 0
        paths = []
        for node_source in total_nodes:
            for node_target in total_nodes:
                if node_source != node_target:
                    for path in nx.all_simple_paths(self.graph, node_source, node_target):
                        if verbose:
                            print(f'{num_paths} - From node {node_source} to node {node_target}: {path}')
                        paths.append(path)
                        num_paths += 1
        if verbose:
            print(f'\nTotal number of paths: {num_paths}')
        return paths

    def node_neighbors(self, n):
        """ Returns a dict_key iterator of nodes connected to node n """
        return list(nx.neighbors(self.graph, n))

    def degree(self, nbunch=None, weight=None):
        """Returns a degree view of single node or of nbunch of nodes.
        If nbunch is omitted, then return degrees of *all* nodes.
        """
        return self.graph.degree(nbunch, weight)

    def path_distance(self, path):
        """
        Returns total cost associated with specified path and weight.
        There may be more than one shortest path between a source and target.
        This returns only one of them.

            Parameters:
            Graph
            A NetworkX graph.

            path: list
            A list of node labels which defines the path to traverse

            weight: string
            A string indicating which edge attribute to use for path cost

            Returns:
            cost: int or float
            An integer or a float representing the total cost with respect to the specified weight of the specified path

            Raises:
            NetworkXNoPath
            If the specified edge does not exist.
        """
        return nx.path_weight(self.graph, path, weight='weight')

    def paths_and_distance(self, source, target):
        """ Returns a list of nodes in a shortest path between source and
        target using the A* (“A-star”) algorithm. """
        return nx.astar_path_length(self.graph, source, target)
