# DEPENDENCIES
import copy
import re
import random
import gym
import numpy as np
import pandas as pd
from gym import spaces
from graph import Routes
from config import (
    MAX_STEPS,
    SEED,
    VNF_BW,
    VNF_GPU,
    VNF_RAM,
    VNF_RTT,
)

# GLOBALS
VERBOSE = False  # For debug purposes


# FUNCTIONS
def color_array(arr, color, *lst, **kwargs):
    """Takes the same keyword arguments as np.array2string()."""
    # adjust some kwarg name differences from _make_options_dict()
    names = {"max_line_width": "linewidth", "suppress_small": "suppress"}
    options_kwargs = {names.get(k, k): v for k, v in kwargs.items() if v is not None}
    overrides = np.core.arrayprint._make_options_dict(**options_kwargs)
    options = np.get_printoptions()
    options.update(overrides)
    format_function = np.core.arrayprint._get_format_function(arr, **options)

    # convert input index lists to tuples
    target_indices = set(map(tuple, lst))

    def color_formatter(i):
        # convert flat index to coordinates
        idx = np.unravel_index(i, arr.shape)
        s = format_function(arr[idx])
        if idx in target_indices:
            return EscapeString(f"\033[{34 + color}m{s}\033[0m")
        return s

    # array of indices into the flat array
    indices = np.arange(arr.size).reshape(arr.shape)
    kwargs["formatter"] = {"int": color_formatter}
    return np.array2string(indices, **kwargs)


# CLASS

class EscapeString(str):
    """A string that excludes SGR escape sequences from its length."""

    def __len__(self):
        return len(re.sub(r"\033\[[\d:;]*m", "", self))

    def __add__(self, other):
        return EscapeString(super().__add__(other))

    def __radd__(self, other):
        return EscapeString(str.__add__(other, self))


class Environment(gym.Env):
    """Custom Environment that follows gym interface and
    creates an environment to which the agent can interact with."""
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 scenario,
                 edges_cost,
                 network,
                 fec_coverage,
                 state_features,
                 no_up_nodes,
                 no_down_nodes,
                 no_left_nodes,
                 no_right_nodes,
                 nodes,
                 route,
                 vehicles_num):

        super().__init__()

        # Var
        self.fec_ram_status = None
        self.fec_gpu_status = None
        self.i = None
        self.network = None
        self.visited_node = None
        self.shortest_paths = None
        self.route_buffer = None
        self.vehicles = None
        self.counter_steps = None
        self.old_current_state = None
        self.agent_info = None
        self.observation = None
        self.vnf = None
        self.assigned_route = route
        self.nodes = nodes
        self.vehicles_num = vehicles_num
        self.default_network = network
        self.no_up_nodes = no_up_nodes
        self.no_down_nodes = no_down_nodes
        self.no_left_nodes = no_left_nodes
        self.no_right_nodes = no_right_nodes
        self.scenario = scenario
        self.edges_cost = edges_cost
        self.fec_coverage = fec_coverage
        self.obs_features = state_features
        # Max steps for the agent in the env
        self.max_agent_steps = MAX_STEPS
        # Action space
        self.actions = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
        self.action_space = spaces.Discrete(len(self.actions))
        # Observation space
        self.observation_space = spaces.Box(low=-1,
                                            high=2048,
                                            shape=np.array((len(self.obs_features),)),
                                            dtype=np.int_)
        # All scenario routes
        self.routes = Routes(self.edges_cost)

        # Seed for random generators
        if SEED is not None:
            random.seed(SEED)

    # METHODS
    def generate_vnf(self, background_vehicle_vnf=False):
        """
        Generates VNF demands with random value choices from a determined range.

        Args:
            background_vehicle_vnf (bool, optional): Flag indicating whether it's a background vehicle VNF.
                                                     Defaults to False.
        Returns:
            dict: A dictionary representing the VNF demands.
        """
        if not background_vehicle_vnf:
            if self.assigned_route == 'random':
                routes = [[1, 16], [9, 12], [4, 13]]
                source, target = random.choice(routes)

            elif self.assigned_route == "any_route":
                source, target = random.sample(self.nodes, 2)

            else:
                source, target = self.assigned_route
        else:
            source, target = random.sample(self.nodes, 2)

        vnf = {
            'vnf_source': source,
            'vnf_target': target,
            'vnf_gpu': VNF_GPU,
            'vnf_ram': VNF_RAM,
            'vnf_bw': VNF_BW,
            'vnf_rtt': VNF_RTT
        }

        return vnf

    def close(self):
        del self

    def render(self, mode='human'):
        scenario = self.scenario
        coord_row = np.where(self.scenario == self.agent_info['current_node'])[0]
        coord_col = np.where(self.scenario == self.agent_info['current_node'])[1]
        print()
        print(color_array(scenario, 1, [int(coord_row), int(coord_col)]))

    def reward_function(self, movement_status, is_terminal):
        if movement_status == 'No_mov':
            return -100, False, {'Invalid action': True, 'Edge cost': 0, 'hop': 0}

        if movement_status == 'No_resources':
            return -100, True, {'No resources': True, 'Edge cost': 0, 'hop': 0}

        if movement_status == 'Max_step_reach':
            return -100, True, {'TimeStepsLimit': True, 'Edge cost': 0, 'hop': 0}

        if movement_status == 'Yes_mov':
            current_edge = None
            edge_cost = None

            for edge, cost in self.edges_cost.items():
                if self.old_current_state in edge and self.agent_info['current_node'] in edge:
                    current_edge = edge
                    edge_cost = cost
                    break

            if current_edge is None:
                raise ValueError("Current edge not found.")

            self.visited_node.append(self.agent_info['current_node'])
            visited_count = self.visited_node.count(self.agent_info['current_node'])

            if visited_count > 1:
                reward = -visited_count * edge_cost
            else:
                reward = -edge_cost

            done = False
            info = {'Moving': True, 'Edge cost': edge_cost, 'hop': 1}

            if is_terminal:
                extra_reward = 0

                if self.route_buffer in self.routes.all_shortest_paths(self.vnf['vnf_source'], self.vnf['vnf_target']):
                    extra_reward = 100

                reward = 50 + extra_reward + edge_cost
                done = True
                info = {'Is terminal': True, 'Edge cost': edge_cost, 'hop': 1}

            return reward, done, info

    def is_terminal(self):
        """
        Checks if the current node is the target node.

        Returns:
            bool: True if the current node is the target node, False otherwise.
        """
        return self.agent_info['current_node'] == self.vnf['vnf_target']

    def agent_move(self, flag):
        """ Computes the next node position for the agent action and checks the
        availability of resources to the corresponding FEC. If resources are available
        the agent moves to the next node."""
        # Get coordinates for vnf node source

        coord_row, coord_col = np.where(self.scenario == self.agent_info['current_node'])

        if flag == 'up':
            # Agent moves, hence new current state (row, col)
            new_current_state = self.scenario[int(coord_row) - 1, int(coord_col)]
        elif flag == 'down':
            new_current_state = self.scenario[int(coord_row) + 1, int(coord_col)]
        elif flag == 'left':
            new_current_state = self.scenario[int(coord_row), int(coord_col) - 1]
        elif flag == 'right':
            new_current_state = self.scenario[int(coord_row), int(coord_col) + 1]

        # Validate FEC resources
        if self.available_resources(new_current_state):
            # Update agent new current state
            self.agent_info['current_node'] = new_current_state.tolist()
            return True
        else:
            return False

    def get_fec_node(self, agent_trajectory):
        """Get the FEC node to be serving for the agent trajectory"""
        for fec_node, trajectories in self.fec_coverage.items():
            for trajectory in trajectories:
                if set(trajectory) == set(agent_trajectory):
                    return fec_node
        return None

    def offload_resources(self, fec_node, vnf_demand):
        """ Offloading resources in the previous FEC node after linking to a new FEC node"""
        self.network[fec_node]['fec' + str(fec_node) + '_gpu'] += vnf_demand['vnf_gpu']  # Update GPU
        self.network[fec_node]['fec' + str(fec_node) + '_ram'] += vnf_demand['vnf_ram']  # Update RAM
        self.network[fec_node]['fec' + str(fec_node) + '_bw'] += vnf_demand['vnf_bw']  # Update BW

    def available_resources(self, new_current_state):
        """ Identify the FEC node to serve and allocate resources if available. """

        # Get the agent's trajectory
        agent_trajectory = [self.old_current_state, new_current_state]
        # Identify the FEC node to request resources
        fec_node = self.get_fec_node(agent_trajectory)

        # Allocate resources if the agent was just reset
        if self.agent_info['fec_linked_id'] == -1:
            self.agent_info['fec_linked_id'] = fec_node
            if self.fec_resources(fec_node, self.vnf):
                self.network[fec_node]['fec' + str(fec_node) + '_linked'] = 1
                return True

        # Reallocate resources if the FEC node has changed
        elif fec_node != self.agent_info['fec_linked_id']:
            if self.fec_resources(fec_node, self.vnf):
                self.offload_resources(self.agent_info['fec_linked_id'], self.vnf)
                # Update olf FEC node link status back to 0
                self.network[self.agent_info['fec_linked_id']][
                    'fec' + str(self.agent_info['fec_linked_id']) + '_linked'] = 0
                # Update agent's FEC node id link
                self.agent_info['fec_linked_id'] = fec_node
                # Update FEC node link status
                self.network[fec_node]['fec' + str(fec_node) + '_linked'] = 1
                return True

        # If the agent is still linked to the same FEC node, no action is needed
        else:
            return True

        # If resources couldn't be allocated, return False
        return False

    def fec_resources(self, fec_node, vnf_demand):
        """
        Check if there are enough resources available on the FEC node to host the VNF.

        :param fec_node: The FEC node to check for available resources.
        :param vnf_demand: A dictionary containing the VNFs resource demands.
        :return: True if there are enough resources available, False otherwise.
        """
        fec_prefix = 'fec' + str(fec_node) + '_'

        resource_checks = [
            self.network[fec_node][fec_prefix + 'gpu'] >= vnf_demand['vnf_gpu'],  # Check GPU
            self.network[fec_node][fec_prefix + 'ram'] >= vnf_demand['vnf_ram'],  # Check RAM
            self.network[fec_node][fec_prefix + 'bw'] >= vnf_demand['vnf_bw'],  # Check BW
            self.network[fec_node][fec_prefix + 'rtt'] <= vnf_demand['vnf_rtt']  # Check Round Trip Time
        ]

        if all(resource_checks):
            # If all resource checks pass, update the resource values and return True
            self.network[fec_node][fec_prefix + 'gpu'] -= vnf_demand['vnf_gpu']
            self.network[fec_node][fec_prefix + 'ram'] -= vnf_demand['vnf_ram']
            self.network[fec_node][fec_prefix + 'bw'] -= vnf_demand['vnf_bw']
            return True
        else:
            # If any resource check fails, return False
            return False

    def generate_background_traffic_info(self, short_paths=True):
        # Generate initial incoming VNFs
        info = self.generate_vnf(background_vehicle_vnf=True)
        # Get all shortest paths according to VNF demands for each vehicle
        if short_paths:
            route = self.routes.all_shortest_paths(
                info['vnf_source'],
                info['vnf_target']
            )
        # Get all simple paths according to VNF demands for each vehicle
        else:  # WARNING: do not choose this option for large scenarios, may overload CPU and idle the running process
            route = self.routes.all_simple_paths(
                info['vnf_source'],
                info['vnf_target']
            )
        # Out of all possibilities, get only one route randomly
        info['route'] = random.choice(route)
        # Update each vehicle current node
        info['current_node'] = info['route'][0]

        return info

    def initialize_background_traffic(self, num_veh):
        """
            Initializes the background traffic, generating an incoming VNF demand for every vehicle.

            Returns:
                A dictionary containing information about each vehicle:
                    vehicle[i] = {
                        'vnf_source': source_node,
                        'vnf_target': target_node,
                        'vnf_gpu': gpu_demand,
                        'route': [],
                        'active': False,
                        'fec_link': False,
                        'current_node': source_node
                    }
            """
        # Generate incoming VNFs for every background vehicle: {vnf_source, vnf_target, vnf_gpu}

        self.vehicles = {k: {} for k in range(num_veh)}

        for key, _ in self.vehicles.items():
            self.vehicles[key] = self.generate_vehicle_info()

        if VERBOSE:
            print('\nINITIALIZING BACKGROUND TRAFFIC:')
            for k, v in self.vehicles.items():
                print(f"Veh. {k}:  {v}")

    def generate_vehicle_info(self, short_paths=True):
        """
            Generates vehicle information related to the route based on VNF demand and active status.

            Args:
                use_shortest_paths (bool): If True, uses only the shortest paths. Otherwise, uses all possible paths.

            Returns:
                A dictionary containing information about the vehicle:
                    {
                        'vnf_source': source_node,
                        'vnf_target': target_node,
                        'vnf_gpu': gpu_demand,
                        'route': route,
                        'active': False
                    }
                where route is a list of nodes representing the route from the source node to the target node.
                :param short_paths:
            """
        # Generate incoming VNF demand
        vehicle = self.generate_vnf(background_vehicle_vnf=True)  # {vnf_source:, vnf_target:, vnf_gpu:}
        # Get source and target nodes
        source_n = vehicle['vnf_source']
        target_n = vehicle['vnf_target']

        # Compute all possible paths
        if short_paths:
            routes = self.routes.all_shortest_paths(source_n, target_n)
        else:  # WARNING: do not choose this option for large scenarios, may overload CPU and idle the running process
            routes = self.routes.all_simple_paths(source_n, target_n)

        # Choose a random route
        vehicle['route'] = random.choice(routes)
        # Vehicle active status
        vehicle['active'] = False

        return vehicle

    def move_active_vehicle(self, key, vehicle):
        # Get the FEC node that will be serving the vehicle's next trajectory.
        index_current_node = vehicle['route'].index(vehicle['current_node'])
        source_n, next_n = vehicle['route'][index_current_node: index_current_node + 2]
        trajectory = [source_n, next_n]
        fec_n = self.get_fec_node(trajectory)

        # Next vehicle's trajectory requires new FEC node
        if fec_n != vehicle['fec_link']:
            # Resources offloading and allocation
            if self.fec_resources(fec_n, vehicle):
                # Offload resources in current FEC node
                self.offload_resources(vehicle['fec_link'], vehicle)
                # Update vehicle's current node
                self.vehicles[key].update({
                    'current_node': next_n,
                    'fec_link': fec_n
                })

                if VERBOSE:
                    print(f"Veh. {key} active with different FEC node after movement - {vehicle}")

        # Next vehicle's trajectory shares the current FEC node
        else:
            # Update vehicle's current state
            self.vehicles[key]['current_node'] = next_n

            if VERBOSE:
                print(f"Veh. {key} active with same FEC node after movement - {vehicle}")

    def move_inactive_vehicle(self, key, vehicle):
        # Get the FEC node that will be serving the vehicle's first route trajectory.
        source_n = vehicle['vnf_source']
        next_n = vehicle['route'][1]
        trajectory = [source_n, next_n]
        fec_n = self.get_fec_node(trajectory)
        # current node when not active, before the first move, is always the node source from the VNF demand
        self.vehicles[key]['current_node'] = source_n

        # Allocate resources
        if self.fec_resources(fec_n, vehicle):
            # Update vehicle keys
            self.vehicles[key].update({
                'current_node': next_n,
                'fec_link': fec_n,
                'active': True
            })

            if VERBOSE:
                print(f"Veh. {key} new VNF not active - {vehicle}")

    def final_destination_reached(self, key, vehicle):
        """ Checks if background vehicles has reached destination """
        if VERBOSE:
            print(f"Veh. {key} reached final destination - {vehicle}")
        # Offload resources to current FEC node
        self.offload_resources(vehicle['fec_link'], vehicle)
        # Generate new incoming VNF
        self.vehicles[key] = self.generate_vehicle_info()

    def move_background_traffic(self):
        """ Moves the background traffic on every time step."""
        if VERBOSE:
            print('\nMOVING BACKGROUND TRAFFIC:')

        for key, vehicle in self.vehicles.items():
            # key: {'vnf_source':, 'vnf_target':, 'vnf_gpu':, 'vnf_ram':, 'vnf_bw':, 'vnf_rtt': , 'route':, 'active':}
            active = vehicle['active']
            # Vehicles not yet active
            if not active:
                self.move_inactive_vehicle(key, vehicle)
            # Vehicle is active
            elif active:
                self.move_active_vehicle(key, vehicle)
            # Vehicle reaches final destination
            if vehicle['vnf_target'] == vehicle['current_node']:
                self.final_destination_reached(key, vehicle)

    def get_closest_fec_nodes(self, current_node):
        """Gets the closest fec nodes from a node"""
        fec_nodes = []
        for fec, trajectories in self.fec_coverage.items():
            if any(current_node in nodes for nodes in trajectories):
                fec_nodes.append(fec)
        return fec_nodes

    def state_dic_to_list(self):
        return [v for value in self.network.values() for v in value.values()]

    def get_results(self):
        return self.fec_gpu_status, self.fec_ram_status

    def reset(self):
        self.i = 0
        # New incoming VNF
        self.vnf = self.generate_vnf(background_vehicle_vnf=False)
        # Get coordinates for VNF node source
        coord_row, coord_col = np.where(self.scenario == self.vnf['vnf_source'])
        # Get initial current state
        current_node = self.scenario[coord_row.item(), coord_col.item()]
        # Route buffer
        self.route_buffer = []
        # Visited nodes buffer
        self.visited_node = [current_node]
        # Update starting node in route_buffer
        self.route_buffer.append(current_node)

        # Agent info
        self.agent_info = {
            'current_node': current_node,
            'fec_linked_id': -1,
            'remain_agent_steps': self.max_agent_steps,
        }
        # Set network conditions
        self.network = copy.deepcopy(self.default_network)
        # Initialize background traffic
        self.initialize_background_traffic(num_veh=self.vehicles_num)
        # Generate observation
        state = self.state_dic_to_list()
        self.observation = list({**self.vnf, **self.agent_info}.values()) + state

        if VERBOSE:
            df2 = pd.DataFrame([self.observation], columns=[self.obs_features])
            print(f'\nINITIAL STATE:\n{df2}')

        # Return
        return np.array(self.observation, dtype=int)

    def step(self, action):
        self.fec_gpu_status = [self.network[i]['fec' + str(i) + '_gpu'] for i in range(len(self.network))]
        self.fec_ram_status = [self.network[i]['fec' + str(i) + '_ram'] for i in range(len(self.network))]

        self.i += 1
        self.old_current_state = self.agent_info['current_node']

        # Determine movement direction based on the action
        movements = ['up', 'down', 'left', 'right']
        direction = movements[action]

        # Check if movement is valid
        if self.agent_info['current_node'] in getattr(self, 'no_' + direction + '_nodes', []):
            flag = 'No_mov'
        else:
            if self.agent_move(direction):
                flag = 'Yes_mov'
            else:
                flag = 'No_resources'

        # Check if maximum steps reached
        if self.agent_info['remain_agent_steps'] == 0:
            flag = 'Max_step_reach'

        self.move_background_traffic()
        self.route_buffer.append(self.agent_info['current_node'])

        reward, done, info = self.reward_function(flag, self.is_terminal())
        self.agent_info['remain_agent_steps'] -= 1

        state = self.state_dic_to_list()
        self.observation = list({**self.vnf, **self.agent_info}.values()) + state

        if VERBOSE:
            df2 = pd.DataFrame([self.observation], columns=[self.obs_features])
            print('\nSTEP:', self.i, '\n', df2, '\n', reward, done, info)

        return np.array(self.observation, dtype=int), reward, done, info
