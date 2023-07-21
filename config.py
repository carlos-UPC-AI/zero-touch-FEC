# DEPENDENCIES
import os
import numpy as np


# FUNCTIONS
def get_graph_nodes(edges_cost):
    nodes = []
    for key in edges_cost:
        for node in key:
            if node not in nodes:
                nodes.append(node)
    return nodes


########################################################################################################################
#                                                   GLOBALS                                                            #
########################################################################################################################
# Directories
PROJECT_NAME = "** INSERT-PROJECT-NAME-HERE **"
PROJECT_DIR = "results/" + PROJECT_NAME + "/"
MODELS = "models/"
LOGS = "logs/"
CHECKPOINTS = "checkpoints/"
MODELS_SERVER = "models_server/"

# Parent dir
if not os.path.exists(PROJECT_DIR):
    os.makedirs(PROJECT_DIR)
    print(f"Creating parent dir: {PROJECT_DIR}")
# model dir
if not os.path.exists(PROJECT_DIR + MODELS):
    os.makedirs(PROJECT_DIR + MODELS)
    print(f"Creating dir: {PROJECT_DIR + MODELS}")
# logs dir
if not os.path.exists(PROJECT_DIR + LOGS):
    os.makedirs(PROJECT_DIR + LOGS)
    print(f"Creating dir: {PROJECT_DIR + LOGS}")
# checkpoints dir
if not os.path.exists(PROJECT_DIR + CHECKPOINTS):
    os.makedirs(PROJECT_DIR + CHECKPOINTS)
    print(f"Creating dir: {PROJECT_DIR + CHECKPOINTS}")

# Training
TIME_STEPS = 75000
SEED = 1976
SAVE_FREQ = 50000
MAX_STEPS = 25
BACKGROUND_VEHICLES = 10
EVAL_EPISODES = 10
TRAINING_ROUNDS = 5

# VNF demands
VNF_GPU = 512
VNF_RAM = 8
VNF_BW = 1
VNF_RTT = 15

# Resources max values
MAX_GPU = 2048
MAX_RAM = 32
MAX_BW = 10
MAX_RTT = 10

########################################################################################################################
#                                                   SMALL SCENARIO                                                     #
########################################################################################################################
SCENARIO_SMALL = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ])
EDGE_COST_SMALL = {
    # row 1
    (1, 2): 5,
    (2, 3): 5,
    (3, 4): 5,
    # row 2
    (5, 6): 5,
    (6, 7): 1,
    (7, 8): 5,
    # row 3
    (9, 10): 1,
    (10, 11): 5,
    (11, 12): 1,
    # row 4
    (13, 14): 5,
    (14, 15): 1,
    (15, 16): 5,

    # column 1
    (1, 5): 5,
    (5, 9): 5,
    (9, 13): 5,
    # column 2
    (2, 6): 1,
    (6, 10): 1,
    (10, 14): 1,
    # column 3
    (3, 7): 1,
    (7, 11): 9,
    (11, 15): 1,
    # column 4
    (4, 8): 1,
    (8, 12): 5,
    (12, 16): 1,
}
# Network
NETWORK_SMALL = {
    0: {'fec0_id': 0, 'fec0_gpu': MAX_GPU, 'fec0_ram': MAX_RAM, 'fec0_bw': MAX_BW, 'fec0_rtt': MAX_RTT,
        'fec0_linked': 0},
    1: {'fec0_id': 1, 'fec1_gpu': MAX_GPU, 'fec1_ram': MAX_RAM, 'fec1_bw': MAX_BW, 'fec1_rtt': MAX_RTT,
        'fec1_linked': 0},
    2: {'fec0_id': 2, 'fec2_gpu': MAX_GPU, 'fec2_ram': MAX_RAM, 'fec2_bw': MAX_BW, 'fec2_rtt': MAX_RTT,
        'fec2_linked': 0},
    3: {'fec0_id': 3, 'fec3_gpu': MAX_GPU, 'fec3_ram': MAX_RAM, 'fec3_bw': MAX_BW, 'fec3_rtt': MAX_RTT,
        'fec3_linked': 0},
    4: {'fec0_id': 4, 'fec4_gpu': MAX_GPU, 'fec4_ram': MAX_RAM, 'fec4_bw': MAX_BW, 'fec4_rtt': MAX_RTT,
        'fec4_linked': 0},
    5: {'fec0_id': 5, 'fec5_gpu': MAX_GPU, 'fec5_ram': MAX_RAM, 'fec5_bw': MAX_BW, 'fec5_rtt': MAX_RTT,
        'fec5_linked': 0},
    6: {'fec0_id': 6, 'fec6_gpu': MAX_GPU, 'fec6_ram': MAX_RAM, 'fec6_bw': MAX_BW, 'fec6_rtt': MAX_RTT,
        'fec6_linked': 0},
    7: {'fec0_id': 7, 'fec7_gpu': MAX_GPU, 'fec7_ram': MAX_RAM, 'fec7_bw': MAX_BW, 'fec7_rtt': MAX_RTT,
        'fec7_linked': 0},
    8: {'fec0_id': 8, 'fec8_gpu': MAX_GPU, 'fec8_ram': MAX_RAM, 'fec8_bw': MAX_BW, 'fec8_rtt': MAX_RTT,
        'fec8_linked': 0},
}
# FEC node coverage
FEC_COVERAGE_SMALL = {
    0: [[2, 6], [5, 6]],
    1: [[1, 2], [2, 3], [3, 4]],
    2: [[3, 7], [7, 8]],
    3: [[1, 5], [5, 9], [9, 13]],
    4: [[6, 7], [7, 11], [11, 10], [10, 6]],
    5: [[4, 8], [8, 12], [12, 16]],
    6: [[9, 10], [10, 14]],
    7: [[11, 12], [15, 11]],
    8: [[13, 14], [14, 15], [15, 16]],
}
# Obs space
STATE_FEATURES_SMALL = [
    # Variables from the incoming VNF demands
    'vnf_source',  # 1
    'vnf_target',  # 2
    'vnf_gpu',  # 3
    'vnf_ram',  # 4
    'vnf_bw',  # 5
    'vnf_rtt',  # 6
    # Agent's vehicle current node location
    'current_node',  # 7
    'fec_linked_id',  # 8
    # Max steps limit countdown
    'remain_agent_steps',  # 9
    # FEC nodes variables
    'fec0_id', 'fec0_gpu', 'fec0_ram', 'fec0_bw', 'fec0_rtt', 'fec0_linked',  # 10-15
    'fec1_id', 'fec1_gpu', 'fec1_ram', 'fec1_bw', 'fec1_rtt', 'fec1_linked',  # 16-21
    'fec2_id', 'fec2_gpu', 'fec2_ram', 'fec2_bw', 'fec2_rtt', 'fec2_linked',  # 22-27
    'fec3_id', 'fec3_gpu', 'fec3_ram', 'fec3_bw', 'fec3_rtt', 'fec3_linked',  # 28-33
    'fec4_id', 'fec4_gpu', 'fec4_ram', 'fec4_bw', 'fec4_rtt', 'fec4_linked',  # 34-39
    'fec5_id', 'fec5_gpu', 'fec5_ram', 'fec5_bw', 'fec5_rtt', 'fec5_linked',  # 40-45
    'fec6_id', 'fec6_gpu', 'fec6_ram', 'fec6_bw', 'fec6_rtt', 'fec6_linked',  # 46-51
    'fec7_id', 'fec7_gpu', 'fec7_ram', 'fec7_bw', 'fec7_rtt', 'fec7_linked',  # 52-57
    'fec8_id', 'fec8_gpu', 'fec8_ram', 'fec8_bw', 'fec8_rtt', 'fec8_linked',  # 58-63
]
# Min and Max values
MIN_MAX_STATE_SMALL = {
    'vnf_source': [min(get_graph_nodes(EDGE_COST_SMALL)), max(get_graph_nodes(EDGE_COST_SMALL))],
    'vnf_target': [min(get_graph_nodes(EDGE_COST_SMALL)), max(get_graph_nodes(EDGE_COST_SMALL))],
    'vnf_gpu': [0, MAX_GPU],
    'vnf_ram': [0, MAX_RAM],
    'vnf_bw': [0, MAX_BW],
    'vnf_rtt': [MAX_RTT, VNF_RTT],
    'current_node': [min(get_graph_nodes(EDGE_COST_SMALL)), max(get_graph_nodes(EDGE_COST_SMALL))],
    'fec_linked_id': [-1, max(FEC_COVERAGE_SMALL)],
    'remain_agent_steps': [0, 25],
    'fec_id': [min(FEC_COVERAGE_SMALL), max(FEC_COVERAGE_SMALL)],
    'fec_gpu': [0, MAX_GPU],
    'fec_ram': [0, MAX_RAM],
    'fec_bw': [0, MAX_BW],
    'fec_rtt': [0, MAX_RTT],
    'fec_link_active': [0, 1],
}
NO_UP_NODES_SMALL = [1, 2, 3, 4]
NO_DOWN_NODES_SMALL = [13, 14, 15, 16]
NO_LEFT_NODES_SMALL = [1, 5, 9, 13]
NO_RIGHT_NODES_SMALL = [4, 8, 12, 16]
ENV_SMALL = {
    'scenario': SCENARIO_SMALL,
    'edges_cost': EDGE_COST_SMALL,
    'network': NETWORK_SMALL,
    'fec_coverage': FEC_COVERAGE_SMALL,
    'state_features': STATE_FEATURES_SMALL,
    'no_up_nodes': NO_UP_NODES_SMALL,
    'no_down_nodes': NO_DOWN_NODES_SMALL,
    'no_left_nodes': NO_LEFT_NODES_SMALL,
    'no_right_nodes': NO_RIGHT_NODES_SMALL,
    'nodes': get_graph_nodes(EDGE_COST_SMALL)
}

########################################################################################################################
#                                                   MEDIUM SCENARIO                                                    #
########################################################################################################################
SCENARIO_MEDIUM = np.array(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20]
    ])
EDGE_COST_MEDIUM = {
    # row 1
    (1, 2): 5,
    (2, 3): 5,
    (3, 4): 5,
    # row 2
    (5, 6): 5,
    (6, 7): 1,
    (7, 8): 5,
    # row 3
    (9, 10): 1,
    (10, 11): 5,
    (11, 12): 1,
    # row 4
    (13, 14): 5,
    (14, 15): 1,
    (15, 16): 5,
    # row 5
    (17, 18): 5,
    (18, 19): 5,
    (19, 20): 5,
    # column 1
    (1, 5): 5,
    (5, 9): 5,
    (9, 13): 5,
    (13, 17): 5,
    # column 2
    (2, 6): 1,
    (6, 10): 1,
    (10, 14): 1,
    (14, 18): 1,
    # column 3
    (3, 7): 1,
    (7, 11): 9,
    (11, 15): 1,
    (15, 19): 1,
    # column 4
    (4, 8): 1,
    (8, 12): 5,
    (12, 16): 1,
    (16, 20): 1
}
# Network
NETWORK_MEDIUM = {
    0: {'fec0_id': 0, 'fec0_gpu': MAX_GPU, 'fec0_ram': MAX_RAM, 'fec0_bw': MAX_BW, 'fec0_rtt': MAX_RTT,
        'fec0_linked': 0},
    1: {'fec0_id': 1, 'fec1_gpu': MAX_GPU, 'fec1_ram': MAX_RAM, 'fec1_bw': MAX_BW, 'fec1_rtt': MAX_RTT,
        'fec1_linked': 0},
    2: {'fec0_id': 2, 'fec2_gpu': MAX_GPU, 'fec2_ram': MAX_RAM, 'fec2_bw': MAX_BW, 'fec2_rtt': MAX_RTT,
        'fec2_linked': 0},
    3: {'fec0_id': 3, 'fec3_gpu': MAX_GPU, 'fec3_ram': MAX_RAM, 'fec3_bw': MAX_BW, 'fec3_rtt': MAX_RTT,
        'fec3_linked': 0},
    4: {'fec0_id': 4, 'fec4_gpu': MAX_GPU, 'fec4_ram': MAX_RAM, 'fec4_bw': MAX_BW, 'fec4_rtt': MAX_RTT,
        'fec4_linked': 0},
    5: {'fec0_id': 5, 'fec5_gpu': MAX_GPU, 'fec5_ram': MAX_RAM, 'fec5_bw': MAX_BW, 'fec5_rtt': MAX_RTT,
        'fec5_linked': 0},
    6: {'fec0_id': 6, 'fec6_gpu': MAX_GPU, 'fec6_ram': MAX_RAM, 'fec6_bw': MAX_BW, 'fec6_rtt': MAX_RTT,
        'fec6_linked': 0},
    7: {'fec0_id': 7, 'fec7_gpu': MAX_GPU, 'fec7_ram': MAX_RAM, 'fec7_bw': MAX_BW, 'fec7_rtt': MAX_RTT,
        'fec7_linked': 0},
    8: {'fec0_id': 8, 'fec8_gpu': MAX_GPU, 'fec8_ram': MAX_RAM, 'fec8_bw': MAX_BW, 'fec8_rtt': MAX_RTT,
        'fec8_linked': 0},
    9: {'fec0_id': 9, 'fec9_gpu': MAX_GPU, 'fec9_ram': MAX_RAM, 'fec9_bw': MAX_BW, 'fec9_rtt': MAX_RTT,
        'fec9_linked': 0},
    10: {'fec0_id': 10, 'fec10_gpu': MAX_GPU, 'fec10_ram': MAX_RAM, 'fec10_bw': MAX_BW, 'fec10_rtt': MAX_RTT,
         'fec10_linked': 0},
    11: {'fec0_id': 11, 'fec11_gpu': MAX_GPU, 'fec11_ram': MAX_RAM, 'fec11_bw': MAX_BW, 'fec11_rtt': MAX_RTT,
         'fec11_linked': 0},
}
# FEC node coverage
FEC_COVERAGE_MEDIUM = {
    0: [[2, 6], [5, 6]],
    1: [[1, 2], [2, 3], [3, 4]],
    2: [[3, 7], [7, 8]],
    3: [[1, 5], [5, 9]],
    4: [[6, 7], [7, 11], [11, 10], [10, 6]],
    5: [[4, 8], [8, 12]],
    6: [[9, 10], [10, 14], [14, 13], [13, 9]],
    7: [[11, 12], [16, 15], [15, 11], [12, 16]],
    8: [[13, 17]],
    9: [[14, 15], [15, 19], [14, 18]],
    10: [[16, 20]],
    11: [[17, 18], [18, 19], [19, 20]],
}
# Obs space
STATE_FEATURES_MEDIUM = [
    # Variables from the incoming VNF demands
    'vnf_source',  # 1
    'vnf_target',  # 2
    'vnf_gpu',  # 3
    'vnf_ram',  # 4
    'vnf_bw',  # 5
    'vnf_rtt',  # 6
    # Agent's vehicle current node location
    'current_node',  # 7
    'fec_linked_id',  # 8
    # Max steps limit countdown
    'remain_agent_steps',  # 9
    # FEC nodes variables
    'fec0_id', 'fec0_gpu', 'fec0_ram', 'fec0_bw', 'fec0_rtt', 'fec0_linked',  # 10-15
    'fec1_id', 'fec1_gpu', 'fec1_ram', 'fec1_bw', 'fec1_rtt', 'fec1_linked',  # 16-21
    'fec2_id', 'fec2_gpu', 'fec2_ram', 'fec2_bw', 'fec2_rtt', 'fec2_linked',  # 22-27
    'fec3_id', 'fec3_gpu', 'fec3_ram', 'fec3_bw', 'fec3_rtt', 'fec3_linked',  # 28-33
    'fec4_id', 'fec4_gpu', 'fec4_ram', 'fec4_bw', 'fec4_rtt', 'fec4_linked',  # 34-39
    'fec5_id', 'fec5_gpu', 'fec5_ram', 'fec5_bw', 'fec5_rtt', 'fec5_linked',  # 40-45
    'fec6_id', 'fec6_gpu', 'fec6_ram', 'fec6_bw', 'fec6_rtt', 'fec6_linked',  # 46-51
    'fec7_id', 'fec7_gpu', 'fec7_ram', 'fec7_bw', 'fec7_rtt', 'fec7_linked',  # 52-57
    'fec8_id', 'fec8_gpu', 'fec8_ram', 'fec8_bw', 'fec8_rtt', 'fec8_linked',  # 58-63
    'fec9_id', 'fec9_gpu', 'fec9_ram', 'fec9_bw', 'fec9_rtt', 'fec9_linked',  # 64-69
    'fec10_id', 'fec10_gpu', 'fec10_ram', 'fec10_bw', 'fec10_rtt', 'fec10_linked',  # 70-75
    'fec11_id', 'fec11_gpu', 'fec11_ram', 'fec11_bw', 'fec11_rtt', 'fec11_linked'  # 76-81
]
# Min and Max values
MIN_MAX_STATE_MEDIUM = {
    'vnf_source': [min(get_graph_nodes(EDGE_COST_MEDIUM)), max(get_graph_nodes(EDGE_COST_MEDIUM))],
    'vnf_target': [min(get_graph_nodes(EDGE_COST_MEDIUM)), max(get_graph_nodes(EDGE_COST_MEDIUM))],
    'vnf_gpu': [0, MAX_GPU],
    'vnf_ram': [0, MAX_RAM],
    'vnf_bw': [0, MAX_BW],
    'vnf_rtt': [MAX_RTT, VNF_RTT],
    'current_node': [min(get_graph_nodes(EDGE_COST_MEDIUM)), max(get_graph_nodes(EDGE_COST_MEDIUM))],
    'fec_linked_id': [-1, max(FEC_COVERAGE_MEDIUM)],
    'remain_agent_steps': [0, 25],
    'fec_id': [min(FEC_COVERAGE_MEDIUM), max(FEC_COVERAGE_MEDIUM)],
    'fec_gpu': [0, MAX_GPU],
    'fec_ram': [0, MAX_RAM],
    'fec_bw': [0, MAX_BW],
    'fec_rtt': [0, MAX_RTT],
    'fec_link_active': [0, 1],
}
NO_UP_NODES_MEDIUM = [1, 2, 3, 4]
NO_DOWN_NODES_MEDIUM = [17, 18, 19, 20]
NO_LEFT_NODES_MEDIUM = [1, 5, 9, 13, 17]
NO_RIGHT_NODES_MEDIUM = [4, 8, 12, 16, 20]
ENV_MEDIUM = {
    'scenario': SCENARIO_MEDIUM,
    'edges_cost': EDGE_COST_MEDIUM,
    'network': NETWORK_MEDIUM,
    'fec_coverage': FEC_COVERAGE_MEDIUM,
    'state_features': STATE_FEATURES_MEDIUM,
    'no_up_nodes': NO_UP_NODES_MEDIUM,
    'no_down_nodes': NO_DOWN_NODES_MEDIUM,
    'no_left_nodes': NO_LEFT_NODES_MEDIUM,
    'no_right_nodes': NO_RIGHT_NODES_MEDIUM,
    'nodes': get_graph_nodes(EDGE_COST_MEDIUM)
}

########################################################################################################################
#                                                   LARGE SCENARIO                                                     #
########################################################################################################################
# Scenario
SCENARIO_LARGE = np.array(
    [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14, 15, 16, 17, 18],
        [19, 20, 21, 22, 23, 24, 25, 26, 27],
        [28, 29, 30, 31, 32, 33, 34, 35, 36],
        [37, 38, 39, 40, 41, 42, 43, 44, 45]
    ]
)
# Graph edges
EDGES_COST_LARGE = {
    # row 1
    (1, 2): 5,
    (2, 3): 5,
    (3, 4): 5,
    (4, 5): 5,
    (5, 6): 5,
    (6, 7): 5,
    (7, 8): 5,
    (8, 9): 5,

    (1, 10): 5,
    (2, 11): 1,
    (3, 12): 1,
    (4, 13): 1,
    (5, 14): 1,
    (6, 15): 1,
    (7, 16): 1,
    (8, 17): 1,
    (9, 18): 5,
    # row 2
    (10, 11): 5,
    (11, 12): 1,
    (12, 13): 5,
    (13, 14): 1,
    (14, 15): 5,
    (15, 16): 1,
    (16, 17): 5,
    (17, 18): 1,

    (10, 19): 5,
    (11, 20): 1,
    (12, 21): 9,
    (13, 22): 5,
    (14, 23): 9,
    (15, 24): 5,
    (16, 25): 9,
    (17, 26): 5,
    (18, 27): 5,
    # row 3
    (19, 20): 1,
    (20, 21): 5,
    (21, 22): 1,
    (22, 23): 5,
    (23, 24): 1,
    (24, 25): 5,
    (25, 26): 1,
    (26, 27): 1,

    (19, 28): 5,
    (20, 29): 1,
    (21, 30): 1,
    (22, 31): 1,
    (23, 32): 1,
    (24, 33): 1,
    (25, 34): 1,
    (26, 35): 1,
    (27, 36): 5,
    # row 4
    (28, 29): 5,
    (29, 30): 1,
    (30, 31): 5,
    (31, 32): 1,
    (32, 33): 5,
    (33, 34): 1,
    (34, 35): 5,
    (35, 36): 1,

    (28, 37): 5,
    (29, 38): 1,
    (30, 39): 1,
    (31, 40): 1,
    (32, 41): 1,
    (33, 42): 1,
    (34, 43): 1,
    (35, 44): 1,
    (36, 45): 5,
    # row 5
    (37, 38): 5,
    (38, 39): 5,
    (39, 40): 5,
    (40, 41): 5,
    (41, 42): 5,
    (42, 43): 5,
    (43, 44): 5,
    (44, 45): 5

}
# Network
NETWORK_LARGE = {
    0: {'fec0_id': 0, 'fec0_gpu': MAX_GPU, 'fec0_ram': MAX_RAM, 'fec0_bw': MAX_BW, 'fec0_rtt': MAX_RTT,
        'fec0_linked': 0},
    1: {'fec1_id': 1, 'fec1_gpu': MAX_GPU, 'fec1_ram': MAX_RAM, 'fec1_bw': MAX_BW, 'fec1_rtt': MAX_RTT,
        'fec1_linked': 0},
    2: {'fec2_id': 2, 'fec2_gpu': MAX_GPU, 'fec2_ram': MAX_RAM, 'fec2_bw': MAX_BW, 'fec2_rtt': MAX_RTT,
        'fec2_linked': 0},
    3: {'fec3_id': 3, 'fec3_gpu': MAX_GPU, 'fec3_ram': MAX_RAM, 'fec3_bw': MAX_BW, 'fec3_rtt': MAX_RTT,
        'fec3_linked': 0},
    4: {'fec4_id': 4, 'fec4_gpu': MAX_GPU, 'fec4_ram': MAX_RAM, 'fec4_bw': MAX_BW, 'fec4_rtt': MAX_RTT,
        'fec4_linked': 0},
    5: {'fec5_id': 5, 'fec5_gpu': MAX_GPU, 'fec5_ram': MAX_RAM, 'fec5_bw': MAX_BW, 'fec5_rtt': MAX_RTT,
        'fec5_linked': 0},
    6: {'fec6_id': 6, 'fec6_gpu': MAX_GPU, 'fec6_ram': MAX_RAM, 'fec6_bw': MAX_BW, 'fec6_rtt': MAX_RTT,
        'fec6_linked': 0},
    7: {'fec7_id': 7, 'fec7_gpu': MAX_GPU, 'fec7_ram': MAX_RAM, 'fec7_bw': MAX_BW, 'fec7_rtt': MAX_RTT,
        'fec7_linked': 0},
    8: {'fec8_id': 8, 'fec8_gpu': MAX_GPU, 'fec8_ram': MAX_RAM, 'fec8_bw': MAX_BW, 'fec8_rtt': MAX_RTT,
        'fec8_linked': 0},
    9: {'fec9_id': 9, 'fec9_gpu': MAX_GPU, 'fec9_ram': MAX_RAM, 'fec9_bw': MAX_BW, 'fec9_rtt': MAX_RTT,
        'fec9_linked': 0},
    10: {'fec10_id': 10, 'fec10_gpu': MAX_GPU, 'fec10_ram': MAX_RAM, 'fec10_bw': MAX_BW, 'fec10_rtt': MAX_RTT,
         'fec10_linked': 0},
    11: {'fec11_id': 11, 'fec11_gpu': MAX_GPU, 'fec11_ram': MAX_RAM, 'fec11_bw': MAX_BW, 'fec11_rtt': MAX_RTT,
         'fec11_linked': 0},
    12: {'fec12_id': 12, 'fec12_gpu': MAX_GPU, 'fec12_ram': MAX_RAM, 'fec12_bw': MAX_BW, 'fec12_rtt': MAX_RTT,
         'fec12_linked': 0},
    13: {'fec13_id': 13, 'fec13_gpu': MAX_GPU, 'fec13_ram': MAX_RAM, 'fec13_bw': MAX_BW, 'fec13_rtt': MAX_RTT,
         'fec13_linked': 0},
    14: {'fec14_id': 14, 'fec14_gpu': MAX_GPU, 'fec14_ram': MAX_RAM, 'fec14_bw': MAX_BW, 'fec14_rtt': MAX_RTT,
         'fec14_linked': 0},
    15: {'fec15_id': 15, 'fec15_gpu': MAX_GPU, 'fec15_ram': MAX_RAM, 'fec15_bw': MAX_BW, 'fec15_rtt': MAX_RTT,
         'fec15_linked': 0},
    16: {'fec16_id': 16, 'fec16_gpu': MAX_GPU, 'fec16_ram': MAX_RAM, 'fec16_bw': MAX_BW, 'fec16_rtt': MAX_RTT,
         'fec16_linked': 0},
    17: {'fec17_id': 17, 'fec17_gpu': MAX_GPU, 'fec17_ram': MAX_RAM, 'fec17_bw': MAX_BW, 'fec17_rtt': MAX_RTT,
         'fec17_linked': 0},
    18: {'fec18_id': 18, 'fec18_gpu': MAX_GPU, 'fec18_ram': MAX_RAM, 'fec18_bw': MAX_BW, 'fec18_rtt': MAX_RTT,
         'fec18_linked': 0},
    19: {'fec19_id': 19, 'fec19_gpu': MAX_GPU, 'fec19_ram': MAX_RAM, 'fec19_bw': MAX_BW, 'fec19_rtt': MAX_RTT,
         'fec19_linked': 0},
    20: {'fec20_id': 20, 'fec20_gpu': MAX_GPU, 'fec20_ram': MAX_RAM, 'fec20_bw': MAX_BW, 'fec20_rtt': MAX_RTT,
         'fec20_linked': 0},
    21: {'fec21_id': 21, 'fec21_gpu': MAX_GPU, 'fec21_ram': MAX_RAM, 'fec21_bw': MAX_BW, 'fec21_rtt': MAX_RTT,
         'fec21_linked': 0},
    22: {'fec22_id': 22, 'fec22_gpu': MAX_GPU, 'fec22_ram': MAX_RAM, 'fec22_bw': MAX_BW, 'fec22_rtt': MAX_RTT,
         'fec22_linked': 0},
    23: {'fec23_id': 23, 'fec23_gpu': MAX_GPU, 'fec23_ram': MAX_RAM, 'fec23_bw': MAX_BW, 'fec23_rtt': MAX_RTT,
         'fec23_linked': 0},
    24: {'fec24_id': 24, 'fec24_gpu': MAX_GPU, 'fec24_ram': MAX_RAM, 'fec24_bw': MAX_BW, 'fec24_rtt': MAX_RTT,
         'fec24_linked': 0},
    25: {'fec25_id': 25, 'fec25_gpu': MAX_GPU, 'fec25_ram': MAX_RAM, 'fec25_bw': MAX_BW, 'fec25_rtt': MAX_RTT,
         'fec25_linked': 0},
    26: {'fec26_id': 26, 'fec26_gpu': MAX_GPU, 'fec26_ram': MAX_RAM, 'fec26_bw': MAX_BW, 'fec26_rtt': MAX_RTT,
         'fec26_linked': 0},
    27: {'fec27_id': 27, 'fec27_gpu': MAX_GPU, 'fec27_ram': MAX_RAM, 'fec27_bw': MAX_BW, 'fec27_rtt': MAX_RTT,
         'fec27_linked': 0},
}
# FEC node coverage
FEC_COVERAGE_LARGE = {
    0: [[1, 2], [10, 11], [2, 11]],
    1: [[2, 3]],
    2: [[3, 12], [12, 13], [4, 13], [3, 4]],
    3: [[4, 5]],
    4: [[5, 14], [14, 15], [15, 6], [6, 5]],
    5: [[6, 7]],
    6: [[7, 16], [16, 17], [17, 8], [8, 7]],
    7: [[8, 9]],
    8: [[1, 10], [10, 19]],
    9: [[11, 20], [20, 21], [21, 12], [12, 11]],
    10: [[13, 22], [22, 23], [23, 14], [14, 13]],
    11: [[15, 24], [24, 25], [25, 16], [16, 15]],
    12: [[17, 26], [26, 27], [27, 18], [18, 17]],
    13: [[9, 18]],
    14: [[19, 28], [28, 29], [29, 20], [20, 19]],
    15: [[21, 30], [30, 31], [31, 22], [22, 21]],
    16: [[23, 32], [32, 33], [33, 24], [24, 23]],
    17: [[25, 34], [34, 35], [35, 26], [26, 25]],
    18: [[28, 37]],
    19: [[29, 38], [38, 39], [39, 30], [30, 29]],
    20: [[31, 40], [40, 41], [41, 32], [32, 31]],
    21: [[33, 42], [42, 43], [43, 34], [34, 33]],
    22: [[35, 44], [44, 45], [45, 36], [36, 35]],
    23: [[27, 36]],
    24: [[37, 38]],
    25: [[39, 40]],
    26: [[41, 42]],
    27: [[43, 44], [44, 45]],
}
# Obs space
STATE_FEATURES_LARGE = [
    # Variables from the incoming VNF demands
    'vnf_source',  # 1
    'vnf_target',  # 2
    'vnf_gpu',  # 3
    'vnf_ram',  # 4
    'vnf_bw',  # 5
    'vnf_rtt',  # 6
    # Agent's vehicle current node location
    'current_node',  # 7
    'fec_linked_id',  # 8
    # Max steps limit countdown
    'remain_agent_steps',  # 9
    # FEC nodes variables
    'fec0_id', 'fec0_gpu', 'fec0_ram', 'fec0_bw', 'fec0_rtt', 'fec0_linked',
    'fec1_id', 'fec1_gpu', 'fec1_ram', 'fec1_bw', 'fec1_rtt', 'fec1_linked',
    'fec2_id', 'fec2_gpu', 'fec2_ram', 'fec2_bw', 'fec2_rtt', 'fec2_linked',
    'fec3_id', 'fec3_gpu', 'fec3_ram', 'fec3_bw', 'fec3_rtt', 'fec3_linked',
    'fec4_id', 'fec4_gpu', 'fec4_ram', 'fec4_bw', 'fec4_rtt', 'fec4_linked',
    'fec5_id', 'fec5_gpu', 'fec5_ram', 'fec5_bw', 'fec5_rtt', 'fec5_linked',
    'fec6_id', 'fec6_gpu', 'fec6_ram', 'fec6_bw', 'fec6_rtt', 'fec6_linked',
    'fec7_id', 'fec7_gpu', 'fec7_ram', 'fec7_bw', 'fec7_rtt', 'fec7_linked',
    'fec8_id', 'fec8_gpu', 'fec8_ram', 'fec8_bw', 'fec8_rtt', 'fec8_linked',
    'fec9_id', 'fec9_gpu', 'fec9_ram', 'fec9_bw', 'fec9_rtt', 'fec9_linked',
    'fec10_id', 'fec10_gpu', 'fec10_ram', 'fec10_bw', 'fec10_rtt', 'fec10_linked',
    'fec11_id', 'fec11_gpu', 'fec11_ram', 'fec11_bw', 'fec11_rtt', 'fec11_linked',
    'fec12_id', 'fec12_gpu', 'fec12_ram', 'fec12_bw', 'fec12_rtt', 'fec12_linked',
    'fec13_id', 'fec13_gpu', 'fec13_ram', 'fec13_bw', 'fec13_rtt', 'fec13_linked',
    'fec14_id', 'fec14_gpu', 'fec14_ram', 'fec14_bw', 'fec14_rtt', 'fec14_linked',
    'fec15_id', 'fec15_gpu', 'fec15_ram', 'fec15_bw', 'fec15_rtt', 'fec15_linked',
    'fec16_id', 'fec16_gpu', 'fec16_ram', 'fec16_bw', 'fec16_rtt', 'fec16_linked',
    'fec17_id', 'fec17_gpu', 'fec17_ram', 'fec17_bw', 'fec17_rtt', 'fec17_linked',
    'fec18_id', 'fec18_gpu', 'fec18_ram', 'fec18_bw', 'fec18_rtt', 'fec18_linked',
    'fec19_id', 'fec19_gpu', 'fec19_ram', 'fec19_bw', 'fec19_rtt', 'fec19_linked',
    'fec20_id', 'fec20_gpu', 'fec20_ram', 'fec20_bw', 'fec20_rtt', 'fec20_linked',
    'fec21_id', 'fec21_gpu', 'fec21_ram', 'fec21_bw', 'fec21_rtt', 'fec21_linked',
    'fec22_id', 'fec22_gpu', 'fec22_ram', 'fec22_bw', 'fec22_rtt', 'fec22_linked',
    'fec23_id', 'fec23_gpu', 'fec23_ram', 'fec23_bw', 'fec23_rtt', 'fec23_linked',
    'fec24_id', 'fec24_gpu', 'fec24_ram', 'fec24_bw', 'fec24_rtt', 'fec24_linked',
    'fec25_id', 'fec25_gpu', 'fec25_ram', 'fec25_bw', 'fec25_rtt', 'fec25_linked',
    'fec26_id', 'fec26_gpu', 'fec26_ram', 'fec26_bw', 'fec26_rtt', 'fec26_linked',
    'fec27_id', 'fec27_gpu', 'fec27_ram', 'fec27_bw', 'fec27_rtt', 'fec27_linked',
]
NO_UP_NODES_LARGE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
NO_DOWN_NODES_LARGE = [37, 38, 39, 40, 41, 42, 43, 44, 45]
NO_LEFT_NODES_LARGE = [1, 10, 19, 28, 37]
NO_RIGHT_NODES_LARGE = [9, 18, 27, 36, 45]
ENV_LARGE = {
    'scenario': SCENARIO_LARGE,
    'edges_cost': EDGES_COST_LARGE,
    'network': NETWORK_LARGE,
    'fec_coverage': FEC_COVERAGE_LARGE,
    'state_features': STATE_FEATURES_LARGE,
    'no_up_nodes': NO_UP_NODES_LARGE,
    'no_down_nodes': NO_DOWN_NODES_LARGE,
    'no_left_nodes': NO_LEFT_NODES_LARGE,
    'no_right_nodes': NO_RIGHT_NODES_LARGE,
    'nodes': get_graph_nodes(EDGES_COST_LARGE),
}
########################################################################################################################

# Scenario selection (large or small)
MIN_MAX_STATE_VALUES = MIN_MAX_STATE_MEDIUM
STATE_FEATURES = STATE_FEATURES_MEDIUM
