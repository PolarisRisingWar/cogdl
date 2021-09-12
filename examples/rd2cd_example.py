import sys
sys.path.insert(0,'whj_code2/cogdl_fork/cogdl')

from cogdl import experiment

experiment(task="node_classification", dataset="whj_code2/cogdl_fork/cogdl_ds/Github_data.pt", model="gcn")
experiment(task="node_classification", dataset="rd2cd_Github", model="gcn")
experiment(task="node_classification", dataset="whj_code2/cogdl_fork/cogdl_ds/Pokec_z_data.pt", model="gcn")