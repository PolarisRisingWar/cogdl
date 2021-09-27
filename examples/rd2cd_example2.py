import sys
sys.path.insert(0,'whj_code2/cogdl_fork/cogdl')

from cogdl import experiment

device_id=[3]
task="node_classification"
datasets=['rd2cd_Alpha','rd2cd_Github','rd2cd_Elliptic','rd2cd_bgp','rd2cd_ssn5']
models=['gcn','gat','mlp','correct_smooth_mlp','ppnp','sage']
experiment(task=task,dataset=datasets,model=models,device_id=device_id)