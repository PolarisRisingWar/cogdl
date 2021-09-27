import sys
sys.path.insert(0,'whj_code2/cogdl_fork/cogdl')

from cogdl import experiment

experiment(task="node_classification", dataset="rd2cd_Alpha", model="gcn",device_id=[3])
experiment(task="node_classification", dataset="rd2cd_Github", model="gcn",device_id=[3])
experiment(task="node_classification", dataset="rd2cd_Elliptic", model="gcn",device_id=[3])
experiment(task="node_classification", dataset="rd2cd_bgp", model="gcn",device_id=[3])
experiment(task="node_classification", dataset="rd2cd_ssn5", model="gcn",device_id=[3])
experiment(task="node_classification", dataset="rd2cd_Alpha", model="gat",device_id=[3])
experiment(task="node_classification", dataset="rd2cd_bgp", model="ppnp",device_id=[3])
experiment(task="node_classification", dataset="rd2cd_Elliptic", model="correct_smooth_mlp",device_id=[3])
experiment(task="node_classification", dataset="rd2cd_Github", model="mlp",device_id=[3])
experiment(task="node_classification", dataset="rd2cd_ssn5", model="sage",device_id=[3])

"""
Failed to load C version of sampling, use python version instead.
Namespace(activation='relu', actnn=False, checkpoint=None, cpu=False, dataset='rd2cd_Alpha', device_id=[3], dropout=0.5, fast_spmm=False, hidden_size=64, inference=False, lr=0.01, max_epoch=500, missing_rate=0, model='gcn', norm=None, num_classes=None, num_features=None, num_layers=2, patience=100, residual=False, save_dir='.', save_model=None, seed=1, task='node_classification', trainer=None, use_best_config=False, weight_decay=0.0005)
Epoch: 499, Train: 0.9344, Val: 0.9466, ValLoss: 0.2763: 100%|█████████████████████| 500/500 [00:29<00:00, 16.93it/s]
Valid accurracy =  0.9496
Test accuracy = 0.9389
| Variant                | Acc           | ValAcc        |
|------------------------|---------------|---------------|
| ('rd2cd_Alpha', 'gcn') | 0.9389±0.0000 | 0.9496±0.0000 |
Namespace(activation='relu', actnn=False, checkpoint=None, cpu=False, dataset='rd2cd_Github', device_id=[3], dropout=0.5, fast_spmm=False, hidden_size=64, inference=False, lr=0.01, max_epoch=500, missing_rate=0, model='gcn', norm=None, num_classes=None, num_features=None, num_layers=2, patience=100, residual=False, save_dir='.', save_model=None, seed=1, task='node_classification', trainer=None, use_best_config=False, weight_decay=0.0005)
Epoch: 199, Train: 0.9019, Val: 0.8550, ValLoss: 0.3704:  39%|████████▎            | 197/500 [00:09<00:13, 21.83it/s]
Valid accurracy =  0.8584
Test accuracy = 0.8527
| Variant                 | Acc           | ValAcc        |
|-------------------------|---------------|---------------|
| ('rd2cd_Github', 'gcn') | 0.8527±0.0000 | 0.8584±0.0000 |
Namespace(activation='relu', actnn=False, checkpoint=None, cpu=False, dataset='rd2cd_Elliptic', device_id=[3], dropout=0.5, fast_spmm=False, hidden_size=64, inference=False, lr=0.01, max_epoch=500, missing_rate=0, model='gcn', norm=None, num_classes=None, num_features=None, num_layers=2, patience=100, residual=False, save_dir='.', save_model=None, seed=1, task='node_classification', trainer=None, use_best_config=False, weight_decay=0.0005)
Epoch: 499, Train: 0.8567, Val: 0.8537, ValLoss: 0.3750: 100%|█████████████████████| 500/500 [00:27<00:00, 18.47it/s]
Valid accurracy =  0.8541
Test accuracy = 0.8509
| Variant                   | Acc           | ValAcc        |
|---------------------------|---------------|---------------|
| ('rd2cd_Elliptic', 'gcn') | 0.8509±0.0000 | 0.8541±0.0000 |
Namespace(activation='relu', actnn=False, checkpoint=None, cpu=False, dataset='rd2cd_bgp', device_id=[3], dropout=0.5, fast_spmm=False, hidden_size=64, inference=False, lr=0.01, max_epoch=500, missing_rate=0, model='gcn', norm=None, num_classes=None, num_features=None, num_layers=2, patience=100, residual=False, save_dir='.', save_model=None, seed=1, task='node_classification', trainer=None, use_best_config=False, weight_decay=0.0005)
Epoch: 499, Train: 0.6739, Val: 0.6025, ValLoss: 1.1521: 100%|█████████████████████| 500/500 [00:24<00:00, 20.77it/s]
Valid accurracy =  0.6128
Test accuracy = 0.6159
| Variant              | Acc           | ValAcc        |
|----------------------|---------------|---------------|
| ('rd2cd_bgp', 'gcn') | 0.6159±0.0000 | 0.6128±0.0000 |
Namespace(activation='relu', actnn=False, checkpoint=None, cpu=False, dataset='rd2cd_ssn5', device_id=[3], dropout=0.5, fast_spmm=False, hidden_size=64, inference=False, lr=0.01, max_epoch=500, missing_rate=0, model='gcn', norm=None, num_classes=None, num_features=None, num_layers=2, patience=100, residual=False, save_dir='.', save_model=None, seed=1, task='node_classification', trainer=None, use_best_config=False, weight_decay=0.0005)
Epoch: 499, Train: 0.8314, Val: 0.8293, ValLoss: 0.5145: 100%|█████████████████████| 500/500 [00:33<00:00, 15.08it/s]
Valid accurracy =  0.8298
Test accuracy = 0.8361
| Variant               | Acc           | ValAcc        |
|-----------------------|---------------|---------------|
| ('rd2cd_ssn5', 'gcn') | 0.8361±0.0000 | 0.8298±0.0000 |
Namespace(actnn=False, alpha=0.2, attn_drop=0.5, checkpoint=None, cpu=False, dataset='rd2cd_Alpha', device_id=[3], dropout=0.6, fast_spmm=False, hidden_size=8, inference=False, last_nhead=1, lr=0.01, max_epoch=500, missing_rate=0, model='gat', nhead=8, norm=None, num_classes=None, num_features=None, num_layers=2, patience=100, residual=False, save_dir='.', save_model=None, seed=1, task='node_classification', trainer=None, use_best_config=False, weight_decay=0.0005)
Epoch: 499, Train: 0.9328, Val: 0.9374, ValLoss: 0.3098: 100%|█████████████████████| 500/500 [02:16<00:00,  3.67it/s]
Valid accurracy =  0.9374
Test accuracy = 0.9374
| Variant                | Acc           | ValAcc        |
|------------------------|---------------|---------------|
| ('rd2cd_Alpha', 'gat') | 0.9374±0.0000 | 0.9374±0.0000 |
Namespace(actnn=False, alpha=0.1, checkpoint=None, cpu=False, dataset='rd2cd_bgp', device_id=[3], dropout=0.5, fast_spmm=False, hidden_size=64, inference=False, lr=0.01, max_epoch=500, missing_rate=0, model='ppnp', num_classes=None, num_features=None, num_iterations=10, num_layers=2, patience=100, propagation_type='appnp', save_dir='.', save_model=None, seed=1, task='node_classification', trainer=None, use_best_config=False, weight_decay=0.0005)
Epoch: 117, Train: 0.2259, Val: 0.2167, ValLoss: 1.9389:  23%|████▉                | 117/500 [00:04<00:15, 24.34it/s]
Valid accurracy =  0.5420
Test accuracy = 0.5496
| Variant               | Acc           | ValAcc        |
|-----------------------|---------------|---------------|
| ('rd2cd_bgp', 'ppnp') | 0.5496±0.0000 | 0.5420±0.0000 |
Namespace(activation='relu', actnn=False, autoscale=False, checkpoint=None, correct_alpha=1.0, correct_norm='sym', cpu=False, dataset='rd2cd_Elliptic', device_id=[3], dropout=0.5, fast_spmm=False, hidden_size=16, inference=False, lr=0.01, max_epoch=500, missing_rate=0, model='correct_smooth_mlp', norm=None, num_classes=None, num_correct_prop=50, num_features=None, num_layers=2, num_smooth_prop=50, patience=100, save_dir='.', save_model=None, scale=1.0, seed=1, smooth_alpha=0.8, smooth_norm='row', task='node_classification', trainer=None, use_best_config=False, use_embeddings=False, weight_decay=0.0005)
Epoch: 499, Train: 0.8822, Val: 0.8790, ValLoss: 0.3318: 100%|█████████████████████| 500/500 [00:14<00:00, 34.46it/s]
Valid accurracy =  0.8798
Correct and Smoothing...
Test accuracy = 0.7977
| Variant                                  | Acc           | ValAcc        |
|------------------------------------------|---------------|---------------|
| ('rd2cd_Elliptic', 'correct_smooth_mlp') | 0.7977±0.0000 | 0.7976±0.0000 |
Namespace(activation='relu', actnn=False, checkpoint=None, cpu=False, dataset='rd2cd_Github', device_id=[3], dropout=0.5, fast_spmm=False, hidden_size=16, inference=False, lr=0.01, max_epoch=500, missing_rate=0, model='mlp', norm=None, num_classes=None, num_features=None, num_layers=2, patience=100, save_dir='.', save_model=None, seed=1, task='node_classification', trainer=None, use_best_config=False, weight_decay=0.0005)
Epoch: 188, Train: 0.9211, Val: 0.8638, ValLoss: 0.3497:  38%|█▉   | 188/500 [00:04<00:08, 38.20it/s]
Valid accurracy =  0.8675
Test accuracy = 0.8603
| Variant                 | Acc           | ValAcc        |
|-------------------------|---------------|---------------|
| ('rd2cd_Github', 'mlp') | 0.8603±0.0000 | 0.8675±0.0000 |
Namespace(activation='relu', actnn=False, aggr='mean', checkpoint=None, cpu=False, dataset='rd2cd_ssn5', device_id=[3], dropout=0.5, fast_spmm=False, hidden_size=128, inference=False, lr=0.01, max_epoch=500, missing_rate=0, model='sage', norm=None, normalize=False, num_layers=2, patience=100, save_dir='.', save_model=None, seed=1, task='node_classification', trainer=None, use_best_config=False, weight_decay=0.0005)
Epoch: 499, Train: 0.8378, Val: 0.8358, ValLoss: 0.6071: 100%|█████████████████████| 500/500 [01:15<00:00,  6.64it/s]
Valid accurracy =  0.8368
Test accuracy = 0.8414
| Variant                | Acc           | ValAcc        |
|------------------------|---------------|---------------|
| ('rd2cd_ssn5', 'sage') | 0.8414±0.0000 | 0.8368±0.0000 |
"""