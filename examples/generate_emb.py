import numpy as np
from cogdl import pipeline

# build a pipeline for generating embeddings
# pass model name with its hyper-parameters to this API
generator = pipeline("generate-emb", model="prone")

# generate embedding by an unweighted graph
edge_index = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [2, 3]])
outputs = generator(edge_index)
print(outputs)

# generate embeddings by a weighted graph
edge_weight = np.array([0.1, 0.3, 1.0, 0.8, 0.5])
outputs = generator(edge_index, edge_weight)
print(outputs)
