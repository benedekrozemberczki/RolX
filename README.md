# ReFeX and RolX
Recursive structural feature and role extraction machine.
<p align="justify">
ReFex is a structural graph feature extraction algorithm which creates binary features which describe structural properties of nodes in a large graph. First, continuous features are extracted based on descriptive statistics of neighbourhoods. These statistics are aggregated recursively. The original algorithm was extended in this implementation in such way that more advanced descriptive statistics can be extracted during the recursion phase. In addition, the number of feature extraction recursions and the binary binning also have controllable parameters. Finally, the strongly correlated features can be dropped based on an arbitrarily  chosen threshold.
  
RolX is an algorithm which takes features extracted with ReFeX and factorizes the binary node-feature matrix in order to create low dimensional structural node representations. Nodes with similar structural features will be clustered together in the latent space. The original model uses non-negative matrix factorization, in our work we use an implicit matrix factorization model which is trained with a potent variant of gradient descent. Our implementation supports GPU use.
</p>

This repository provides a custom implementation for ReFex and RolX as described in the papers:

> It's who you know: graph mining using recursive structural features.
> Keith Henderson, Brian Gallagher, Lei Li, Leman Akoglu, Tina Eliassi-Rad, Hanghang Tong and Christos Faloutsos.
> Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining.
> http://www.cs.cmu.edu/~leili/pubs/henderson-kdd2011.pdf


> It's who you know: graph mining using recursive structural features.
> Keith Henderson, Brian Gallagher, Lei Li, Leman Akoglu, Tina Eliassi-Rad, Hanghang Tong and Christos Faloutsos.
> Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining.
> http://www.cs.cmu.edu/~leili/pubs/henderson-kdd2011.pdf

### Requirements

The codebase is implemented in Python 2.7.
package versions used for development are just below.
```
networkx          1.11
tqdm              4.19.5
numpy             1.13.3
pandas            0.20.3
tensorflow-gpu    1.3.0
jsonschema        2.6.0
texttable         1.2.1
```

### Datasets

The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. A sample graph for the `Facebook Politicians` dataset is included in the  `data/` directory.

### Logging

The models are defined in a way that parameter settings and cluster quality is logged in every single epoch. Specifically we log the followings:

```
1. Hyperparameter settings.     We save each hyperparameter used in the experiment.
2. Cost per epoch.              Reconstruction cost is stored depending on the model type.
3. Cluster quality.             Measured by modularity. We calculate it based on k-means clustering per epoch.
4. Runtime.                     We measure the time needed for optimization -- measured by seconds.
```

### Options

Learning of the embedding is handled by the `src/factorizer.py` script which provides the following command line arguments.

#### Input and output options

```
  --input STR                   Input graph path.                                 Default is `data/politician_edges.csv`.
  --embedding-output STR        Embeddings path.                                  Default is `output/embeddings/politician_embedding.csv`.
  --cluster-mean-output         Cluster centers path.                             Default is `output/cluster_means/politician_means.csv`.
  --log-output STR              Log path.                                         Default is `output/logs/politician.log`.
  --assignment-output STR       Node-cluster assignment dictionary path.          Default is `output/assignments/politician.json`.
  --dump-matrices BOOL          Whether the trained model should be saved.        Default is `True`.
  --model STR                   Model used.                                       Default is `GRAF`.
```

#### ReFeX options

```
  --epochs INT                    Number of epochs.                                   Default is 10.
  --batch-size INT                Number of edges in batch.                           Default is 128.
  --target-weighting STR          Target edge weight strategy.                        Default is `overlap`.
  --regularization-weighting STR  Regularization weighing strategy.                   Default is `normalized_overlap`.
  --dimensions INT                Number of dimensions.                               Default is 16.
  --initial-learning-rate FLOAT   Initial learning rate.                              Default is 0.01.
  --minimal-learning-rate FLOAT   Final learning rate.                                Default is 0.001.
  --annealing-factor FLOAT        Annealing factor for learning rate.                 Default is 1.0.
  --lambd FLOAR                   Weight regularization penalty.                      Default is 2**-4.
  --cluster-number INT            Number of clusters.                                 Default is 20.
  --initial-gamma FLOAT           Initial clustering cost weight.                     Default is 0.1.
  --regularization-noise FLOAT    Gradient noise.                                     Default is 10**-8.
```

#### RolX options

```
  --epochs INT                    Number of epochs.                                   Default is 10.
  --batch-size INT                Number of edges in batch.                           Default is 128.
  --target-weighting STR          Target edge weight strategy.                        Default is `overlap`.
  --regularization-weighting STR  Regularization weighing strategy.                   Default is `normalized_overlap`.
  --dimensions INT                Number of dimensions.                               Default is 16.
  --initial-learning-rate FLOAT   Initial learning rate.                              Default is 0.01.
  --minimal-learning-rate FLOAT   Final learning rate.                                Default is 0.001.
  --annealing-factor FLOAT        Annealing factor for learning rate.                 Default is 1.0.
  --lambd FLOAR                   Weight regularization penalty.                      Default is 2**-4.
  --cluster-number INT            Number of clusters.                                 Default is 20.
  --initial-gamma FLOAT           Initial clustering cost weight.                     Default is 0.1.
  --regularization-noise FLOAT    Gradient noise.                                     Default is 10**-8.
```

### Examples

The following commands learn a graph embedding and cluster center and writes them to disk. The node representations are ordered by the ID.

Creating a GRAF embedding of the default dataset with the default hyperparameter settings. Saving the embedding, cluster centres and the log file at the default path.

```
python src/factorizer.py
```

Turning off the model saving.

```
python src/factorizer.py --dump-matrices False
```

Creating an embedding of an other dataset the `Facebook Companies`. Saving the output and the log in a custom place.

```
python src/factorizer.py --input data/company_edges.csv  --embedding-output output/embeddings/company_embedding.csv --log-output output/cluster_means/company_means.csv
```

Creating a clustered embedding of the default dataset in 128 dimensions and 10 cluster centers.

```
python src/factorizer.py --dimensions 128 --cluster-number 10
```
