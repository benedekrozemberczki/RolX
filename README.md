ReFeX and RolX
===============================================
![License](https://img.shields.io/github/license/benedekrozemberczki/RolX.svg?color=blue&style=plastic)
[![codebeat badge](https://codebeat.co/badges/f688b042-0641-4aa7-a122-9719e3372ca9)](https://codebeat.co/projects/github-com-benedekrozemberczki-rolx-master)
<p align="justify">
ReFex is a structural graph feature extraction algorithm which creates binary features which describe structural properties of nodes in a large graph. First, continuous features are extracted based on descriptive statistics of neighbourhoods. These statistics are aggregated recursively. The original algorithm was extended in this implementation in such way that more advanced descriptive statistics can be extracted during the recursion phase. In addition, the number of feature extraction recursions and the binary binning also have controllable parameters. Finally, the strongly correlated features can be dropped based on an arbitrarily  chosen threshold.
</p>

<p align="center">
  <img width="720" src="structural.jpeg">
</p>

<p align="justify">
RolX is an algorithm which takes features extracted with ReFeX and factorizes the binary node-feature matrix in order to create low dimensional structural node representations. Nodes with similar structural features will be clustered together in the latent space. The original model uses non-negative matrix factorization, in our work we use an implicit matrix factorization model which is trained with a potent variant of gradient descent. Our implementation supports GPU use.
</p>

This repository provides a custom implementation for ReFex and RolX as described in the papers:

> It's who you know: graph mining using recursive structural features.
> Keith Henderson, Brian Gallagher, Lei Li, Leman Akoglu, Tina Eliassi-Rad, Hanghang Tong and Christos Faloutsos.
> Proceedings of the 17th ACM SIGKDD international conference on Knowledge discovery and data mining.
> [[Paper]](http://www.cs.cmu.edu/~leili/pubs/henderson-kdd2011.pdf)


> RolX: Structural Role Extraction & Mining in Large Graphs
> Keith Henderson, Brian Gallagher, Tina Eliassi-Rad, Hanghang Tong, Sugato Basu, Leman Akoglu, Danai Koutra, Christos Faloutsos and Lei Li.
> Proceedings of the 18th ACM SIGKDD international conference on Knowledge discovery and data mining.
> [[Paper]](https://web.eecs.umich.edu/~dkoutra/papers/12-kdd-recursiverole.pdf)

Another Python implementation is available [[here]](https://github.com/dkaslovsky/GraphRole).

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

<p align="justify">
The code takes an input graph in a csv file. Every row indicates an edge between two nodes separated by a comma. The first row is a header. Nodes should be indexed starting with 0. A sample graph for the `Facebook TVshows` dataset is included in the  `data/` directory.</p>

### Logging

<p align="justify">
The models are defined in a way that parameter settings, extracted features and factorization loss are logged in every single epoch. Specifically we log the followings:</p>

```
1. Hyperparameter settings.                  We save each hyperparameter used in the experiment.
3. Number of extracted features per epoch.   We take the number of features before and after pruning.
2. Cost per epoch.                           Reconstruction cost is stored in every iteration.
4. Runtime.                                  We measure the time needed for feature extraction and optimization -- measured by seconds.
```

### Options

<p align="justify">
The feature extraction and factorization are handled by the `src/main.py` script which provides the following command line arguments.</p>

#### Input and output options

```
  --input                        STR   Input graph path.           Default is `data/tvshow_edges.csv`.
  --embedding-output             STR   Embeddings path.            Default is `output/embeddings/tvhsow_embedding.csv`.
  --recursive-features-output    STR   Recursive features path.    Default is `output/features/tvhsow_features.csv`.
  --log-output                   STR   Log path.                   Default is `output/logs/tvhsow.log`.
```

#### ReFeX options

```
  --recursive-iterations  INT      Number of recursions.                                Default is 3.
  --bins                  INT      Number of binarization bins.                         Default is 4.
  --aggregator            STR      Aggregation strategy (simple/complex).               Default is `simple`.
  --pruning-cutoff        FLOAT    Absolute correlation for feature dropping.           Default is 0.9.
```

#### RolX options

```
  --epochs                  INT       Number of epochs.                           Default is 10.
  --batch-size              INT       Number of edges in batch.                   Default is 32.
  --dimensions              INT       Number of dimensions.                       Default is 16.
  --initial-learning-rate   FLOAT     Initial learning rate.                      Default is 0.01.
  --minimal-learning-rate   FLOAT     Final learning rate.                        Default is 0.001.
  --annealing-factor        FLOAT     Annealing factor for learning rate.         Default is 1.0.
  --lambd                   FLOAT     Weight regularization penalty.              Default is 10**-3.
```

### Examples

<p align="justify">
The following commands create structural features, learn a graph embedding and write these to disk. The node representations are ordered by the ID.</p>

<p align="justify">
Creating a RolX embedding of the default dataset with the default hyperparameter settings. Saving the ReFeX features, RolX embedding and the log file at the default path.</p>

```
python src/main.py
```
Creating an embedding of an other dataset the `Facebook Companies`. Saving the output and the log in a custom place.

```
python src/main.py --input data/company_edges.csv  --embedding-output output/embeddings/company_embedding.csv --recursive-features-output output/features/company_features.csv --log-output output/logs/company_log.json
```

Creating an embedding of the default dataset in 128 dimensions with 8 binary feature bins.

```
python src/main.py --dimensions 128 --bins 8
```
