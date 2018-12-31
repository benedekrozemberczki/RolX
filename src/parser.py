import argparse

def parameter_parser():

    """
    A method to parse up command line parameters. By default it gives an embedding of the Facebook tvshow network.
    The default hyperparameters give a good quality representation and good candidate cluster means without grid search.
    """

    parser = argparse.ArgumentParser(description = "Run RolX.")

    #------------------------------------------------------------------
    # Input and output file parameters.
    #------------------------------------------------------------------

    parser.add_argument("--input",
                        nargs = "?",
                        default = "./input/tvshow_edges.csv",
	                help = "Input graph path.")

    parser.add_argument("--recursive-features-output",
                        nargs = "?",
                        default = "./output/features/tvhsow_features.csv",
	                help = "Embeddings path.")

    parser.add_argument("--embedding-output",
                        nargs = "?",
                        default = "./output/embeddings/tvshow_embedding.csv",
	                help = "Embeddings path.")

    parser.add_argument("--log-output",
                        nargs = "?",
                        default = "./output/logs/tvshow_log.json",
    	                 help = "Log path.")

    #-----------------------------------------------------------------------
    # Recursive feature extraction parameters.
    #-----------------------------------------------------------------------

    parser.add_argument("--recursive-iterations",
                        type = int,
                        default = 3,
	                help = "Number of recursions.")

    parser.add_argument("--aggregator",
                        nargs = "?",
                        default = "simple",
    	                 help = "Aggregator statistics extracted.")

    parser.add_argument("--bins",
                        type = int,
                        default = 4,
	                help = "Number of quantization bins.")

    parser.add_argument("--pruning-cutoff",
                        type = float,
                        default = 0.5,
	                help = "Absolute correlation for feature pruning.")

    #------------------------------------------------------------------
    # Factor model parameters.
    #------------------------------------------------------------------

    parser.add_argument("--dimensions",
                        type = int,
                        default = 16,
	                help = "Number of dimensions. Default is 16.")

    parser.add_argument("--batch-size",
                        type = int,
                        default = 32,
	                help = "Number of edges in batch. Default is 128.")

    parser.add_argument("--epochs",
                        type = int,
                        default = 5,
	                help = "Number of epochs. Default is 50.")

    parser.add_argument("--initial-learning-rate",
                        type = float,
                        default = 0.01,
	                help = "Initial learning rate. Default is 0.01.")

    parser.add_argument("--minimal-learning-rate",
                        type = float,
                        default = 0.001,
	                help = "Minimal learning rate. Default is 0.001.")

    parser.add_argument("--annealing-factor",
                        type = float,
                        default = 1,
	                help = "Annealing factor. Default is 1.0.")
    
    return parser.parse_args()
