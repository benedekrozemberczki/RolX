from parser import parameter_parser
from rolx import ROLX

def create_and_run_model(args):
    """
    Function to read the graph, create an embedding and train it.
    """
    model = ROLX(args)   
    model.train()

if __name__ == "__main__":
    args = parameter_parser()
    create_and_run_model(args)
