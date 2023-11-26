from util import check_path
from runmodel import RunModel
import logging
import torch
import random
import numpy as np
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    import argparse
    same_seeds(11111)
    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--db', default='yaleb', choices=['coil20', 'coil100', 'orl','yaleb'])
    args = parser.parse_args()
    print(args)

    db = args.db
    check_path()
    logging.basicConfig(level=logging.DEBUG, filename='./results/' + db + '.log', filemode='a')

    run = RunModel(db)
    run.train_dsc()
