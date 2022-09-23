import json
import os
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import OrderedDict

import numpy as np
import torch
from tqdm.auto import tqdm


def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--result_file', type=str, required=True)
    args = parser.parse_args()

    results = read_from_pickle(args.result_file)
    for result in results:
        for key in result:
            print('{}\t{:.4f}'.format(key, result[key].mean()))
