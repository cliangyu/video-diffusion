from collections import defaultdict
from typing import OrderedDict
import torch
import numpy as np
from argparse import ArgumentParser
import os
from tqdm.auto import tqdm
from pathlib import Path
import json
import pickle
from collections import defaultdict

def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--result_file", type=str, required=True)
    args = parser.parse_args()

    results = read_from_pickle(args.result_file)
    for result in results:
        for key in result:
            print('{}\t{:.4f}'.format(key, result[key].mean()))