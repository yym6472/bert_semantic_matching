import os
import tqdm
import argparse
import collections
from typing import Any, Union, Dict, Iterable, List, Optional, Tuple

from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from models import SemanticMatching
from predictors import SemanticMatchingPredictor
from dataset_readers import SemanticMatchingDatasetReader

from allennlp.data import vocabulary
vocabulary.DEFAULT_OOV_TOKEN = "[UNK]"  # set for bert


def main(args):
    archive = load_archive(args.output_dir)
    predictors = Predictor.from_archive(archive=archive, predictor_name="bert_semantic_matching")
    output = predictors.predict_json({"sent1": "谁有狂三这张高清的", "sent2": "这张高清图，谁有"})
    label = output["predicted_labels"]
    print(label)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--output_dir", type=str, default="./output/bert-lcqmc",
                            help="the directory that stores training output")
    args = arg_parser.parse_args()
    main(args)