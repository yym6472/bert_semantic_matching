from typing import Optional, Iterator, List, Dict, Tuple

import os
import json
import logging
import random
import numpy as np
import pandas as pd
from pytorch_transformers import AutoTokenizer

from allennlp.data import Instance
from allennlp.data.fields import ArrayField, LabelField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary


logger = logging.getLogger(__name__)


@DatasetReader.register("semantic_matching")
class SemanticMatchingDatasetReader(DatasetReader):
    def __init__(self,
                 model_name_or_path: str) -> None:
        super().__init__(lazy=False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    def text_to_instance(self, sent1: str, sent2: str, label: Optional[str] = None) -> Instance:
        encoded_ids = self.tokenizer.encode(sent1, sent2, add_special_tokens=True)
        padding_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        sent_pair_field = ArrayField(np.array(encoded_ids), padding_value=padding_id, dtype=np.int)

        sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        first_sep_index = encoded_ids.index(sep_id)
        token_type_ids = [0] * (first_sep_index + 1) + [1] * (len(encoded_ids) - first_sep_index - 1)
        token_type_ids_field = ArrayField(np.array(token_type_ids), padding_value=0, dtype=np.int)

        fields = {"input_ids": sent_pair_field, "token_type_ids": token_type_ids_field}
        if label:
            label_field = LabelField(label=label)
            fields["labels"] = label_field
        return Instance(fields)
    
    def _read(self, file_path: str) -> Iterator[Instance]:
        df = pd.read_csv(file_path)
        for line_idx in range(len(df)):
            sentence1 = str(df.loc[line_idx, "sentence1"]).strip()
            sentence2 = str(df.loc[line_idx, "sentence2"]).strip()
            label = str(df.loc[line_idx, "label"]).strip()
            if not sentence1 or not sentence2 or not label:
                continue
            yield self.text_to_instance(sentence1, sentence2, label)


if __name__ == "__main__":
    reader = SemanticMatchingDatasetReader("bert-base-chinese")
    instances = reader.read("./data/lcqmc/LCQMC_test.csv")
    print(instances[0].fields["token_ids"].array)