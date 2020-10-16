import numpy as np
from typing import Tuple, List
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors import Predictor

@Predictor.register("bert_semantic_matching")
class SemanticMatchingPredictor(Predictor):

    def predict(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        output_dict = self.predict_instance(instance)
        outputs = {}
        outputs["predicted_labels"] = output_dict["predicted_labels"]
        if "true_labels" in inputs:
            outputs["true_labels"] = inputs["true_labels"]
        return outputs

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance = self._dataset_reader.text_to_instance(json_dict["sent1"], json_dict["sent2"])
        return instance