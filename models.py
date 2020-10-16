from typing import Dict, Optional, List, Any

import os
import torch
import logging
import collections

from allennlp.common import Params
from allennlp.models import Model
from allennlp.modules import TimeDistributed
from allennlp.modules.text_field_embedders.basic_text_field_embedder import BasicTextFieldEmbedder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.token_embedders.embedding import Embedding
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import PytorchSeq2SeqWrapper
from allennlp.modules.conditional_random_field import ConditionalRandomField, allowed_transitions
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from pytorch_transformers import AutoModel

logger = logging.getLogger(__name__)


@Model.register("bert_semantic_matching")
class SemanticMatching(Model):
    def __init__(self, 
                 vocab: Vocabulary,
                 model_name_or_path: str,
                 bert_trainable: bool = False,
                 encoder: Optional[Seq2VecEncoder] = None,
                 dropout: Optional[float] = None) -> None:
        super().__init__(vocab)

        self.bert_model = AutoModel.from_pretrained(model_name_or_path)
        self.encoder = encoder

        if encoder:
            hidden2label_in_dim = encoder.get_output_dim()
        else:
            hidden2label_in_dim = self.bert_model.hidden_size
        self.hidden2label = torch.nn.Linear(
            in_features=hidden2label_in_dim,
            out_features=vocab.get_vocab_size("labels")
        )
        
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.acc = CategoricalAccuracy()
        
        if bert_trainable:
            for param in self.bert_model.parameters():
                param.requires_grad_(True)
        else:
            for param in self.bert_model.parameters():
                param.requires_grad_(False)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self,
                input_ids: torch.Tensor,
                token_type_ids: torch.Tensor,
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        output = {}

        mask = (input_ids != 0).long()
        output["mask"] = mask

        encoded, pooled = self.bert_model(input_ids, token_type_ids, mask)
        if self.dropout:
            encoded = self.dropout(encoded)
            pooled = self.dropout(pooled)
        output["encoded"] = encoded
        output["pooled"] = pooled

        if self.encoder:
            sent_vectors = self.encoder(encoded, mask)
            if self.dropout:
                sent_vectors = self.dropout(sent_vectors)
            output["sent_vectors"] = sent_vectors
        else:
            sent_vectors = pooled

        logits = self.hidden2label(sent_vectors)
        output["logits"] = logits
        output["predicted_labels"] = [self.vocab.get_token_from_index(index, "labels") for index in logits.argmax(dim=-1).tolist()]

        if labels is not None:
            self.acc(logits, labels)
            loss = self.loss_func(logits, labels)
            output["loss"] = loss
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        acc = self.acc.get_metric(reset)
        return {"acc": acc}