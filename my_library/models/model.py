from typing import Dict, Optional
from overrides import overrides

import numpy as np
import torch
import torch.nn.functional as F

from allennlp.nn import util
from allennlp.models import Model
from allennlp.data import Vocabulary
from allennlp.common.checks import check_dimensions_match
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.modules import FeedForward, TextFieldEmbedder
from allennlp.modules import Seq2SeqEncoder, Seq2VecEncoder
from allennlp.training.metrics.categorical_accuracy import CategoricalAccuracy

from my_library.metrics.multilabel_f1 import MultiLabelF1Measure

@Model.register("toxic_classifier")
class ToxicCommentClassifier(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            encoder.get_input_dim(),
            "text field embedding dim",
            "comment_text encoder input dim")
        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        self.encoder = encoder
        self.classifier_feedforward = classifier_feedforward
        self.f1 = MultiLabelF1Measure()
        self.loss = torch.nn.MultiLabelSoftMarginLoss()
        initializer(self)

    @overrides
    def forward(self,
                text: Dict[str, torch.Tensor],
                labels: torch.FloatTensor = None) -> Dict[str, torch.Tensor]:
        embedded_text = self.text_field_embedder(text)
        mask = util.get_text_field_mask(text)
        encoded_text = self.encoder(embedded_text, mask)

        logits = self.classifier_feedforward(encoded_text)
        probabilities = torch.sigmoid(logits)

        output_dict = {"logits": logits,
                       "probabilities": probabilities}
        if labels is not None:
            loss = self.loss(logits, labels.squeeze(-1).float())
            output_dict["loss"] = loss
            predictions = (logits.data > 0.0).long()
            label_data = labels.squeeze(-1).data.long()
            self.f1(predictions, label_data)
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        precision, recall, f1 = self.f1.get_metric(reset)
        return {
            'precision' : precision,
            'recall' : recall,
            'f1' : f1
        }