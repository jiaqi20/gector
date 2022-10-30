from typing import Dict, Tuple, Optional, List, Any, cast

import torch
from torch.nn.modules.linear import Linear

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import (
    ConditionalRandomField,
    FeedForward,
)
from gector.crf_emission import (
    ConditionalRandomFieldWeightEmission
)
# from allennlp.modules.conditional_random_field.conditional_random_field import allowed_transitions
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure
from gector.beta import FBetaVerboseMeasure

def allowed_transitions(constraint_type: str, labels: Dict[int, str]) -> List[Tuple[int, int]]:
    """
    Given labels and a constraint type, returns the allowed transitions. It will
    additionally include transitions for the start and end states, which are used
    by the conditional random field.
    # Parameters
    constraint_type : `str`, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    labels : `Dict[int, str]`, required
        A mapping {label_id -> label}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()
    # Returns
    `List[Tuple[int, int]]`
        The allowed transitions (from_label_id, to_label_id).
    """
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [(start_tag, "START"), (end_tag, "END")]

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed(constraint_type, from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed


def is_transition_allowed(
    constraint_type: str, from_tag: str, from_entity: str, to_tag: str, to_entity: str
):
    """
    Given a constraint type and strings `from_tag` and `to_tag` that
    represent the origin and destination of the transition, return whether
    the transition is allowed under the given constraint type.
    # Parameters
    constraint_type : `str`, required
        Indicates which constraint to apply. Current choices are
        "BIO", "IOB1", "BIOUL", and "BMES".
    from_tag : `str`, required
        The tag that the transition originates from. For example, if the
        label is `I-PER`, the `from_tag` is `I`.
    from_entity : `str`, required
        The entity corresponding to the `from_tag`. For example, if the
        label is `I-PER`, the `from_entity` is `PER`.
    to_tag : `str`, required
        The tag that the transition leads to. For example, if the
        label is `I-PER`, the `to_tag` is `I`.
    to_entity : `str`, required
        The entity corresponding to the `to_tag`. For example, if the
        label is `I-PER`, the `to_entity` is `PER`.
    # Returns
    `bool`
        Whether the transition is allowed under the given `constraint_type`.
    """

    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if constraint_type == "BIOUL":
        if from_tag == "START":
            return to_tag in ("O", "B", "U")
        if to_tag == "END":
            return from_tag in ("O", "L", "U")
        return any(
            [
                # O can transition to O, B-* or U-*
                # L-x can transition to O, B-*, or U-*
                # U-x can transition to O, B-*, or U-*
                from_tag in ("O", "L", "U") and to_tag in ("O", "B", "U"),
                # B-x can only transition to I-x or L-x
                # I-x can only transition to I-x or L-x
                from_tag in ("B", "I") and to_tag in ("I", "L") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BIO":
        if from_tag == "START":
            return to_tag in ("O", "B")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or B-x
                to_tag in ("O", "B"),
                # Can only transition to I-x from B-x or I-x
                to_tag == "I" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "IOB1":
        if from_tag == "START":
            return to_tag in ("O", "I")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        return any(
            [
                # Can always transition to O or I-x
                to_tag in ("O", "I"),
                # Can only transition to B-x from B-x or I-x, where
                # x is the same tag.
                to_tag == "B" and from_tag in ("B", "I") and from_entity == to_entity,
            ]
        )
    elif constraint_type == "BMES":
        if from_tag == "START":
            return to_tag in ("B", "S")
        if to_tag == "END":
            return from_tag in ("E", "S")
        return any(
            [
                # Can only transition to B or S from E or S.
                to_tag in ("B", "S") and from_tag in ("E", "S"),
                # Can only transition to M-x from B-x, where
                # x is the same tag.
                to_tag == "M" and from_tag in ("B", "M") and from_entity == to_entity,
                # Can only transition to E-x from B-x or M-x, where
                # x is the same tag.
                to_tag == "E" and from_tag in ("B", "M") and from_entity == to_entity,
            ]
        )
    else:
        raise ConfigurationError(f"Unknown constraint type: {constraint_type}")

@Model.register("crf_taggerr")
class CrfTaggerr(Model):
    """
    The `CrfTagger` encodes a sequence of text with a `Seq2SeqEncoder`,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.
    Registered as a `Model` with name "crf_tagger".
    # Parameters
    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the tokens `TextField` we get as input to the model.
    encoder : `Seq2SeqEncoder`
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : `str`, optional (default=`labels`)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    feedforward : `FeedForward`, optional, (default = `None`).
        An optional feedforward layer to apply after the encoder.
    label_encoding : `str`, optional (default=`None`)
        Label encoding to use when calculating span f1 and constraining
        the CRF at decoding time . Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if `calculate_span_f1` or `constrain_crf_decoding` is true.
    include_start_end_transitions : `bool`, optional (default=`True`)
        Whether to include start and end transition parameters in the CRF.
    constrain_crf_decoding : `bool`, optional (default=`None`)
        If `True`, the CRF is constrained at decoding time to
        produce valid sequences of tags. If this is `True`, then
        `label_encoding` is required. If `None` and
        label_encoding is specified, this is set to `True`.
        If `None` and label_encoding is not specified, it defaults
        to `False`.
    calculate_span_f1 : `bool`, optional (default=`None`)
        Calculate span-level F1 metrics during training. If this is `True`, then
        `label_encoding` is required. If `None` and
        label_encoding is specified, this is set to `True`.
        If `None` and label_encoding is not specified, it defaults
        to `False`.
    dropout:  `float`, optional (default=`None`)
        Dropout probability.
    verbose_metrics : `bool`, optional (default = `False`)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    top_k : `int`, optional (default=`1`)
        If provided, the number of parses to return from the crf in output_dict['top_k_tags'].
        Top k parses are returned as a list of dicts, where each dictionary is of the form:
        {"tags": List, "score": float}.
        The "tags" value for the first dict in the list for each data_item will be the top
        choice, and will equal the corresponding item in output_dict['tags']
    ignore_loss_on_o_tags : `bool`, optional (default=`False`)
        If True, we compute the loss only for actual spans in `tags`, and not on `O` tokens.
        This is useful for computing gradients of the loss on a _single span_, for
        interpretation / attacking.
    label_weights : `Dict[str, float]`, optional (default=`None`)
        A mapping {label : weight} to be used in the loss function in order to
        give different weights for each token depending on its label. This is useful to
        deal with highly unbalanced datasets. There are three available strategies to deal
        with weighted labels (see below). The default strategy is "emission".
    weight_strategy : `str`, optional (default=`None`)
        If `label_weights` is given and this is `None`, then it is the same as "emission".
        It indicates which strategy is used to sample weighting. Valid options are:
        "emission", "emission_transition", "lannoy". If "emission" then the emission score
        of each tag is multiplied by the corresponding weight (as given by `label_weights`).
        If "emission_transition", both emission and transition scores of each tag are multiplied
        by the corresponding weight. In this case, a transition score `t(i,j)`, between consecutive
        tokens `i` and `j`, is multiplied by `w[tags[i]]`, i.e., the weight related to the tag of token `i`.
        If `weight_strategy` is "lannoy" then we use the strategy proposed by
        [Lannoy et al. (2019)](https://perso.uclouvain.be/michel.verleysen/papers/ieeetbe12gdl.pdf).
        You can see an experimental comparison among these three strategies and a brief discussion
        of their differences [here](https://eraldoluis.github.io/2022/05/10/weighted-crf.html).
    """

    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        encoder: Seq2SeqEncoder,
        label_namespace: str = "labels",
        feedforward: Optional[FeedForward] = None,
        label_encoding: Optional[str] = None,
        include_start_end_transitions: bool = True,
        constrain_crf_decoding: bool = None,
        calculate_span_f1: bool = None,
        dropout: Optional[float] = None,
        verbose_metrics: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        top_k: int = 1,
        ignore_loss_on_o_tags: bool = False,
        label_weights: Optional[Dict[str, float]] = None,
        weight_strategy: str = None,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.label_namespace = label_namespace
        self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(label_namespace)
        self.encoder = encoder
        self.top_k = top_k
        self.ignore_loss_on_o_tags = ignore_loss_on_o_tags
        self._verbose_metrics = verbose_metrics
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._feedforward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = self.encoder.get_output_dim()
        self.tag_projection_layer = TimeDistributed(Linear(output_dim, self.num_tags))

        # if  constrain_crf_decoding and calculate_span_f1 are not
        # provided, (i.e., they're None), set them to True
        # if label_encoding is provided and False if it isn't.
        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.label_encoding = label_encoding
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError(
                    "constrain_crf_decoding is True, but no label_encoding was specified."
                )
            labels = self.vocab.get_index_to_token_vocabulary(label_namespace)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None

        # Label weights are given as a dict {label: weight} but we convert it to a list of weights for each label,
        # and weights for omitted labels are set to 1.
        if label_weights is None:
            if weight_strategy is not None:
                raise ConfigurationError(
                    "`weight_strategy` can only be used when `label_weights` is given"
                )

            # ordinary CRF (not weighted)
            self.crf = ConditionalRandomField(
                self.num_tags,
                constraints,
                include_start_end_transitions,
            )
        else:  # label_weights is not None
            label_to_index = vocab.get_token_to_index_vocabulary(label_namespace)
            self.label_weights = [1.0] * len(label_to_index)
            for label, weight in label_weights.items():
                try:
                    self.label_weights[label_to_index[label]] = weight
                except KeyError:
                    raise ConfigurationError(
                        f"'{label}' not found in vocab namespace '{label_namespace}')"
                    )

            if weight_strategy is None or weight_strategy == "emission":
                self.crf = ConditionalRandomFieldWeightEmission(
                    self.num_tags,
                    self.label_weights,
                    constraints,
                    include_start_end_transitions,
                )
            elif weight_strategy == "emission_transition":
                self.crf = ConditionalRandomFieldWeightTrans(
                    self.num_tags,
                    self.label_weights,
                    constraints,
                    include_start_end_transitions,
                )
            elif weight_strategy == "lannoy":
                self.crf = ConditionalRandomFieldWeightLannoy(
                    self.num_tags,
                    self.label_weights,
                    constraints,
                    include_start_end_transitions,
                )
            else:
                raise ConfigurationError(
                    "weight_strategy must be one of 'emission', 'emission_transition' or 'lannoy'"
                )

        self.include_start_end_transitions = include_start_end_transitions

        self.metrics = {
            "accuracy": CategoricalAccuracy(),
            "accuracy3": CategoricalAccuracy(top_k=3),
        }

        self.calculate_span_f1 = calculate_span_f1
        if calculate_span_f1:
            if not label_encoding:
                raise ConfigurationError(
                    "calculate_span_f1 is True, but no label_encoding was specified."
                )
            self._f1_metric = SpanBasedF1Measure(
                vocab, tag_namespace=label_namespace, label_encoding=label_encoding
            )
        elif verbose_metrics:
            # verbose metrics for token classification (not span-based)
            self._f_beta_measure = FBetaVerboseMeasure(
                index_to_label=vocab.get_index_to_token_vocabulary(label_namespace),
            )

        check_dimensions_match(
            text_field_embedder.get_output_dim(),
            encoder.get_input_dim(),
            "text field embedding dim",
            "encoder input dim",
        )
        if feedforward is not None:
            check_dimensions_match(
                encoder.get_output_dim(),
                feedforward.get_input_dim(),
                "encoder output dim",
                "feedforward input dim",
            )
        initializer(self)

    def forward(
        self,  # type: ignore
        tokens: Dict[str, torch.LongTensor],
        labels: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        ignore_loss_on_o_tags: Optional[bool] = None,
        **kwargs,  # to allow for a more general dataset reader that passes args we don't need
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters
        tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. This output is a dictionary mapping keys to `TokenIndexer`
            tensors.  At its most basic, using a `SingleIdTokenIndexer` this is : `{"tokens":
            Tensor(batch_size, num_tokens)}`. This dictionary will have the same keys as were used
            for the `TokenIndexers` when you created the `TextField` representing your
            sequence.  The dictionary is designed to be passed directly to a `TextFieldEmbedder`,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels of shape
            `(batch_size, num_tokens)`.
        metadata : `List[Dict[str, Any]]`, optional, (default = `None`)
            metadata containing the original words in the sentence to be tagged under a 'words' key.
        ignore_loss_on_o_tags : `Optional[bool]`, optional (default = `None`)
            If True, we compute the loss only for actual spans in `tags`, and not on `O` tokens.
            This is useful for computing gradients of the loss on a _single span_, for
            interpretation / attacking.
            If `None`, `self.ignore_loss_on_o_tags` is used instead.
        # Returns
        An output dictionary consisting of:
        logits : `torch.FloatTensor`
            The logits that are the output of the `tag_projection_layer`
        mask : `torch.BoolTensor`
            The text field mask for the input tokens
        tags : `List[List[int]]`
            The predicted tags using the Viterbi algorithm.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised. Only computed if gold label `tags` are provided.
        """
        ignore_loss_on_o_tags = (
            ignore_loss_on_o_tags
            if ignore_loss_on_o_tags is not None
            else self.ignore_loss_on_o_tags
        )
        embedded_text_input = self.text_field_embedder(tokens)
        mask = util.get_text_field_mask(tokens)

        if self.dropout:
            embedded_text_input = self.dropout(embedded_text_input)

        encoded_text = self.encoder(embedded_text_input, mask)

        if self.dropout:
            encoded_text = self.dropout(encoded_text)

        if self._feedforward is not None:
            encoded_text = self._feedforward(encoded_text)

        logits = self.tag_projection_layer(encoded_text)
        # best_paths = self.crf.viterbi_tags(logits, mask, top_k=self.top_k)
        best_paths = self.crf.viterbi_tags(logits, mask)
        # print(best_paths)
        # Just get the top tags and ignore the scores.
        predicted_tags = []
        
        for x in best_paths:
            predicted_tags.append([x[0][0]])
        # cast(List[List[int]], [x[0][0] for x in best_paths])

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        if self.top_k > 1:
            output["top_k_tags"] = best_paths

        if labels is not None:
            if ignore_loss_on_o_tags:
                o_tag_index = self.vocab.get_token_index("O", namespace=self.label_namespace)
                crf_mask = mask & (labels != o_tag_index)
            else:
                crf_mask = mask
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(logits, labels, crf_mask)

            output["loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.0
            #print(predicted_tags)
            for i, instance_tags in enumerate(predicted_tags):
                #print(instance_tags)
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, labels, mask)
            if self.calculate_span_f1:
                self._f1_metric(class_probabilities, labels, mask)
            elif self._verbose_metrics:
                self._f_beta_measure(class_probabilities, labels, mask)
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]
        return output

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        `output_dict["tags"]` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """

        def decode_tags(tags):
            return [
                self.vocab.get_token_from_index(tag, namespace=self.label_namespace) for tag in tags
            ]

        def decode_top_k_tags(top_k_tags):
            return [
                {"tags": decode_tags(scored_path[0]), "score": scored_path[1]}
                for scored_path in top_k_tags
            ]

        output_dict["tags"] = [decode_tags(t) for t in output_dict["tags"]]

        if "top_k_tags" in output_dict:
            output_dict["top_k_tags"] = [decode_top_k_tags(t) for t in output_dict["top_k_tags"]]

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics_to_return = {
            metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()
        }

        if self.calculate_span_f1:
            f1_dict = self._f1_metric.get_metric(reset=reset)
            if self._verbose_metrics:
                metrics_to_return.update(f1_dict)
            else:
                metrics_to_return.update({x: y for x, y in f1_dict.items() if "overall" in x})
        elif self._verbose_metrics:
            # verbose metrics for token classification (not span-based)
            f_beta_dict = self._f_beta_measure.get_metric(reset=reset)
            metrics_to_return.update(f_beta_dict)

        return metrics_to_return

    default_predictor = "sentence_tagger"