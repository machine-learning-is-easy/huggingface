from transformers import TFBertModel, TFBertPreTrainedModel, TFDistilBertPreTrainedModel, TFDistilBertModel
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import TFDistilBertForMaskedLM

from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Dense, Dropout
from tensorflow_addons.layers import CRF
import tensorflow as tf


class BertCRF(TFDistilBertForMaskedLM):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # self.bert = TFBertModel(config, add_pooling_layer=False)
        self.distillbert = TFDistilBertModel(config)
        self.dropout = Dropout(0.2)
        self.classifier = Dense(self.num_labels)
        self.crf = CRF(units=config.num_labels)

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        # token_type_ids=None,
        # position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.distillbert(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        decoded_sequence, potentials, sequence_length, chain_kernel = self.crf(logits)
        tags = tf.convert_to_tensor(decoded_sequence)

        if not return_dict:
            output = (tags,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return potentials, tags

    def from_pretrained(self, model_name, config):
        self.distillbert = TFDistilBertForMaskedLM.from_pretrained(model_name, config=config)
