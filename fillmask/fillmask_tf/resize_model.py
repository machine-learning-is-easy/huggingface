import os
from transformers import TFDistilBertForMaskedLM
from torch import nn as nn
from transformers.models.distilbert import DistilBertConfig


current_folder = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(current_folder, "model")
import copy


model_selected = "distilbert-base-uncased"
model = TFDistilBertForMaskedLM.from_pretrained(model_selected)
model.summary()

config = DistilBertConfig()
config.n_layers = 3
# model = DistilBertModel.from_pretrained(model_selected)
model = TFDistilBertForMaskedLM.from_pretrained(model_selected, config=config)
model.summary()


def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.transformer.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(0, num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.transformer.layer = newModuleList

    return copyOfModel

a = 0