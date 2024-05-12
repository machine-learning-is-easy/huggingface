from transformers import BertTokenizer, BertForMaskedLM
from transformers import DistilBertTokenizer, DistilBertModel
import time
import os
from transformers import BertTokenizer, BertForMaskedLM
import torch
from torch import nn as nn
import copy
import numpy as np
# model_selected = "bert-base-uncased"
model_selected = "distilbert-base-uncased"
current_folder = os.path.dirname(os.path.abspath(__file__))

tokenizer = DistilBertTokenizer.from_pretrained(model_selected)
def add_token(token_list):
    for token in token_list:
        ids = len(tokenizer)
        tokenizer.add_tokens(token)
        tokenizer.ids_to_tokens.update({ids: token})
        tokenizer.vocab[token] = ids


def deleteEncodingLayers(model, num_layers_to_keep):  # must pass in the full bert model
    oldModuleList = model.bert.encoder.layer
    newModuleList = nn.ModuleList()

    # Now iterate over all layers, only keepign only the relevant layers.
    for i in range(0, num_layers_to_keep):
        newModuleList.append(oldModuleList[i])

    # create a copy of the model, modify it with the new list, and return
    copyOfModel = copy.deepcopy(model)
    copyOfModel.bert.encoder.layer = newModuleList

    return copyOfModel

def total_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

token_list = {"amazon", "meta", "apple"}
add_token(token_list)
add_id = set([tokenizer.vocab[token] for token in token_list])

model = BertForMaskedLM.from_pretrained(model_selected, cache_dir=os.path.join(current_folder, "model"))

model = deleteEncodingLayers(model, 3)
model.resize_token_embeddings(len(tokenizer))


with open(os.path.join(current_folder, 'clean.csv'), 'r') as fp:
    text = fp.read().split('\n')

inputs = tokenizer(text, return_tensors='pt', max_length=32, truncation=True, padding='max_length')
inputs['labels'] = inputs.input_ids.detach().clone()

def get_attention(inputs, dict):
    # create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(inputs.input_ids.shape)
    # create mask array
    mask_arr = (rand < 0)
    mask_arr = (inputs.input_ids != 101) * (inputs.input_ids != 102) * (inputs.input_ids != 0)

    tmp_mask_arr = (rand < 0)
    for ids in dict:
         tmp_mask_arr= torch.logical_or(tmp_mask_arr, (inputs.input_ids == ids))
    mask_arr = torch.logical_and(mask_arr, tmp_mask_arr)

    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

get_attention(inputs, add_id)

class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

dataset = MeditationsDataset(inputs)

loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device
model.to(device)
# activate training mode
model.train()

from transformers import AdamW
# initialize optimizer
optim = AdamW(model.parameters(), lr=1e-5)

from tqdm import tqdm  # for our progress bar

epochs = 100

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        # initialize calculated gradients (from prev step)
        optim.zero_grad()
        # pull all tensor batches required for training
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        # process
        outputs = model(input_ids, attention_mask=attention_mask,
                        labels=labels)
        # outputs = model(input_ids)
        # extract loss
        loss = outputs.loss
        # calculate loss for every parameter that needs grad update
        loss.backward()
        # update parameters
        optim.step()
        # print relevant info to progress bar
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())

input = ["[MASK] is a wonderful company"]
input = tokenizer(input, return_tensors='pt', max_length=32, truncation=True, padding='max_length')
time0 = time.time()
predict = model(input.input_ids, attention_mask=input["attention_mask"])
print("time for predict is {}".format(time.time() - time0))
prob = torch.softmax(predict.logits, dim=-1)


def print_pro(ind):
    print("index is {}".format(ind))
    print("meta prob is {}".format(prob[0, ind, tokenizer.vocab["meta"]]))
    print("is prob is {}".format(prob[0, ind, tokenizer.vocab["is"]]))
    print("wonderful prob is {}".format(prob[0, ind, tokenizer.vocab["wonderful"]]))
    print("Maximum index is {}".format(torch.argmax(prob, dim=-1)[0, ind]))

# save model
model.save_pretrained(os.path.join(current_folder, "model"))
tokenizer.save_pretrained(os.path.join(current_folder, "tokenizer"))


print_pro(0)
print("***************")
print_pro(1)
print("***************")
print_pro(2)
print("***************")
print_pro(3)
print("***************")
print_pro(4)