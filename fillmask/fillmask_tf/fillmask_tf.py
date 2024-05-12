# from transformers import DefaultDataCollator
import tensorflow as tf
import torch
from datasets import load_dataset
import os
from transformers import TFAutoModelForSequenceClassification, TFDistilBertForMaskedLM, TFBertForMaskedLM
from transformers import create_optimizer
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer

model_selected = "distilbert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
# #defining tokenizer
current_folder = os.path.dirname(os.path.abspath(__file__))
#
# add_id = set([tokenizer.vocab[token] for token in token_list])

# pyspark processing the data
with open(os.path.join(current_folder, '../clean.csv'), 'r') as fp:
    text = fp.read().split('\n')

inputs = tokenizer(text, return_tensors='pt', max_length=32, truncation=True, padding='max_length')
inputs['labels'] = inputs.input_ids.detach().clone()

def get_attention(inputs, dict):
    # create random array of floats with equal dimensions to input_ids tensor
    rand = tf.random.uniform(inputs.input_ids.shape)

    # create mask array
    mask_arr = tf.logical_or(inputs.input_ids == 101, inputs.input_ids == 102)
    mask_arr = tf.logical_or(mask_arr, inputs.input_ids == 0)

    tmp_mask_arr = (rand < 0)
    for ids in dict:
        tmp_mask_arr = tf.logical_or(tmp_mask_arr, (inputs.input_ids == ids))
    mask_arr = tf.logical_and(mask_arr, tmp_mask_arr)

    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            tf.where(tf.not_equal(mask_arr[i], tf.zeros_like(mask_arr[i]))).numpy().tolist()
        )

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

from transformers import BitsAndBytesConfig
double_quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)
model = TFDistilBertForMaskedLM.from_pretrained(model_selected, cache_dir=os.path.join(current_folder, "model"), quantization_config=double_quant_config)
model.resize_token_embeddings(len(tokenizer))


def decode_csv_line(line):
    # all_data is a list of scalar tensors
    line_text = tf.io.decode_csv(records=line, record_defaults=[[0]], field_delim=",")
    inputs = tokenizer(line_text, return_tensors='pt', max_length=32, truncation=True, padding='max_length')
    inputs['labels'] = inputs.input_ids.detach().clone()
    # mask tokens
    get_attention(inputs, dict)
    return inputs

def get_epochs():
    # Provide the ability to decode a CSV
    dataset = tf.data.TextLineDataset([os.path.join(current_folder, '../clean.csv')])
    dataset = dataset.map(decode_csv_line)
    dataset = dataset.shuffle(buffer_size=10000 * 4)
    dataset = dataset.repeat(1).batch(4)
    return dataset

get_epochs()

train_dataset = tf.data.Dataset.from_tensors(inputs)
train_dataset = train_dataset.shuffle(buffer_size=10000 * 4)
train_dataset = train_dataset.repeat(1).batch(4)


class MeditationsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)
#
dataset = MeditationsDataset(inputs)


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.metrics.SparseCategoricalAccuracy(),
)

model.fit(get_epochs)

input = ["Machine learning is an revolutionary technique"]
input = tokenizer(input, return_tensors='pt', max_length=32, truncation=True, padding='max_length')
predict = model(input.input_ids, attention_mask=input["attention_mask"])

prob = torch.softmax(predict.logits, dim=-1)

def print_pro(ind):
    print("index is {}".format(ind))
    print("machine prob is {}".format(prob[0, ind, tokenizer.vocab["machine"]]))
    print("learning prob is {}".format(prob[0, ind, tokenizer.vocab["learning"]]))
    print("is prob is {}".format(prob[0, ind, tokenizer.vocab["is"]]))
    print("an prob is {}".format(prob[0, ind, tokenizer.vocab["an"]]))
    print("revolutionary prob is {}".format(prob[0, ind, tokenizer.vocab["revolutionary"]]))
    print("technique prob is {}".format(prob[0, ind, tokenizer.vocab["technique"]]))

    print("Maximum index is {}".format(torch.argmax(prob, dim=-1)[0, ind]))

# save model
model.save_pretrained(os.path.join(current_folder, "../model"))


print_pro(0)
print("***************")
print_pro(1)
print("***************")
print_pro(2)
print("***************")
print_pro(3)
print("***************")
print_pro(4)