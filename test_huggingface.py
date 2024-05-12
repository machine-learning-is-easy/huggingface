from transformers import create_optimizer
from transformers import TFAutoModelForSequenceClassification, TFDistilBertForMaskedLM, TFBertForMaskedLM
from transformers import create_optimizer
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from datasets import load_dataset

imdb = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

tf_train_set = tokenized_imdb["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
)

tf_validation_set = tokenized_imdb["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
)
batch_size = 16
num_epochs = 5
batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

model = TFDistilBertForMaskedLM.from_pretrained("distilbert-base-uncased", num_labels=2)
model.summary()
model = TFBertForMaskedLM.from_pretrained("bert-base-uncased", num_labels=2)
model.summary()


import tensorflow as tf

batch_size = 16
num_epochs = 5
batches_per_epoch = len(tokenized_imdb["train"]) // batch_size

model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=3)