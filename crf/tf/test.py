from tmp.research.crf.tf.model_tf_1 import TFDistillBertCRF
from transformers.models.distilbert import DistilBertConfig
from transformers.models.distilbert import DistilBertTokenizer

config = DistilBertConfig()
config.n_layers = 1

config.num_labels = 787
config.resize_input_output = 31152
model = TFDistillBertCRF.from_pretrained("distilbert-base-uncased", config=config)
model.resize_token_embeddings(31152)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
input = tokenizer('total sales', return_tensors='tf', max_length=32, truncation=True, padding='max_length')

model(input)
a = 1