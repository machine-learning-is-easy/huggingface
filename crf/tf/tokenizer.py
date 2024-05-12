
from transformers import BertTokenizerFast, DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

encode = tokenizer("Alibaba is not a retail company", max_length=32, truncation=True, padding="max_length").data
input_ids = encode["input_ids"]
mask_ids = encode["attention_mask"]