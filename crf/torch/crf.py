from datasets import load_dataset
from transformers import BertTokenizerFast
from models import BertCRF

train_dataset, test_dataset = load_dataset('conll2003', split=['train', 'test'])
print(train_dataset, test_dataset)

model = BertCRF.from_pretrained('bert-base-cased', num_labels=9)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')

