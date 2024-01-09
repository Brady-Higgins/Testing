from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "impira/layoutlm-document-qa"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("question-answering",model=model, tokenizer=tokenizer)

res = classifier("We are happy!")

print(res)