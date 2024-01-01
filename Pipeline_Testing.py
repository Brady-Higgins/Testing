from transformers import pipeline
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

classifier = pipeline("sentiment-analysis",model=model, tokenizer=tokenizer)
res = classifier("We are happy!")

print(res)

tokens = tokenizer.tokenize("We are very happy")
tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

X_train = ["we are happy","I am sad"]
batch = tokenizer(X_train,padding=True,truncation=True,max_length=512,return_tensors="pt")

with torch.no_grad():
    outputs = model(**batch)       #unpacks the dictionary that batch is in
    predictions = F.softmax(outputs.logits, dim=1)
    print(predictions)
    labels = torch.argmax(predictions, dim=1)     #finds max
    print(labels)
    labels = [model.config.id2label[label_id] for label_id in labels.tolist()]
    print(labels)

# save_directory = "saved"
# tokenizer.save_pretrained(save_directory)
# model.save_pretrained(save_directory)
    
model_name = "model_name"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
