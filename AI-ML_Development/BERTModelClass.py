import torch
import torch.nn as nn
from transformers import BertModel

class BERTModel(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', input_dim=300, hidden_dim=768):
        super(BERTModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.classifier = nn.Linear(hidden_dim, vocab_size)  # vocab_size es el tamaño del vocabulario de salida

    def forward(self, x):
        embedded_input = self.embedding(x)
        attention_mask = (x != 0).float()  # Si usas padding, necesitarás una máscara de atención.
        bert_output = self.bert(inputs_embeds=embedded_input, attention_mask=attention_mask)
        logits = self.classifier(bert_output.last_hidden_state)
        return logits

