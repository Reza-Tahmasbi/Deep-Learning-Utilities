import torch
from transformers import BertTokenizer, BertModel

#load-pretrained-model-tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text = """Hold fast to dreams, for if dreams die, life is a broken-winged bird,→ that cannot fly."""

# tokenize the sentence with BERT tokenizer
tokenized_sentence = tokenizer.tokenize(text)

# print out the tokens
print(tokenized_sentence)

# convert token to vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)

print("Indexed Items: ", indexed_tokens)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])

# define BERT model
model = BertModel.from_pretrained("bert-base-uncased")

# set the model to evaluation mode to deactivate the dropout moduels
model.eval()

# if gpu available
try:
    tokens_tensor = tokens_tensor.to("cuda")
    model.to("cuda")
except:
    print("GPU is not available")
    
with torch.no_grad():
    outputs =  model(tokens_tensor)
    # Transformers models always output tuples.
    # See the models docstrings for the detail of all the outputs
    # In our case, the first element is the hidden state of the last layer of the Bert model
    encoded_layers = outputs[0]
    # We have encoded our input sequence in a FloatTensor of shape (batch size, sequence length, model hidden dimension)

print(encoded_layers)