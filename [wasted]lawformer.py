from transformers import AutoModel, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("thunlp/Lawformer")
model = AutoModel.from_pretrained("thunlp/Lawformer")
inputs = tokenizer("任某提起诉讼，请求判令解除婚姻关系并对夫妻共同财产进行分割。", return_tensors="pt")
outputs = model(**inputs)
print(tokenizer.decode(inputs["input_ids"][0]))