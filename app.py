from flask import Flask, request, jsonify
import uvicorn
from fastapi import FastAPI, encoders
from starlette import responses
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification




app = FastAPI()

##################Loading the outputs
bert_model = TFBertForSequenceClassification.from_pretrained("outputs/bert_model")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

@app.get('/')
def index():
    return {"Congrats! Server is working"}

# get the json data
@app.get('/get_sentiment/{text_review}')
def get_sentiment(text_review: str):

	text = [text_review]


	tf_batch = bert_tokenizer(text, max_length=128, padding=True, truncation=True, return_tensors='tf')
	tf_outputs = bert_model(tf_batch)
	tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
	labels = ['Negative', 'Positive']
	label = tf.argmax(tf_predictions, axis=1)
	label = label.numpy()
	sent = labels[label[0]]

	#tx = request.get_json(force = True)
	#text = tx['Review']

	#sent = outputs.get_prediction(text)
	json_item = encoders.jsonable_encoder(sent)
	return responses.JSONResponse(content=json_item)



if __name__ == '__main__':
	uvicorn.run(app, host='127.0.0.1', port=8000)




