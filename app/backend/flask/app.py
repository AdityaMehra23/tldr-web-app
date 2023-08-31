from flask import Flask, request
from transformers import pipeline
from flask_cors import CORS, cross_origin

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


@app.route('/summarize', methods=['POST'])
@cross_origin()
def summarize():
    text1 = request.form.get('test_string')

    summarizer = pipeline("summarization", model="./model/")
    summary= summarizer(text1)
    summarised = summary[0]["summary_text"]

    # print('\n' + summarised)
    return summarised




'''from transformers import pipeline


text1 = "Write text to summarize"

summarizer = pipeline("summarization", model="./model/patent_v2_1000/")
summary= summarizer(text1)

print(summary[0]["summary_text"])'''