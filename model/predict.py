from transformers import pipeline
import os

#Text to summarize
text = "This is a just a dummy text that i m using to check if the inference of the model is going to work or not in the Sonic Cluster that are provided by the ucd"


#Loading the saved model
summarizer = pipeline("summarization", model="./model/")
summary= summarizer(text)

with open("summary.txt","w") as f:
    f.write(text)
    f.close()

