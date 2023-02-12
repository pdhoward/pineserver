import datetime
import os
import json
import flask
import openai

from dotenv import load_dotenv



load_dotenv()

WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
OPEN_API_KEY = os.environ.get("OPENAI_API_KEY")

#######################################
# get API key from top-right dropdown on OpenAI website
openai.api_key = OPEN_API_KEY

app = flask.Flask(__name__)
port = int(os.getenv("PORT", 9099))

query = "who was the 12th person on the moon and when did they land?"

# function to handle openai prompt
def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()



@app.route('/message', methods=['POST'])
def message():
    query = "What is an agent"    
    #docs = vectorstore.similarity_search(query)   
    #response = {'message': docs[0].page_content}
    #return flask.jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
