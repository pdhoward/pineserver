import datetime
import os
import json
import flask
import openai
import pinecone

from dotenv import load_dotenv

load_dotenv()

OPEN_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")

#######################################
# get API key from top-right dropdown on OpenAI website
openai.api_key = OPEN_API_KEY

app = flask.Flask(__name__)
port = int(os.getenv("PORT", 9099))

embed_model = "text-embedding-ada-002"

index_name = 'product'

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

# connect to index
index = pinecone.Index(index_name)
# view index stats
print(index.describe_index_stats())

###############################################
######   openai query before vectors     #####
##############################################

simplequery = "You work in a retail store that sells hammers. Summarize the top 3 hammers that you have and provide a price for each."

complexquery = """
    The partial schema for the product catalog is 

    { name: string,
      brand: string,
      overview: integer,
      Manufacturer Warranty: string }
    
    Given the following context, write snappy advertisement for the lowest priced hammer retrieved
"""
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

result = complete(simplequery)
print(result)
answer_returned = 'The lowest priced hammer available is the Stanley FatMax Xtreme AntiVibe Rip Claw Hammer, which retails for around $10'

###############################################
######   openai query with vectors       #####
##############################################

limit = 3750

def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']
    
    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    #print(res)
    contexts = [
        x['metadata']['overview'] for x in res['matches']        
    ]
    #contexts.append(x['metadata']['name'] for x in res['matches'])
    #contexts.append(x['metadata']['brand'] for x in res['matches'])   
    #contexts.append(x['metadata']['specifications'] for x in res['matches'])

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    #print(contexts[:50])
    return prompt

# using same query prompt - we first retrieve relevant contexts


query_with_contexts = retrieve(complexquery)
print(query_with_contexts)

# then we complete the context-infused query
vector_result = complete(query_with_contexts)
print(vector_result)

vector_answer='Ivy Classic SDS plus tungsten carbide hammer bits are the lowest priced hammer available.'




"""
@app.route('/message', methods=['POST'])
def message():
    query = "What is an agent"    
    #docs = vectorstore.similarity_search(query)   
    #response = {'message': docs[0].page_content}
    #return flask.jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
"""

