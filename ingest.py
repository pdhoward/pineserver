import datetime
import os
import json
import flask
import openai
import pinecone
import tiktoken
from time import sleep
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

OPEN_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENV = os.environ.get("PINECONE_ENV")
MONGO_DB=os.environ.get("DB")

#####################################
# connect to product catalog on mongdb
client = MongoClient(MONGO_DB)
db = client['openai']
collection = db['product']

#######################################
# get API key from top-right dropdown on OpenAI website
openai.api_key = OPEN_API_KEY

embed_model = "text-embedding-ada-002"

"""
res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=embed_model
)

"""

index_name = 'product'

# initialize connection to pinecone (get API key at app.pinecone.io)
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV
)

"""
# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine',
        metadata_config={'indexed': ['channel_id', 'published']}
    )

"""

# connect to index
index = pinecone.Index(index_name)
# view index stats
print(index.describe_index_stats())

# compute number of tokens in a string
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# function to take images from dataset and return an array of strings
def convertArray(str):
  if str is None or str == "":
     return []
  return str.split('~')

# function to clean integer data from mongo database
def cleanIntegerData(int):
  if int is None or int == "":
     return 0
  return int
# function to validate that string property is string
def cleanStringData(strng):
  if type(strng) is str:
     return strng
  return "not available"

batch_size = 100  # number of embeddings we create and insert in a batch

# function to create batches from Mongo cursor
cursor = collection.find({}, batch_size=batch_size)
def yield_rows(cursor, batch_size):
    """
    Generator to yield chunks (batches) from cursor
    :param cursor:
    :param chunk_size:
    :return:
    """
    chunk = []
    for i, row in enumerate(cursor):
        if i % batch_size == 0 and i > 0:
            yield chunk
            del chunk[:]
        chunk.append(row)    
    yield chunk

chunks = yield_rows(cursor, batch_size)

y = 0   # batch counter
######################################################
###### for each batch of documents,          #########
###### cleanse data, create embeddings,      #########
###### update PineCone                       #########
######################################################
for chunk in chunks:
    y += 1
    print("Working on batch ", y)
    meta_batch = chunk    
    #cleanup data    
    meta_batch = [{
          "objectId": str(docs["_id"]),
          "url": docs["url"],
          "crawledAt": docs["crawled_at"],
          "source": docs["source"],
          "name": docs["name"],
          "images": convertArray(docs["images"]),
          "description": docs["description"],
          "brand": docs["brand"],
          "skuId": cleanIntegerData(docs["sku_id"]),
          "price": cleanIntegerData(docs["price"]),
          "inStock": docs["in_stock"],
          "currency": docs["currency"],
          "color": cleanStringData(docs["color"]),
          "breadcrumbs": docs["breadcrumbs"],
          "averageRating": cleanIntegerData(docs["avg_rating"]),
          "totalReviews": cleanIntegerData(docs["total_reviews"]),
          "overview": docs["overview"],
          "specifications": docs["specifications"],
          "productId": docs["id"]
        } for docs in meta_batch]
    # create 
    ids_batch = [x['objectId'] for x in meta_batch]
    texts = ['name: ' + x['name'] + 'brand: ' + x['brand'] + 'overview: ' + x['overview'] + 'specifications: ' + x['specifications'] for x in meta_batch]
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                print("Batch ", y, "failed on embedding")
                pass
    embeds = [record['embedding'] for record in res['data']]
    to_upsert = list(zip(ids_batch, embeds, meta_batch))
    # upsert to Pinecone
    index.upsert(vectors=to_upsert)
    if y == 1:
      print(texts[0])
      print(num_tokens_from_string(texts[0],"cl100k_base" ))
