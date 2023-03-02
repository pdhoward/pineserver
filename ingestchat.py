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
from components.factify import Factifier

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

context = """Your task is to take the content from of a product catalog, and summarize 7 unique pertinent facts about the content from the description and specifications. One of the facts should be price. Do not make up any facts not found in the content. If price is not available include a statement that price is not published."""

product1 = """productcatalog: {
"name": "Feathers In Watercolor Design by Third and Wall Framed Wall Art 11 in. x 14 in.",
"price": 32.11,
"specifications": "[{'Art Movement': 'Abstract Expressionism'}, {'Art Material': 'Wood'}, {'Includes': 'No additional hardware'}, {'Frame Primary Finish': 'Silver'}, {'Theme': 'Abstract'}, {'Artwork name': 'Feathers In Watercolor Design'}, {\"Artist's name\": 'Third and Wall'}, {'Print Type': 'Graphic Print'}, {'Wall Decor Type': 'Wall Art'}, {'Art Medium': 'Graphic Art'}, {'Frame Color/Finish': 'Grey'}, {'Returnable': '90-Day'}, {'Mount Type': 'Wall Mount'}, {'Number of Pieces Included': '1 Piece'}, {'Subject': 'Abstract'}, {'Orientation': 'Portrait'}, {'Color/Finish': 'Multi-Color'}, {'Art Classification': 'Contemporary Looks'}, {'Color Family': 'Multi-Colored'}, {'Hanging method': 'Other'}, {'Features': 'No Additional Features'}, {'Product Weight (lb.)': '2'}, {'Frame Type': 'Framed'}, {'Product Height (in.)': '14'}, {'Product Depth (in.)': '1.5'}, {'Product Width (in.)': '11'}]",
}"""
summary1 = """[
"Art work by the artist Third and Wall called Feathers in Watercolor",
"Art work with a price of 32.11",
"Portrait wall art with dimensions of 11 in X 1.5 in X 14 in. with a product weight of 2 lbs",
"Product is returnable in 90 days",
"Frame primary finaish is silver and grey."
]"""

product2 = """productcatalog: {
"name": "#10-24 x 3/8 in. Black Oxide Coated Steel Set Screws (25-Pack)",
"price": 11.81,
"specifications": "[{'Features': 'No Additional Features'}, {'Fastener Type': 'Set Screws'}, {'Interior/Exterior': 'Interior'}, {'Measurement Standard': 'SAE'}, {'Drive Style': 'Internal Hex'}, {'Returnable': '90-Day'}, {'Fastener/Connector Material': 'Alloy'}, {'Fastener Plating': 'Black Oxide'}, {'Head Style': 'Headless'}, {'Color Family': 'Black'}, {'Included': 'No Additional Items Included'}, {'Finish': 'Black Oxide'}, {'Product Weight (lb.)': '0.005'}, {'Package Quantity': '25'}, {'Size': '#10'}, {'Manufacturer Warranty': 'Goods are warranted against manufacturing defects for 1 year. In no case is Prime-Line responsible for user related damage or damage incurred during installation. Warranty is void if products are subjected to abnormal conditions, misapplication or abuse.'}, {'Screw Length': '3/8 in'}, {'Thread Pitch': '24'}]"
}"""
summary2 = """[
"Set of Screws with a length of 3/8 in",
"The screws are Black Oxide plating, and are intended for interior use",
"The package quantity is 25. The size is #10. The thread pitch is 24",
"The price is 11.81",
"Goods are warranted against manufacturing defects for 1 year. The warranty is void if products are subjected to abnormal conditions, misapplication or abuse."
]"""

product3 = """productcatalog: {
"name": "#10-32 Stainless Steel Fine Cap Nut",
"price": '85.43',
"description":"This acorn cap nut is a hex nut with an enclosed dome and it is partially internally threaded. The dome feature offers protection of exposed screw threads and a decorative appearance as a finishing fastener. The cap nut also protects from injury by covering exposed threads. Can be used in hobby and furniture projects, fencing, lawn or playground equipment or any application where a nut is required and a finished appearance preferred.",
"specifications": "[{'Thread pitch': '32.0'}, {'Fastener Thread Type': 'Fine'}, {'Fastener Type': 'Cap Nut'}, {'Grade': 'A307'}, {'Material': 'Steel'}, {'Measurement Standard': 'SAE'}, {'Returnable': '90-Day'}, {'Fastener Plating': 'Zinc'}, {'ACQ Rated Fastener': 'Yes'}, {'Fastener Callout Size': '#10-32'}, {'Product Weight (lb.)': '0.01 lb'}, {'Package Quantity': '1'}, {'Size': '#10'}, {'Finish Family': 'Metallic'}, {'Manufacturer Warranty': 'None'}, {'Assembled Depth (in.)': '0.52 in'}, {'Outside Width (in.)': '0.375 in'}, {'Assembled Width (in.)': '0.375 in'}, {'Inside Diameter': '0.19 in'}, {'Assembled Height (in.)': '0.433 in'}]"
}"""




####################################################
#############  text turbo chat     #################
####################################################
"""
turboresponse = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": product1},
        {"role": "assistant", "content": summary1},
        {"role": "user", "content": product2},
        {"role": "user", "content": summary2},
        {"role": "user", "content": product3}       
    ]
)
"""

turboresponse = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": product2}              
    ]
)

"""
 {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
"""
print(" --------- testing turbo ---------")
#print(turboresponse['choices'][0]['messages']['content'])
print(turboresponse['choices'][0].message.content)
print(turboresponse['usage']['total_tokens'])



#####################################################

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
def cleanIntegerData(x):
  if isinstance(x, int):
     return x
  return 0
# function to validate that string property is string
def cleanStringData(strng):
  if type(strng) is str:
     return strng
  return "not available"
# function to clean integer data from mongo database
def cleanFloatData(flt):
  if isinstance(flt, float):
     return flt
  return 0.0
batch_size = 100  # number of embeddings we create and insert in a batch

######################################
###        prompt engineering     ####
###  simplify product catalogue   ####
###         for q & a             ####
######################################

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

f = Factifier({'context': 'You are a sales person giving the facts on a product'})

"""

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
          "price": cleanFloatData(docs["price"]),
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
    
    doc = 'here are the facts, x is good. y is better. paris is in france. charlotte is in nc'
    #refactor = [f.factify(x['overview']) for x in meta_batch]
    #refactor = [f.factify({'page_content': doc})]


    
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
"""