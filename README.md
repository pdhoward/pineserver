## PINECONE VECTOR INDEX SERVER

### prepping the data

The mongodb with Home Depot product is densely packed with a rich set of information description, including abstract product names, extended descriptions, technical specifications and pricing. The first step in creating a vector db for the products is to create a synopsis of the product data, to improve semantic search.

The script at machinechain/seeders/vectorprep updates each mongo document with text which contains 7 essential facts about the product. The facts were produced by Turbo-GPT - essentially AI generating the bext vector for AI to search.

### set up server

A pinecone vector db server is created through the PineCone console

Follow the installation instructions and capture the api key and environment variables

Use the ```ingest.py``` script to migrate mongo data from Atlas to PineCone, creating the vector indexes as part of the process.

Pay attention to declared data types on each document, ensuring that all properties are uniformly cleansed throughout the collection

Use the server ```app.py``` script to handle http calls


### set up client



### RESEARCH

https://docs.pinecone.io/docs/gen-qa-openai



