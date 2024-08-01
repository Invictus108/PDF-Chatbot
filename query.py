import os
import requests
import os
from langchain_community.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone
import requests

def query_AI(context, convo, question, openai_api_key,  pinecone_api_key):
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ['PINECONE_API_KEY'] = pinecone_api_key

    # initalize pinecone
    index_name = "pdf-chatbot"
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    # define embeddings
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    def make_json(text, images, tables,  question, api_key, max_tokens = 1000):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
            }  
        
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": f"Use this context: {text if text else "No context in this query"}, and these tables {"\n\n".join([table.to_string(index=False) for table in tables]) if tables else "No tables in this query"} alongside the images to answer this question {question}"
                    },
                
                ]
                }
            ],
            "max_tokens": max_tokens
            }
        
        for image in images:
            payload["messages"][0]['content'].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image}"
                    },
                    
            })

        return headers, payload

    if context is None:
        context = '''
                You will be engaging in a conversation as a professional analyst, using data to answer questions.
                You will be provided text, images and tables for context as the conversation contiues in order to best respond.
                Be as kind and helpful as possible, and make sure to provide sources.\n
        '''

    context += question + "\n"
    convo += "User: " + question + "\n"

    # fetch similar docs using query as search vector
    top_k=5
    docs = []
    docs_content = []
    while len(docs) < top_k:
        similar_doc = index.query(vector=embedding_model.embed_query(question), filter={'contents' : {'$nin' : docs_content}}, top_k=1, include_metadata=True)
        docs.append(similar_doc['matches'][0])
        docs_content.append(similar_doc['matches'][0]['metadata']['content'])

    # add to context string
    context += f"Context for Question {question}\n"
    tables = []
    images = []
    for doc in docs:
        if doc['metadata']['type'] == 'text':
            context += f"From source {doc['metadata']['source']}: {doc['metadata']['content']}\n"
        
        context += "Sources for the tables (in order): "
        if doc['metadata']['type'] == 'table':
            tables.append(doc['metadata']['content'])
            context += doc['metadata']['source'] + ", "
        context += "\n Sources for images (in order): "
        if doc['metadata']['type'] == 'image':
            images.append(doc['metadata']['content'])
            context += doc['metadata']['source'] + ", "

    # make sure it only answers the most recent question
    len_extra = len(f"\n Answer Questions: {question}")
    context += f"\n Answer Questions: {question}"
    
    # get header and payload
    header, payload = make_json(context, images, tables, question, os.getenv('OPENAI_API_KEY'))
    
    # get answer
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=header, json=payload)
    answer = response.json()['choices'][0]['message']['content']

    # add answer to convo string
    convo += "Agent: " + answer + "\n"

    # remove extra
    context = context[:-len_extra]
    context += f"\nYour answer to question {question}: {answer}\n"
    
    return convo, context

        
