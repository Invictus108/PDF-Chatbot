from pdf_parse import parse_pdf
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone
import requests
import aiohttp
import asyncio
import uuid

def insert_pdf(pdf_path, openai_api_key, pinecone_api_key):
    text, images = parse_pdf(pdf_path)

    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ['PINECONE_API_KEY'] = pinecone_api_key

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
                    "text": f"Use this context: {text if text else 'No context in this query'}, and these tables {'\\n\\n'.join([table.to_string(index=False) for table in tables]) if tables else 'No tables in this query'} alongside the images to answer this question {question}"
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

    # get original summary from chatGPT. Label Images and Tables
    query = "Summerize the PDF"
    headers, payload = make_json(text, images, None, query, os.getenv('OPENAI_API_KEY'))
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    summary = response.json()['choices'][0]['message']['content']

    # chunk text data for embeddings
    def chunk_data(text, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        return chunks

    split_docs = chunk_data(text)


    # async get image summaries
    async def get_image_summaries(images):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for image in images:
                query = "Provide a short summary of the image and 5 specific questions it can be used to answer. Seperate each individual entry with four dashes (----)."
                headers, payload = make_json(None, [image], None, query, os.getenv('OPENAI_API_KEY'), max_tokens=2000)
                tasks.append(fetch_summary(session, headers, payload))
            
            responses = await asyncio.gather(*tasks)
            
            for response_str in responses:
                response_arr = response_str.split('----')
                if response_arr[-1] == '':
                    response_arr = response_arr[:-1]
                image_summaries.append(response_arr)

    async def fetch_summary(session, headers, payload):
        async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as response:
            response.raise_for_status()
            response_data = await response.json()
            return response_data['choices'][0]['message']['content']

    image_summaries = []
    asyncio.run(get_image_summaries(images))

    assert len(image_summaries) == len(images)

    # get embeddings
    def get_embeddings(chunks):
        embeddings = []
        for chunk in chunks:
            embedding = embedding_model.embed_query(chunk)
            embeddings.append(embedding)
        return embeddings

    # get embeddings 
    text_embeddings = get_embeddings(split_docs)
    image_embeddings = []
    for part in image_summaries:
        image_embeddings.append(get_embeddings(part))


    # insert data into pinecone 
    # initialize
    index_name = "pdf-chatbot"
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)

    # insert vectors async
    async def async_upsert(index, vectors):
        index.upsert(vectors)

    # funtion for inserting vectors
    async def insert_vectors(index, embeddings, metadata):
        tasks = []
        assert len(embeddings) == len(metadata)
        for i in range(len(metadata)):
            if not isinstance(embeddings[i][0], list): 
                vectors=[
                        {
                            "id" : str(uuid.uuid1()),
                            "values":embeddings[i], 
                            "metadata":metadata[i]
                        }
                    
                    ]
                tasks.append(async_upsert(index, vectors))
            else:
                for j in range(len(embeddings[i])):
                    vectors=[
                        {
                            "id" : str(uuid.uuid1()),
                            "values":embeddings[i][j], 
                            "metadata":metadata[i]
                        }
                    
                    ]
                    tasks.append(async_upsert(index, vectors))
        await asyncio.gather(*tasks)

    async def insert_vectors_main(index, embeddings, metadata):
        await insert_vectors(index, embeddings, metadata)      


    # insert text vectors
    text_metadata = [{"source": pdf_path, "type": "text", "content": i} for i in split_docs]
    asyncio.run(insert_vectors_main(index, text_embeddings, text_metadata))

    # insert image vectors
    images_metadata = [{"source": pdf_path, "type": "image", "content": i} for i in images]
    asyncio.run(insert_vectors_main(index, image_embeddings, images_metadata))

    return summary



