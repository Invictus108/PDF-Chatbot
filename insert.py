import sys
from pdf_parse import parse_pdf, image_to_base64
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from langchain_community.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone
import aiohttp
import asyncio
import uuid
import instructor
from pydantic import BaseModel
import json
from openai import OpenAI

def insert_pdf(pdf_path, openai_api_key, pinecone_api_key):
    # parse pdf
    text, images = parse_pdf(pdf_path, pages=True)

    # get keys
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ['PINECONE_API_KEY'] = pinecone_api_key

   
    # Initialize the instructor client
    client = instructor.from_openai(OpenAI())

    # define embeddings
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # define make json funtion for formatting data for instructor client
    def make_json(text, images, question):
        messages = [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"Use this context: {text if text else "No context in this query"}, alongside the images to answer this question {question}"
                },
            
            ]
            }
        ]
    
        for image in images:
            messages[0]['content'].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_to_base64(image)}"
                    },
                    
            })

        return messages


    # template for summary
    class Summary(BaseModel):
        summary: str

    # get original summary from chatGPT. Label Images and Tables
    query = "Summerize the PDF"
    message = make_json(text, images, query)
    response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=Summary,
                messages=message
            )
    summary = response.summary + "\n"

    # chunk text data for embeddings
    def chunk_data(text, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_text(text)
        return chunks

    split_docs = chunk_data(text)

    # template for image summaries
    class Image_Response(BaseModel):
        summary: str
        question_1: str
        question_2: str
        question_3: str
        question_4: str

    # async get image summaries
    async def get_image_summaries(images):
        tasks = []
        for image in images:
            query = "Provide a short summary of the image and 4 specific questions about data in the presentation that would be asked by a business analyst. Seperate each individual entry with four dashes (----). Ignore text and only focus on images and tables."
            message = make_json(None, [image], query)
            tasks.append(fetch_summary(message))
        
        return await asyncio.gather(*tasks)
            

    async def fetch_summary(messages):
        # Using the instructor client to extract structured data
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-mini",
            response_model=Image_Response,
            messages=messages
        )
        
        return [response.summary, response.question_1, response.question_2, response.question_3, response.question_4]
        

    image_summaries = asyncio.run(get_image_summaries(images))

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
    images_metadata = [{"source": pdf_path + ":" + str(i), "type": "image", "content": i} for i in images]
    asyncio.run(insert_vectors_main(index, image_embeddings, images_metadata))

    return summary


