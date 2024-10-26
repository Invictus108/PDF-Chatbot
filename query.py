import os
from langchain_community.embeddings.openai import OpenAIEmbeddings
from pinecone import Pinecone
import instructor
from pydantic import BaseModel
import json
from openai import OpenAI
from pdf_parse import image_to_base64

def query_AI(context, convo, question, openai_api_key,  pinecone_api_key):
    os.environ['OPENAI_API_KEY'] = openai_api_key
    os.environ['PINECONE_API_KEY'] = pinecone_api_key

    # initalize pinecone
    index_name = "pdf-chatbot"
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    index = pc.Index(index_name)

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
                "text": f"Use this context: {text if text else "No context in this query"}alongside the images to answer this question {question}"
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

    if context is None:
        context = '''
            You will be engaging in a conversation as a professional analyst, using data to answer questions.
            You will be provided text, images and tables for context as the conversation contiues in order to best respond.
            Be as kind and helpful as possible, and make sure to provide sources.\n
    '''
        
    context += question + "\n"
    convo += "User: " + question + "\n"

    # fetch similar docs using query as search vector
    top_k=10
    docs = []
    docs_content = []

    # get similar docs while making sure there are no repeats 
    while len(docs) < top_k:
        similar_doc = index.query(vector=embedding_model.embed_query(question), filter={'contents' : {'$nin' : docs_content}}, top_k=1, include_metadata=True)
        docs.append(similar_doc['matches'][0])
        docs_content.append(similar_doc['matches'][0]['metadata']['content'])

    # process similar docs
    images = []
    for doc in docs:
        if doc['metadata']['type'] == 'text':
            context += f"\nFrom source {doc['metadata']['source']}: {doc['metadata']['content']}\n"
        context += "\n Sources for images (in order): "
        if doc['metadata']['type'] == 'image':
            images.append(doc['metadata']['content'])
            context += doc['metadata']['source'] + ", "

    # tmp main prompt for chatgpt
    len_extra = len(f"\n Answer {question}, and provide sources")
    context += f"\n Answer {question}, and provide sources"
    
    # get message
    message = make_json(context, images, question)
    
    # template for answer
    class Summary(BaseModel):
        summary: str

    # get answer from chatgpt
    response = client.chat.completions.create(
                model="gpt-4o-mini",
                response_model=Summary,
                messages=message
        )
    answer = response.summary

    # add answer to convo string
    convo += "Agent: " + answer + "\n"

    # remove tmp prompt
    context = context[:-len_extra]
    context += f"\nYour answer to q{question}: {answer}\n"

    temp = "User: " + question + "\n" + "Agent: " + answer + "\n"

    #normally Conva
    return temp, context


        

        
