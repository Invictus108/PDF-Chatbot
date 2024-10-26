# PDF Chatbot: Jaden Cohen MIT Maker Portfolio Project 2

This project is a Python Flask web application that leverages the power of Pinecone for vector storage and ChatGPT embeddings to create a highly interactive custom agent. The primary goal of the agent is to ingest and understand PDF documents, allowing users to ask questions about the content and receive accurate, context-aware responses.

## Key Features

- **Flask Backend**: The application is built on Flask, a lightweight web framework, providing an easy-to-use and scalable backend for handling user requests and serving responses.
  
- **Pinecone for Vector Storage**: Pinecone is used for efficiently storing and retrieving document embeddings. It allows the system to search through ingested PDF data and find the most relevant information based on user queries.

- **ChatGPT Embeddings**: The project uses OpenAI's ChatGPT to generate embeddings from the text extracted from PDFs. These embeddings are then stored in Pinecone, enabling semantic search for precise answers to user questions.

- **PDF Ingestion**: The application can process and ingest PDF documents, extracting the text and converting it into a format that can be embedded and stored. This ensures the chatbot can analyze and query the document's contents.

- **Custom Agent**: The custom-built chatbot is designed to be capable of understanding and answering questions about the ingested PDFs, making it useful for a wide range of applications, including document summarization, knowledge extraction, and FAQ generation.

## How it Works

1. **Upload PDFs**: Users upload one or more PDF documents to the system. The application extracts the text from the PDFs and processes them into a machine-readable format.

2. **Embedding Generation**: The extracted text is sent to ChatGPT to generate embeddings, which capture the semantic meaning of the content.

3. **Vector Storage with Pinecone**: The embeddings are then stored in Pinecone, a fast and scalable vector database, for efficient searching and retrieval.

4. **Querying the Bot**: Users can ask questions related to the content of the PDF. The system compares the user's question against the stored embeddings to find the most relevant sections and generates a context-aware response.

5. **Real-time Interaction**: The custom agent provides responses in real time, allowing users to interact with the PDF content as if they were conversing with an expert on the document.

## Use Cases

- **Academic Research**: Students and researchers can upload research papers and ask questions to better understand the content.
  
- **Business Documents**: Quickly analyze long reports, contracts, or other documents by querying specific sections or details.

- **Customer Support**: Build an FAQ bot by feeding customer support documents and allowing the agent to answer user inquiries.

## Technologies Used

- **Flask**: For building the backend web application.
- **Pinecone**: For vector storage and retrieval of document embeddings.
- **ChatGPT API**: For generating text embeddings and enabling natural language understanding.

## Installation and Setup


   ```bash
   git clone https://github.com/username/pdf-chatbot.git
   cd pdf-chatbot
   pip install -r requirements.txt
   flask run
   
