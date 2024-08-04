from insert import insert_pdf
from query import query_AI
from flask import Flask, flash, redirect, render_template, request, session
import os
from dotenv import load_dotenv
import html

load_dotenv()

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/", methods=["GET", "POST"])
def main():
    if request.method == "GET":
        return render_template("index.html", data="")
    else:
        question = request.form.get("question")
        convo = request.form.get("convo")
        context = request.form.get("context")

        convo, context = query_AI(context, convo, question, os.getenv('OPENAI_API_KEY'), os.getenv('PINECONE_API_KEY'))

        # Replace newline characters with <br> for HTML rendering
        convo = html.escape(convo).replace('\n', '<br>') #+ "<br>"

        return render_template("index.html", data=convo, convo=convo, context=context)


@app.route("/insert", methods=["GET", "POST"])
def insert():
    if request.method == "GET":
        return render_template("index.html", data="")
    else:
        file = request.files['file']
        
        if file.filename == '':
            return render_template("index.html", message="No selected file")
        
        if file and file.filename.endswith('.pdf'):
            # Save the file to a designated location
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            
            # Call insert_pdf function
            summary = insert_pdf(file_path, os.getenv('OPENAI_API_KEY'), os.getenv('PINECONE_API_KEY'))

            # Replace newline characters with <br> for HTML rendering
            summary = html.escape(summary).replace('\n', '<br>') #+ "<br>"

            return render_template("index.html", data=summary, convo=summary, context=summary)
        else:
            return render_template("index.html", data="Invalid File Type. Make sure you upload a .pdf file")
        

    

