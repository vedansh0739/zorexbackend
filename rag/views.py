from django.http import JsonResponse,HttpResponse
from django.views.decorators.csrf import csrf_protect
from django.views.decorators.http import require_POST
import os
import shutil
import ssl
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.text_splitter import CharacterTextSplitter
import os
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
from django.conf import settings
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import threading
import boto3
import requests
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_openai import OpenAIEmbeddings
from botocore.exceptions import NoCredentialsError, ClientError
from unstructured_client import UnstructuredClient
import PyPDF2
globalResults={}
globalFileNames={}
def hello(request):
    return HttpResponse("Hello, world. You're at the rag index.")

timers = {}  
from unstructured.partition.pdf import partition_pdf
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document

from langchain.document_loaders import UnstructuredPDFLoader
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def pdf_to_text(pdf_file_path, dpi=200, pages=None):
    images = convert_from_path(pdf_file_path, dpi=dpi)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text


def delete_files_after_delay(session_id, delay, user_dir):
    def delayed_deletion():
        if os.path.exists(user_dir):
            shutil.rmtree(user_dir)
        del timers[session_id]  # Remove the timer from the dictionary

    if session_id in timers:
        timers[session_id].cancel()  # Cancel existing timer if any

    timer = threading.Timer(delay, delayed_deletion)
    timer.start()
    timers[session_id] = timer  # Store the new timer

    
@require_POST
@csrf_exempt
def upload_files(request):

    
    if not request.session.exists(request.session.session_key):
        request.session.create()
    request.session['init'] = True
    request.session.save()
    if request.method == 'POST':
        session_id = request.session.session_key
    print("|||||")
    print(session_id)
    print("|||||")
    user_dir = os.path.join(settings.MEDIA_ROOT, session_id)
    os.makedirs(user_dir, exist_ok=True)

    elements=[]
    all_text={}
    file_paths=[]

    for f in request.FILES.getlist('files'):
        if f.name.lower().endswith('.pdf'):  
            file_path = os.path.join(user_dir, f.name)
            with open(file_path, 'wb+') as destination:
                for chunk in f.chunks():
                    destination.write(chunk)
                    
    dir_path = user_dir
    books=[]
    filenames=[]
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            extracted_text = pdf_to_text(file_path)
            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4*15000,
            chunk_overlap=4*100,
            length_function=len,
            )
            chunks = text_splitter.create_documents([extracted_text])
            filenames.append(file)
            books.append(chunks)







    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final summary\n"
        "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary"
        "If the context isn't useful, return the original summary."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )
    results=[]
    for book in books:
        result=chain({"input_documents": book}, return_only_outputs=True)
        results.append(result['output_text'])

    globalResults[session_id]=results
    globalFileNames[session_id]=filenames

        
        


    
    delete_files_after_delay(session_id, 900, user_dir)
    
    
    return JsonResponse({'message': 'String received successfully!', 'results': results, 'filenames':filenames})




    
    





@csrf_exempt
def query(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        input_string = data.get('data')
        
        request.session['init'] = True
        request.session.save()
        if request.method == 'POST':
            session_id = request.session.session_key
        print("|||||")
        print(session_id)
        print("|||||")
        

        print(input_string)
        query=input_string
        
        
        
        results=globalResults[session_id]
        filenames=globalFileNames[session_id]
        vectorizer = TfidfVectorizer()
        document_vectors = vectorizer.fit_transform(results)
        query_vector = vectorizer.transform([query])
        similarity_scores = cosine_similarity(query_vector, document_vectors)
        top_k_indices = similarity_scores.argsort()[0][-1:]
        top_k_documents = [results[i] for i in top_k_indices]
        answer=top_k_indices[0]
        print(filenames[answer])
        print(results[answer])

        
        
        return JsonResponse({'result': results[answer], 'filename':filenames[answer]})

    return JsonResponse({'error': 'Invalid request'}, status=400)





@require_POST
@csrf_exempt
def reset_timer(request):
    if request.method == 'POST':
        session_id = request.session.session_key
        if session_id in timers:
            user_dir = os.path.join(settings.MEDIA_ROOT, session_id)
            delete_files_after_delay(session_id, 900, user_dir)  # Reset to 900 seconds
            return JsonResponse({'message': 'Timer reset successfully'}, status=200)
        else:
            return JsonResponse({'message': 'No active timer for this session'}, status=404)

    return JsonResponse({'message': 'Invalid request'}, status=400)




