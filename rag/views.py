from django.http import JsonResponse,HttpResponse
from django.views.decorators.http import require_POST
import shutil
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
import os
import fitz
import langchain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
import logging
from django.conf import settings
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
import threading
import pytesseract
from pdf2image import convert_from_path
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

timers = {}  
globalResults={}
globalFileNames={}
globalCollections={}
import chromadb

client = chromadb.Client()
def hello(request):
    return HttpResponse("Hello, world. You're at the rag index and the backend probably works as you can see this. Congrats ig.")

def get_text_from_pdf(file_path):
    # Open the provided PDF file
    with fitz.open(file_path) as doc:
        text = ""
        # Iterate over each page in the PDF
        for page_num in range(len(doc)):
            # Get a page
            page = doc.load_page(page_num)
            # Extract text from the page
            text += page.get_text()
        return text
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
def upload_files_ocr(request):
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

    for f in request.FILES.getlist('files'):
        if f.name.lower().endswith('.pdf'):  
            file_path = os.path.join(user_dir, f.name)
            with open(file_path, 'wb+') as destination:
                for chunk in f.chunks():
                    destination.write(chunk)
                    
    dir_path = user_dir
    pdfs=[]
    pdfNames=[]
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
            print(f"++{file}++{len(chunks)} are the number of chunks ++ {len(extracted_text)/4} is the number of tokens")
            if(len(extracted_text)/4>100000):
                print(f"{file} will get ignored as it exceeds the 100000 token limit")
                continue
            pdfNames.append(file)
            pdfs.append(chunks)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    prompt_template = """Please provide a comprehensive summary of the text, ensuring to identify and specify the roles of the involved parties within the summary. If the text mentions the plaintiff or the defendant, clearly label them as such in your summary. Additionally, if the prevailing party is identified, please denote whether they were the plaintiff or the defendant. The summary should encapsulate all key points and details from the provided text:
    {text}
    COMPREHENSIVE SUMMARY:"""
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
    
    
    for pdf in pdfs:
        result=chain({"input_documents": pdf}, return_only_outputs=True)
        results.append(result['output_text'])
    
    

    
    
    globalResults[session_id]=results
    globalFileNames[session_id]=pdfNames
    
    
    
    
    collection = client.create_collection(session_id)
    collection.add(
        documents=results, 
        ids=pdfNames,
    )
    
    globalCollections[session_id]=collection
    
    
    
    
    
    
    delete_files_after_delay(session_id, 9000, user_dir)
    return JsonResponse({'message': 'String received successfully!', 'results': results, 'filenames':pdfNames})






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

    for f in request.FILES.getlist('files'):
        if f.name.lower().endswith('.pdf'):  
            file_path = os.path.join(user_dir, f.name)
            with open(file_path, 'wb+') as destination:
                for chunk in f.chunks():
                    destination.write(chunk)
                    
    dir_path = user_dir
    pdfs=[]
    pdfNames=[]
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            file_path = os.path.join(root, file)
            extracted_text = get_text_from_pdf(file_path)
            
            text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4*15000,
            chunk_overlap=4*100,
            length_function=len,
            )
            
            chunks = text_splitter.create_documents([extracted_text])
            print(f"++{file}++{len(chunks)} are the number of chunks ++ {len(extracted_text)/4} is the number of tokens")
            if(len(extracted_text)/4>100000):
                print(f"{file} will get ignored as it exceeds the 100000 token limit")
                continue
            pdfNames.append(file)
            pdfs.append(chunks)

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

    prompt_template = """Please provide a comprehensive summary of the text, ensuring to identify and specify the roles of the involved parties within the summary. If the text mentions the plaintiff or the defendant, clearly label them as such in your summary. Additionally, if the prevailing party is identified, please denote whether they were the plaintiff or the defendant. The summary should encapsulate all key points and details from the provided text:
    {text}
    COMPREHENSIVE SUMMARY:"""
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
    for pdf in pdfs:
        result=chain({"input_documents": pdf}, return_only_outputs=True)
        results.append(result['output_text'])

    globalResults[session_id]=results
    globalFileNames[session_id]=pdfNames
    
    
    
    
    collection = client.create_collection(session_id)
    collection.add(
        documents=results, 
        ids=pdfNames,
    )
    
    globalCollections[session_id]=collection
    
    
    
    
    
    
    delete_files_after_delay(session_id, 900, user_dir)
    return JsonResponse({'message': 'String received successfully!', 'results': results, 'filenames':pdfNames})





from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
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
        filenames=globalFileNames[session_id]
        results=globalResults[session_id]
        
        result_dict = dict(zip(filenames, results))
        
        collection=globalCollections[session_id]
        
        
        chromaanswer=collection.query(query_texts=[query],
        n_results=3 ,
        )
        print('^^^^^')
        print(chromaanswer['ids'][0])
        answerfilenames=chromaanswer['ids'][0]
        print(chromaanswer['distances'][0])
        answerscores=chromaanswer['distances'][0]
        print('^^^^^')
        filescoresdict=dict(zip(answerfilenames,answerscores))
        
        
        
        combined_dict={filename: {
        'result': result_dict[filename],
        'score': filescoresdict[filename]
    }for filename in result_dict
}
        score_result_pairs = [(info['score'], info['result']) for info in combined_dict.values()]
        print(f"**{result_dict}**{filescoresdict}**")
        score_result_pairs.sort(key=lambda x: x[0])
        answerresults= [result for score, result in score_result_pairs]
        return JsonResponse({'summaries': answerresults, 'filenames':answerfilenames, 'scores':answerscores})

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




