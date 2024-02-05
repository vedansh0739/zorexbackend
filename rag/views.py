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
from django.views.decorators.http import require_http_methods
import shutil

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







    folder_path = os.path.join(settings.MEDIA_ROOT, session_id)
    
    # Check if the folder exists
    if not os.path.isdir(folder_path):
        pass   
    try:
        # Remove all contents of the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception:
                pass
    except Exception:
        pass



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
    
    try:
        collection=client.get_collection(name=session_id)
        client.delete_collection(name=session_id)
    except Exception:
        pass

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
   





    folder_path = os.path.join(settings.MEDIA_ROOT, session_id)

    # Check if the folder exists
    if not os.path.isdir(folder_path):
        pass
    try:
        # Remove all contents of the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception:
                pass

    except Exception:
        pass







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
            # if(len(extracted_text)/4>100000):
            #     print(f"{file} will get ignored as it exceeds the 100000 token limit")
            #     continue
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
   
    try:
        collection=client.get_collection(name=session_id)
        client.delete_collection(name=session_id)
    except Exception:
        pass

    collection = client.create_collection(session_id)



    
    collection.add(
        documents=results, 
        ids=pdfNames,
    )
    
    globalCollections[session_id]=collection
    
    
    
    
    
    
    delete_files_after_delay(session_id, 900, user_dir)
    return JsonResponse({'message': 'String received successfully!', 'results': results, 'filenames':pdfNames})





from langchain.prompts import PromptTemplate

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
        


        
        
        
        
        conversionllm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)

        template = """Convert the following user queries into declarative sentences to enhance the effectiveness of a similarity search over a document folder. The conversion should reformulate the question into a statement that encapsulates the essence of the query, aiding in pinpointing the relevant documents. Below are a few examples to demonstrate the desired transformation:

User Query: Find a case law where fashion photographers were accused of copyright infringement.
Conversion: Fashion photographers were accused of copyright infringement.

User Query: Find the PDF where the defense to copyright infringement was freedom of expression.
Conversion: The defense to copyright infringement was freedom of expression.

User Query: Find proceeding where the defendants alleged entrapment against the plaintiffs.
Conversion: The defendants alleged entrapment against the plaintiffs.

User Query: In which judgements of copyright infringement ignorance was a defense that was used by the defendant?
Conversion: Copyright infringement where ignorance was a defense that was used by the defendant.

Now, convert the following user query:

User Query: {query}
Conversion: """
        conversionPrompt = PromptTemplate.from_template(template)
        pr=conversionPrompt.format(query=input_string)
        finalquery=conversionllm.predict(pr)
        print("________finalquery_________")
        print(finalquery)
        print("________finalquery_________")
        
        
        
        
        
        
        
        filenames=globalFileNames[session_id]
        results=globalResults[session_id]
        
        result_dict = dict(zip(filenames, results))
        
        collection=globalCollections[session_id]
        
        
        chromaanswer=collection.query(query_texts=[finalquery],
        n_results=3 ,
        )
        print('^^^^^')
        print(chromaanswer['ids'][0])
        answerfilenames=chromaanswer['ids'][0]
        print(chromaanswer['distances'][0])
        answerscores=chromaanswer['distances'][0]
        print('^^^^^')
        filescoresdict=dict(zip(answerfilenames,answerscores))
        
        
        
        combined_dict={
            filename: {
                'result': result_dict[filename],
                'score': filescoresdict[filename]
            } for filename in result_dict
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




