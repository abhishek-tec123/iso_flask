import os
import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import logging
import warnings

warnings.filterwarnings("ignore")

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_text_from_pdf(pdf_file):
    text = ""
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
    except FileNotFoundError:
        logging.error(f"File '{pdf_file}' not found.")
    except Exception as e:
        logging.error(f"An error occurred while reading the file: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not embeddings:
        logging.error("Failed to generate embeddings.")
        return None
    
    if not text_chunks:
        logging.error("No text chunks provided.")
        return None
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    if not vector_store:
        logging.error("Failed to create vector store.")
        return None
    
    vector_store.save_local("faiss_index")
    return vector_store


def answer():
    prompt_template = """
    You are an AI assistant that provides helpful answers to user queries.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer: 
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, vector_store):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    search = vector_store.similarity_search(user_question)

    ans = answer()

    response = ans(
        {"input_documents": search, "question": user_question}
        , return_only_outputs=True)

    if response:
        answer_text = response['output_text']
        logging.info(f"Question: {user_question}, \n\n  Answer: {answer_text}")
    else:
        answer_text = "Not found in document."
        logging.warning(f"Question: {user_question}, Answer: {answer_text}")

    return clean_text(answer_text)

def clean_text(output_text):
    cleaned_text = output_text.strip()
    cleaned_text = '\n'.join(line for line in cleaned_text.splitlines() if line.strip())
    return cleaned_text

def main():
    logging.info("Loading PDF file...")
    
    file = "docs/ISO+13485-2016.pdf"
    logging.info(f"Loaded file: {file}")
    
    text = get_text_from_pdf(file)
    logging.info("PDF file processed. Extracting text...")
    
    chunks = get_text_chunks(text)
    logging.info("Text extracted. Splitting text into chunks...")
    
    vector_store = get_vector_store(chunks)
    logging.info("Text chunks processed. Vector store created.")

    user_question = input("Ask a Question from the ISO document: ")
    answer_text = user_input(user_question, vector_store)
    
    # print("\n**Answer:**\n")
    # print(answer_text)

if __name__ == "__main__":
    main()
