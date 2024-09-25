import openai  # Import OpenAI for DALL·E
import streamlit as st
from PyPDF2 import PdfReader
import fitz  # PyMuPDF for image extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI and Google API keys
openai.api_key = os.getenv("OPENAI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Predefined abbreviation dictionary
abbreviation_dict = {
    "AI": "Artificial Intelligence",
    "NLP": "Natural Language Processing",
    "ML": "Machine Learning",
    "DL": "Deep Learning",
    # Add more abbreviations as needed
}

def resolve_abbreviation(question):
    """
    Resolves abbreviations in the user question using a predefined dictionary.
    """
    for abbr, full_form in abbreviation_dict.items():
        question = question.replace(abbr, full_form)
    return question

def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Splits a large text into smaller chunks for efficient processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Creates a vector store from a list of text chunks.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversional_chain():
    """
    Creates a conversational chain for question answering.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not available in the context, just say, "answer is not available in the context."
    
    Context:
    {context}?

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to extract images from PDF
def extract_images_from_pdf(pdf_docs):
    """
    Extracts images from the uploaded PDF files using PyMuPDF.

    Args:
        pdf_docs: List of uploaded PDF documents.

    Returns:
        A list of extracted images (in-memory objects).
    """
    images = []
    for pdf in pdf_docs:
        pdf_doc = fitz.open(stream=pdf.read(), filetype="pdf")  # Open PDF from stream
        for page_num in range(len(pdf_doc)):
            page = pdf_doc.load_page(page_num)
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_doc.extract_image(xref)
                image_bytes = base_image["image"]  # Get image bytes
                images.append(image_bytes)
    return images

def user_input(user_question, processed_pdf_text):
    """
    Processes user input and generates a response using the conversational chain.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversional_chain()
    context = f"{processed_pdf_text}\n\nQuestion: {user_question}"
    response = chain({"input_documents": docs, "question": user_question, "context": context}, return_only_outputs=True)
    answer_text = response["output_text"]

    st.write("Reply: ", answer_text)

    # Extract images from the uploaded PDFs
    images = extract_images_from_pdf(st.session_state["pdf_docs"])
    
    if images:
        st.subheader("Extracted Images from PDF:")
        for img_bytes in images:
            st.image(img_bytes, caption="Extracted from PDF", use_column_width=True)
    else:
        st.warning("No images were found in the uploaded PDF files.")

def main():
    """
    Main function for the Streamlit app.
    """
    st.set_page_config("Chat With Multiple PDF")

    # Add CSS styling for the background image
    st.markdown("""
        <style>
        body {
            background-image: url('your-background-image-url');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
            margin: 0;
        }
        </style>
        """, unsafe_allow_html=True)
    st.header("PDF Whisperer 🙋‍♂️")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        # Resolve abbreviations in the user's question
        user_question_resolved = resolve_abbreviation(user_question)

        if st.session_state.get("pdf_docs"):
            processed_pdf_text = get_pdf_text(st.session_state["pdf_docs"])
            user_input(user_question_resolved, processed_pdf_text)
        else:
            st.error("Please upload PDF files first.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload Files & Click Submit to Proceed", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text()
                text_chunks = get_text_chunks(raw_text)
                vector_store = get_vector_store(text_chunks)
                chain = get_conversional_chain()

                st.session_state["pdf_docs"] = pdf_docs
                st.session_state["text_chunks"] = text_chunks
                st.session_state["vector_store"] = vector_store
                st.session_state["chain"] = chain
                st.success("PDFs processed successfully!")

        if st.button("Reset"):
            st.session_state.clear()
            st.experimental_rerun()

        if st.session_state.get("pdf_docs"):
            st.subheader("Uploaded Files:")
            for i, pdf_doc in enumerate(st.session_state["pdf_docs"]):
                st.write(f"{i+1}. {pdf_doc.name}")


if __name__ == "__main__":
    main()
