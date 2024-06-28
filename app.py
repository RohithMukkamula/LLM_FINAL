import os
import numpy as np
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import streamlit as st
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
import pandas as pd

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Set API keys

os.environ["OPENAI_API_KEY"] = 'open_ai_key'



azure_config = {
    "base_url": "",
    "model_deployment": "",
    "model_name": "",
    "embedding_deployment": "",
    "embedding_name": "",
    "api-key": '',
    "api_version": ""
}

# Set Streamlit page configuration
st.set_page_config(
    page_title="Story Q&A",
    page_icon='tredence-squareLogo-1650044669923.webp',
)

# Read the CSV file into a DataFrame
csv_path = "processed_documents.csv"
df = pd.read_csv(csv_path)

# Title and banner image
st.image("TredenceLogo.png", width=200)
st.title("📚 Story Q&A")

# Recreate Document objects from the DataFrame
documents = []
for index, row in df.iterrows():
    doc = Document(
        page_content=row["content"],
        metadata={"pdf_name": row["pdf_name"]}
    )
    documents.append(doc)

# Predefined file names for selection
file_names = [
    'All',
    'Moby Dick; Or the Whale, by Herman Melville.pdf',
    'Dracula, by Bram Stoker.pdf',
    'Grimms’ Fairy Tales, by Jacob Grimm and Wilhelm Grimm.pdf',
    'War and Peace, by Leo Tolstoy.pdf',
    'Alice’s Adventures in Wonderland, by Lewis Carroll.pdf'
]

# Layout with columns
col1, col2 = st.columns(2)

with col1:
    st.header("Select The Option")
    selected_file = st.selectbox("Select file", file_names)

with col2:
    st.header("Ask Your Question")
    question = st.text_input("Question")

# Filter documents based on the selected file name
if selected_file != "All":
    filtered_documents = [doc for doc in documents if doc.metadata["pdf_name"] == selected_file]
else:
    filtered_documents = documents

submit1 = st.button("Submit")

if submit1:
    with st.spinner('Processing...'):
        def ret(docs):
            return FAISS.from_documents(docs, embeddings).as_retriever(search_kwargs={"k": 10})
        retriever = ret(filtered_documents)

        llm = AzureChatOpenAI(
            temperature=0,
            api_key=azure_config["api-key"],
            openai_api_version=azure_config["api_version"], 
            azure_endpoint=azure_config["base_url"],
            model=azure_config["model_deployment"],
            validate_base_url=False
        )

        PROMPT_TEMPLATE = """
        Go through the context and answer given question strictly based on context. 
        Context: {context}
        Question: {question}
        Answer:
        """

        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=retriever, llm=llm
        )   

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type_kwargs={"prompt": PromptTemplate.from_template(PROMPT_TEMPLATE)}, 
            retriever=retriever_from_llm,
            return_source_documents=True
        )

        result = chain({'query': question})

        st.success('Done!')
        st.subheader("Answer")
        st.write(result['result'])

        # Improved display for source documents
        with st.expander("See the documents"):
            for doc in result['source_documents']:
                st.markdown(f"**Document: {doc.metadata['pdf_name']}**")
                st.text_area("", value=doc.page_content, height=150)


