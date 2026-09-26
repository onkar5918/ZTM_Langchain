import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL='text-embedding-3-small'
GPT_MODEL='gpt-5.4'
api_key=os.getenv('OPENAI_API_KEY')

#import essential libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.vectorstores import FAISS
from IPython.display import display, Markdown

#import libraries for document loading
import pandas as pd
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredPDFLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain_core.documents import Document

df=pd.read_excel('Data//Reviews.xlsx')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#one document /chunk per row
chunks=[
    Document(page_content='\n'.join(f'{col}:{val}'for col, val in row.items()),
             metadata={"row_index":int(i)}
             )
for i ,row in df.iterrows()
]


#load the embedding model
embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL)

df_faiss=FAISS.from_documents(chunks,embeddings)
df_faiss

#Test the retrieval system
query='Gie me most positive reviews'

#Retrieve the content
retrieved_docs=df_faiss.similarity_search_with_score(query,k=3)
print(retrieved_docs)