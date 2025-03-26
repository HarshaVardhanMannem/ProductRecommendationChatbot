import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load datasets
df_products = pd.read_csv('amazon_products.csv')
df_categories = pd.read_csv('amazon_categories.csv')

# Merge datasets
df_amazon = df_products.merge(df_categories, left_on='category_id', right_on='id', how='left')

# Filter products with stars > 3
relevant_columns = ['title', 'productURL', 'stars', 'price', 'category_name']
df_Amazon = df_amazon[relevant_columns]
df_filtered = df_Amazon[df_Amazon['stars'] > 3]

# Sample up to 100 products per category_name
df_sampled = df_filtered.groupby('category_name', group_keys=False).apply(lambda x: x.sample(min(3, len(x)), random_state=42))
df_Amazon = df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Chroma vector store
vector_store = Chroma(persist_directory="ecommerce_db", embedding_function=embeddings)

# Function to process and store data
def process_and_store(df, vector_store):
    json_data = df.to_json(orient="records", indent=2)
    docs = [Document(page_content=json_data)]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(docs)
    vector_store.add_documents(documents=all_splits)

# Process and store Amazon data
process_and_store(df_Amazon, vector_store)


