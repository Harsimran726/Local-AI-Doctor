# from helper_utils import load_chromadb , word_wrap 
from http import client

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter , SentenceTransformersTokenTextSplitter
import chromadb
from pathlib import Path
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import umap 
from tqdm import tqdm
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
load_dotenv()
from sentence_transformers import CrossEncoder
from langgraph.types import RetryPolicy
import os


class RAG:
    def __init__(self,embedding_model="all-MiniLM-L6-v2", collection_name="Local_Doctor"):
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.llm = None 
        self.temperature = 0.7
        self.max_tokens = 1000 
        self.query = None
    
    # it uses for the QUERY EXPENSION 
    def set_llm(self,x):
        try:

            client = OpenAI(api_key=os.getenv("openai_api_key"))
            print(f"LLM CLIENT CREATED {x} type {type(x)}")
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role":"system","content": "You are a helpful expert Doctor research assistant for a medical ontolog. Provide an example answer to the given question, that might be found in a document of ICD-11 + ORDO rare disease ontology . Output should be in following format {'query': 'expended query'}"},
                    {"role":"user","content":x}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            # print(f"RESPONSE FROM THE LLM:- {response}")
            response = response.choices[0].message.content
            print(f"RESONSE FROM THE LLM:- {response}")
            return response 
        except Exception as e:
            print(f"Error in set_llm: {e}")
            return None
        # we try to use the open ai 

    def extract_text_from_pdf(self,pdf_path):
        try:
            print(f"Extracting text from PDF: {pdf_path}")
            for i in range(len(pdf_path)):
                print(f"READING PDF")
                try:
                    reader = PdfReader(pdf_path[i])
                    # print(f"READING PAGE {i} {reader}")
                    pdf_text = [p.extract_text().strip() for p in reader.pages]
                    # print(f"PDF TEXT:- \n {pdf_text}")
                    # filter the empty texts 
                    print(f"PROCESSING")
                    pdf_text = [text for text in pdf_text if text]
                    charactersplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=["\n\n", "\n", " ", ""])
                    charactersplitter_text = charactersplitter.split_text('\n\n'.join(pdf_text))

                    token_splitter = SentenceTransformersTokenTextSplitter(model_name="all-MiniLM-L6-v2", tokens_per_chunk=256, chunk_overlap=0)
                    
                    token_split_text = []
                    for text in tqdm(charactersplitter_text):
                        token_split_text.extend(token_splitter.split_text(text)
                                                )
                except Exception as e:
                    print(f"Error reading PDF {pdf_path[i]}: {e}")
                    continue
                # rec_text =textsplitter.split_text(pdf_text)
            print(f"EXTRACTION COMPLETE")
            print(f"Number of chunks created: {len(token_split_text)}")
            return token_split_text

        except Exception as e:
            return None

    def embedding(self):
        try:
            print(f"INSIDE THE EMBEDDING")
            embedding_function = SentenceTransformerEmbeddingFunction(model_name=self.embedding_model)
            return embedding_function
        except Exception as e:
            print(f"Error in embedding function: {e}")
            return None

    def chromadb_client(self):
        try:
            client = chromadb.PersistentClient(path="./chroma_db")
            return client
        except Exception as e:
            return None

    def cross_encoder(self,document, query):
        try:
            print(f"INSIDE THE CROSS ENCODER")

            cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            scores = cross_encoder.predict([(query, doc) for doc in document[0]])
            print(f"CROSS ENCODER TOP DOCS :- {scores}")
            for o in np.argsort(scores)[::-1]:
                print(o+1)
            return scores 
        
        except Exception as e:
            print(f"Error in cross_encoder: {e}")
            return None


    def chromadb_collection(self, rec_text):
        try:
            print(f"CONNECTION WITH CLIENT")
            client = self.chromadb_client()
            print(f"SUCCESFULLUY CONNETED ")

            if client is None:
                print("Failed to create ChromaDB client.")
                return None
            
            print(f"CREATING COLLECTION")
            collection = client.create_collection(name="Local_Doctor_Medicine",embedding_function=self.embedding()) # Local_Doctor_Disease
            print(f"COLLECTION CREATED:- {collection}")
            print(f"REC TEXT :- \n {rec_text}")
            ids = [str(i) for i in range(len(rec_text))]
            batch_size = 5461
            for i in tqdm(range(0, len(rec_text), batch_size)):
                print(f"ADDING DOCUMENTS TO THE COLLECTION {collection}")
                collection.add(
                ids=ids[i:i+batch_size],
                documents=rec_text[i:i+batch_size]
            )
            print(f"Number of documents in the collection: {collection.count()}")

            return collection
        except Exception as e:
            print(f"Error in chromadb_collection: {e}")
            return None
    



        

class AskQuery:
    def __init__(self):
        self.client = RAG().chromadb_client()
        self.rag = RAG()

    def disease_collection(self,x):
        try:
        
            client = self.client.chromadb_client().get_collection(name="Local_Doctor_Disease")
            expended_query = self.rag.set_llm(x)
            if expended_query is None:
                print("Failed to expand the query using LLM.")
                expended_query = x  # Fallback to the original query if LLM fails 
            results = client.query(
                query_texts=[expended_query],
                n_results=20,
                include=['documents', 'distances', 'metadatas', 'embeddings']
            )

            # now passe to the cross encoder for reranking 

            scores = self.rag.cross_encoder(results['documents'], expended_query)

            if scores is not None:
                # filter the only positive scores that contain more than threshold  value 
                threashold = 0.7 # that should be like 70 out of 100 
                top_docs = [doc for doc, score in zip(results['documents'], scores) if score > threashold]
                if not top_docs:
                    print("No documents found with a score above the threshold.")
                    return None  # Fallback to top 5 documents if no scores are above the threshold
                print(f"TOP DOCS AFTER CROSS ENCODER RERANKING:- {top_docs}") 
                return top_docs 
            else :
                return None

        except Exception as e:
            return None

    def laws_collection(self,x):
        try:
            client = self.client.chromadb_client().get_collection(name="Local_Doctor_Laws")
            expended_query = self.rag.set_llm(x)
            if expended_query is None:
                print("Failed to expand the query using LLM.")
                expended_query = x  # Fallback to the original query if LLM fails 
            results = client.query(
                query_texts=[expended_query],
                n_results=20,
                include=['documents', 'distances', 'metadatas', 'embeddings']
            )

            # now passe to the cross encoder for reranking 

            scores = self.rag.cross_encoder(results['documents'], expended_query)

            if scores is not None:
                # filter the only positive scores that contain more than threshold  value 
                threashold = 0.7 # that should be like 70 out of 100 
                top_docs = [doc for doc, score in zip(results['documents'], scores) if score > threashold]
                if not top_docs:
                    print("No documents found with a score above the threshold.")
                    return None  # Fallback to top 5 documents if no scores are above the threshold
                print(f"TOP DOCS AFTER CROSS ENCODER RERANKING:- {top_docs}") 
                return top_docs 
            else :
                return None

        except Exception as e:
            return None
        
    def medicine_collection(self,x):
        try:
            client = self.client.chromadb_client().get_collection(name="Local_Doctor_Medicine")
            expended_query = self.rag.set_llm(x)
            if expended_query is None:
                print("Failed to expand the query using LLM.")
                expended_query = x  # Fallback to the original query if LLM fails 
            results = client.query(
                query_texts=[expended_query],
                n_results=20,
                include=['documents', 'distances', 'metadatas', 'embeddings']
            )

            # now passe to the cross encoder for reranking 

            scores = self.rag.cross_encoder(results['documents'], expended_query)

            if scores is not None:
                # filter the only positive scores that contain more than threshold  value 
                threashold = 0.7 # that should be like 70 out of 100 
                top_docs = [doc for doc, score in zip(results['documents'], scores) if score > threashold]
                if not top_docs:
                    print("No documents found with a score above the threshold.")
                    return None  # Fallback to top 5 documents if no scores are above the threshold
                print(f"TOP DOCS AFTER CROSS ENCODER RERANKING:- {top_docs}") 
                return top_docs 
            else :
                return None

        except Exception as e:
            return None



# def ask_query():
#     try:
#         rag = RAG()

#         # Define the path to your folder
#         folder_path = Path("C:\Data\Projects\Local Doctor\material\Medicines")

#         # Use .glob() to find all PDF files and extract their names
#         pdf_names = [str(folder_path / file.name) for file in folder_path.glob("*.pdf")]

#         print(f"Total PDFs found: {(pdf_names)}\n")

#         # Print each file name
#         # for name in pdf_names:
#         #     print(name)

#         rec_text = rag.extract_text_from_pdf(pdf_names)
#         collection =  rag.chromadb_collection(rec_text)

        # collection = rag.chromadb_client().get_collection(name="Local_Doctor_Disease")
        # print(f"COLLECTION RETRIEVED:- {collection} :- COllection COUNT {collection.count()}")
        # query = " Autosomal recessive chorioretinopathy-microcephaly syndrome "
        # query_expended = rag.set_llm(query)
        # # print(f"QUERY EXPENDED:- {query_expended}")

        # # print(f"QUERYING THE COLLECTION")
        # results = collection.query(
        #     query_texts=[query_expended],
        #     n_results=10,
        #     include=['documents', 'distances', 'metadatas', 'embeddings'] 
        # )

        # top_docs = rag.cross_encoder(results['documents'], query_expended)

        # # print(f"EMBEDDING THE COLLECTION:- {results}")
        # embeddings = collection.get(include=['embeddings'])['embeddings']
        # umap_transform = umap.UMAP(random_state=42,transform_seed=0).fit(embeddings)
        # try:
        #     embedding_function = SentenceTransformerEmbeddingFunction()
        #     # embedding_function = rag.embedding()
        #     query_embedding = embedding_function(query)[0]
        #     query_expended_embedding = embedding_function(query_expended)[0]
        #     # print(f"QUERY EMBEDDING:- {query_embedding}")
            
        # except Exception as e:
        #     print(f"Error in embedding the query: {e}")
        #     query_embedding = None  
        # try:
        #     retrival_embedding = results['embeddings'][0]
        #     print(f"RETrieval EMBEDDING:- {retrival_embedding}")
        # except Exception as e:
        #     print(f"Error in retrieving the embedding: {e}")
        #     retrival_embedding = None
        # return embeddings, umap_transform , query_embedding, retrival_embedding, query_expended_embedding
#     except Exception as e:
#         print(f"Error in ask_query: {e}")
#         return None, None , None, None

# embeddigns , umap_transform , query_embedding, retrival_embedding, query_expended_embedding = ask_query() # , 



# import numpy as np
# def project_embeddigns(embeddings,umap_transform):
#     try:
#         print(f"PROJECTING THE EMBEDDINGS")
#         umap_embeddings = np.empty((len(embeddings), 2))
#         for i,embedding in enumerate(tqdm(embeddings)):
#             umap_embeddings[i] = umap_transform.transform([embedding])
#         return umap_embeddings
#     except Exception as e:
#         print(f"Error in project_embeddigns: {e}")

# # embedding projection 
# projected_dataset_embedding = project_embeddigns(embeddigns,umap_transform)
# query_dataset_embedding = project_embeddigns([query_embedding],umap_transform)
# retrival_dataset_embedding = project_embeddigns(retrival_embedding,umap_transform)
# query_expended_dataset_embedding = project_embeddigns([query_expended_embedding],umap_transform)
# import matplotlib.pyplot as plt

# # put the plots

# plt.figure(figsize=(10, 8))
# plt.scatter(projected_dataset_embedding[:, 0], projected_dataset_embedding[:, 1], s=10)
# plt.scatter(query_dataset_embedding[:, 0], query_dataset_embedding[:, 1], color='red', s=100, marker='X', label='Query Embedding')
# plt.scatter(retrival_dataset_embedding[:, 0], retrival_dataset_embedding[:, 1], color='green', marker='o', s=70, label='Retrieval Embedding')
# plt.scatter(query_expended_dataset_embedding[:, 0], query_expended_dataset_embedding[:, 1], color='blue', marker='X', s=100, label='Query Expended Embedding')
# plt.gca().set_aspect('equal', adjustable='box')
# plt.title('UMAP projection of the dataset embeddings', fontsize=24)
# plt.axis('off')
# plt.show()
# # save the image 
# plt.savefig("umap_projection.png", dpi=300)



# # i will use the Cross Encoder Model with reranking + llm as a query. 
