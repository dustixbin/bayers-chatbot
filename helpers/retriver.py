from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
from .llama_parse_pdf import extract_pdf_llamaparse
from .chunker import text_n_images 
Settings.llm = None
from sqlalchemy import make_url
import os
from sqlmodel import Session
from llama_index.core.response_synthesizers import CompactAndRefine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# load_dotenv(".env")

embedding_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

llm_model = OpenAI(model="gpt-3.5-turbo")

class Retriver():
    def __init__(self, document_id, path=None, embedding_model=embedding_model):

        self.connection_string = os.getenv('PGVECTOR_HOST')
        self.url = make_url(self.connection_string)
        self.db_name = "bayers-dev"
        self.path = path
        self.embedding_model = embedding_model
        self.document_id = document_id
        self.vector_store = PGVectorStore.from_params(
            database=self.db_name,
            host=self.url.host,
            password=self.url.password,
            port=self.url.port,
            user=self.url.username,
            hybrid_search=True,
            table_name=self.document_id,
            embed_dim=1536, 
            hnsw_kwargs={
                "hnsw_m": 16,
                "hnsw_ef_construction": 64,
                "hnsw_ef_search": 40,
                "hnsw_dist_method": "vector_cosine_ops",
            },
        )

    
    async def extract_text_from_pdf(self, session):
        data, num_images, num_tables = await extract_pdf_llamaparse(self.path)
        # print("unstructured", data ,num_images, " ", num_tables)
        docs = text_n_images(data, self.document_id, session)
        # print("docs")
        return docs

    async def upsert(self, session : Session):
        docs = await self.extract_text_from_pdf(session)
        storage_context = StorageContext.from_defaults(vector_store= self.vector_store)
        index = VectorStoreIndex.from_documents(
            docs, 
            embed_model=self.embedding_model, 
            vector_store=self.vector_store, 
            storage_context=storage_context,
            show_progress=True
        )

    def similarity_search(self, query, k=12):
        try:
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store, 
                embed_model=self.embedding_model, 
                show_progress=True
            )

            retriever = index.as_retriever(similarity_top_k=k)
            vector_retriever = index.as_retriever(
                vector_store_query_mode="default",
                similarity_top_k=5,
            )
            text_retriever = index.as_retriever(
                vector_store_query_mode="sparse",
                similarity_top_k=5,  
            )
            retriever = QueryFusionRetriever(
                [vector_retriever, text_retriever],
                similarity_top_k=5,
                num_queries=1,
                mode="relative_score",
                use_async=False,
            )

            response_synthesizer = CompactAndRefine()
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
                response_synthesizer=response_synthesizer,
            )
            
            nodes = query_engine.retrieve(query)
            return nodes
        
        except Exception as e:
            print(f"[WARNING] Retrieval failed for document ID {self.document_id}: {e}")
            return []

    def delete_chunks(self, ids):
        self.vector_store.delete(ids)

    def delete_collection(self):
        self.vector_store.clear()

if __name__ == "__main__":

    test = Retriver(embedding_model=embedding_model,document_id="gaga_test_2", path="/workspaces/bayers-usecase/data/test.pdf")
    # test.upsert()clear

    print(test.similarity_search("What are the High-scale and-quality Data Limitations"))
    