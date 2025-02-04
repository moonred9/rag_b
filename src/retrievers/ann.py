from abc import ABC

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index.node_parser import SimpleNodeParser
from llama_index import download_loader

from llama_index.embeddings import LangchainEmbedding
from llama_index import ServiceContext, StorageContext
from langchain.schema.embeddings import Embeddings
from llama_index.vector_stores import MilvusVectorStore

from pymilvus import MilvusClient, Collection



# from .hnsw_retriever import HNSWIndexRetriever
from .index_retriever.diskann_retriever import DISKANNIndexRetriever

class ANNRetriever(ABC):
    def __init__(
            self, 
            docs_directory: str, 
            embed_model: Embeddings,
            embed_dim: int = 768,
            chunk_size: int = 128,
            chunk_overlap: int = 0,
            collection_name: str = "docs",
            construct_index: bool = False,
            add_index: bool = False,
            similarity_top_k: int=2,
            is_create_vecor_index: bool = False,
            index_type: str = "HNSW",
            index_param: dict = {},
            search_param: dict = {}
        ):
        self.docs_directory = docs_directory
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k
        self.is_create_vector_index = is_create_vecor_index
        self.index_type = index_type
        self.index_param = index_param
        self.search_param = search_param
        
        if index_type == 'FLAT':
            if not self.index_param:
                self.index_param = {
                "metric_type":"L2",
                "index_type":"FLAT"
            }
            if not self.search_param:
                self.search_param = {
                    'metric_type': 'L2'
                }
        elif index_type == 'IVF_FLAT':
            if not self.index_param:
                self.index_param = {
                    'metrivic_type':'L2',
                    'index_type':'IVF_FLAT',
                    "params":{"nlist":1024}
                }
            if not self.search_param:
                self.search_param = {
                    "nprobe": 128,
                }
        elif index_type == 'IVF_PQ':
            if not self.index_param:
                self.index_param = {
                    'metrivic_type':'L2',
                    'index_type':'IVF_PQ',
                    "params":{"nlist":1024, "m":32}
                }
            if not self.search_param:
                self.search_param = {
                    "nprobe": 128,
                }
        elif index_type == 'IVF_SQ8':
            if not self.index_param:
                self.index_param = {
                    'metrivic_type':'L2',
                    'index_type':'IVF_SQ8',
                    "params":{"nlist":1024}
                }
            if not self.search_param:
                self.search_param = {
                    "nprobe": 128,
                    "reorder_k":32,
                }
        elif index_type == 'HNSW':
            if not self.index_param:
                self.index_param = {
                    'metrivic_type':'L2',
                    'index_type':'HNSW',
                    "params":{"M":16, 'efConstruction':32}
                }
            if not self.search_param:
                self.search_param = {
                    "ef": 128,
                }
        elif index_type == 'SCANN':
            if not self.index_param:
                self.index_param = {
                "metric_type":"L2",
                "index_type":"SCANN",
                "params":{"nlist":1024}
            }
            if not self.search_param:
                self.search_param = {
                "nprobe": 128,
                "reorder_k":32,
            }
        elif index_type == 'DISKANN':
            if not self.index_param:
                self.index_param = {
                "metric_type":"L2",
                "index_type":"SCANN",
            }
            if not self.search_param:
                self.search_param = {
                "search_list": 128,
                }
        if construct_index:
            self.construct_index()
        else:
            self.load_index_from_milvus()
        
        if add_index:
            self.add_index()

        # self.query_engine = self.vector_index.as_query_engine()
        retriever = DISKANNIndexRetriever(
            index=self.vector_index,
            similarity_top_k=self.similarity_top_k,
        )

        # assemble query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
        )

    def construct_index(self):
        documents = SimpleDirectoryReader(self.docs_directory).load_data()
        
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        
        self.embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        vector_store = MilvusVectorStore(
            dim=self.embed_dim, overwrite=True,
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Process and index nodes in chunks due to Milvus limitations
        for spilt_ids in range(0, len(nodes), 8000):  
            self.vector_index = GPTVectorStoreIndex(
                nodes[spilt_ids:spilt_ids+8000], service_context=service_context, 
                storage_context=storage_context, show_progress=True
            )
            print(f"Indexing of part {spilt_ids} finished!")

            vector_store = MilvusVectorStore(
                overwrite=False,
                collection_name=self.collection_name
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print("Indexing finished!")

    def add_index(self):
        if self.docs_type == 'json':
            JSONReader = download_loader("JSONReader")
            documents = JSONReader().load_data(self.docs_directory)
        else:
            documents = SimpleDirectoryReader(self.docs_directory).load_data()
        
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        
        self.embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        vector_store = MilvusVectorStore(
            overwrite=False,
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

         # Process and index nodes in chunks due to Milvus limitations
        for spilt_ids in range(0, len(nodes), 8000):  
            self.vector_index = GPTVectorStoreIndex(
                nodes[spilt_ids:spilt_ids+8000], service_context=service_context, 
                storage_context=storage_context, show_progress=True
            )
            print(f"Indexing of part {spilt_ids} finished!")

        print("Indexing finished!")

    def load_index_from_milvus(self):
        if self.is_create_vector_index:
            self.create_vector_index()
        else:
            client = MilvusClient(
                uri="http://localhost:19530"
            )
            collection = Collection(name=self.collection_name, using=client._using)
            collection.load()
        vector_store =  MilvusVectorStore(
            overwrite=False, dim=self.embed_dim, 
            collection_name=self.collection_name
        )
        
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        service_context = ServiceContext.from_defaults(embed_model=self.embed_model, llm=None)
        self.vector_index = GPTVectorStoreIndex(
            [], storage_context=storage_context, 
            service_context=service_context,
        )
        # connections.connect("default", host="localhost", port="19530")
        
        
        
    def create_vector_index(self):
        client = MilvusClient(
            uri="http://localhost:19530"
        )
        collection = Collection(name=self.collection_name, using=client._using)
        collection.release()
        collection.drop_index()
        index_params = {
            "metric_type":"L2",
            "index_type":"DISKANN",
        }
        field_name = 'embedding'
        collection.create_index(
            field_name=field_name, 
            index_params=index_params
        )
        collection.load()
        client.close()


    def search_docs(self, query_text: str):
        response_vector = self.query_engine.query(query_text)
        
        
        
        response_text_list = response_vector.response.split('\n---------------------\n')
        response_text = response_text_list[1].split("\n\n")
        response_text = "\n\n".join([text for text in response_text if not text.startswith("file_path: ")])
        
        return response_text