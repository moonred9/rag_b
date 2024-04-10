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
from .index_retriever.hnsw_retriever import HNSWIndexRetriever

class HNSWRetriever(ABC):
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
            is_create_vecor_index: bool = False
        ):
        self.docs_directory = docs_directory
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k
        self.is_create_vector_index = is_create_vecor_index
        

        if construct_index:
            self.construct_index()
        else:
            self.load_index_from_milvus()
        
        if add_index:
            self.add_index()

        # self.query_engine = self.vector_index.as_query_engine()
        retriever = HNSWIndexRetriever(
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
            "index_type":"HNSW",
            "params":{"M":16, 'efConstruction':32}
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
        
        
        # vector_store_dict = self.vector_index.storage_context.index_store.to_dict()
        # print(vector_store_dict)
        # docs = vector_store_dict['index_store/data'].values()
        # print(docs)
        # docs0 = vector_store_dict['index_store/data'].values()
        # print(docs0)
        # docs1 = list(vector_store_dict['index_store/data'].values())[0]['__data__']
        # print(docs1)
        # docs2 = docs1['doc_id_dict']
        # print(docs2

        





# from llama_index.retrievers import BaseRetriever

# class MyHNSWRetriever(BaseRetriever):
#     def __init__(self,dimension, max_elements_num, space = 'l2', M = 16, ef_construction = 200):
#         self.ann = hnswlib.Index(space = space, dim = dimension)
#         self.ann.init_index(max_elements = max_elements_num, ef_construction = ef_construction, M = M)


#     def add(self, nodes_with_embeddings):
#         for node in nodes_with_embeddings:
#             self.ann.add_item(node.embedding)

#     def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
#         """Query index for top k most similar nodes.

#         Args:
#             query_embedding (List[float]): query embedding
#             similarity_top_k (int): top k most similar nodes
#             doc_ids (Optional[List[str]]): list of doc_ids to filter by
#             node_ids (Optional[List[str]]): list of node_ids to filter by
#             output_fields (Optional[List[str]]): list of fields to return
#             embedding_field (Optional[str]): name of embedding field
#         """
#         if query.mode != VectorStoreQueryMode.DEFAULT:
#             raise ValueError(f"Milvus does not support {query.mode} yet.")

#         expr = []
#         output_fields = ["*"]

#         # Parse the filter
#         if query.filters is not None:
#             expr.extend(_to_milvus_filter(query.filters))

#         # Parse any docs we are filtering on
#         if query.doc_ids is not None and len(query.doc_ids) != 0:
#             expr_list = ['"' + entry + '"' for entry in query.doc_ids]
#             expr.append(f"{self.doc_id_field} in [{','.join(expr_list)}]")

#         # Parse any nodes we are filtering on
#         if query.node_ids is not None and len(query.node_ids) != 0:
#             expr_list = ['"' + entry + '"' for entry in query.node_ids]
#             expr.append(f"{MILVUS_ID_FIELD} in [{','.join(expr_list)}]")

#         # Limit output fields
#         if query.output_fields is not None:
#             output_fields = query.output_fields

#         # Convert to string expression
#         string_expr = ""
#         if len(expr) != 0:
#             string_expr = " and ".join(expr)

#         # Perform the search
#         res = self.milvusclient.search(
#             collection_name=self.collection_name,
#             data=[query.query_embedding],
#             filter=string_expr,
#             limit=query.similarity_top_k,
#             output_fields=output_fields,
#             search_params=self.search_config,
#         )

#         logger.debug(
#             f"Successfully searched embedding in collection: {self.collection_name}"
#             f" Num Results: {len(res[0])}"
#         )

#         nodes = []
#         similarities = []
#         ids = []

#         # Parse the results
#         for hit in res[0]:
#             if not self.text_key:
#                 node = metadata_dict_to_node(
#                     {
#                         "_node_content": hit["entity"].get("_node_content", None),
#                         "_node_type": hit["entity"].get("_node_type", None),
#                     }
#                 )
#             else:
#                 try:
#                     text = hit["entity"].get(self.text_key)
#                 except Exception:
#                     raise ValueError(
#                         "The passed in text_key value does not exist "
#                         "in the retrieved entity."
#                     )
#                 node = TextNode(
#                     text=text,
#                 )
#             nodes.append(node)
#             similarities.append(hit["distance"])
#             ids.append(hit["id"])

#         return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
    
