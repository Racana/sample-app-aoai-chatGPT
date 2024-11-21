from pymongo import MongoClient
from openai import AzureOpenAI
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pymupdf
from langchain.text_splitter import TokenTextSplitter

@dataclass
class Document(object):
    """A data class for storing documents in a MongoDB database.

    Attributes:
        content (str): The content of the document.
        id (Optional[str]): The id of the document.
        filepath (Optional[str]): The filepath of the document.
        metadata (Optional[Dict]): The metadata of the document.
        content (Optional[List[float]]): The content of the document.
    """
    content: str
    id: Optional[str] = None
    title: Optional[str] = None
    filepath: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[Dict] = None
    contentVector: Optional[List[float]] = None

class CosmosDBRAG:
    def __init__(self, 
                 connection_string: str,
                 database_name: str,
                 collection_name: str,):
        self.collection_name = collection_name        
        self.client = MongoClient(connection_string)
        self.database = self.client[database_name]
        self.collection = self.database[collection_name]

    def create_document(self, document: Document) -> None:
        """Create a document in the MongoDB database.

        Args:
            document (Document): The document to create.

        Returns:
            None
        """
        self.collection.insert_one(document.dict())

    def create_or_update_vector_index(self,
                                      index_name: str, 
                                      vector_field: str,
                                      dimensions: int = 1536,
                                      num_lists: int = 30,
                                      ) -> None:
        """Create or update a vector index in the MongoDB database.

        Args:
            index_name (str): The name of the index.
            vector_field (str): The field containing the vector data.
            dimensions (int, optional): The number of dimensions in the vector. Defaults to 1536.
            num_lists (int, optional): The number of lists in the index. Defaults to 30. (Current document size for test is 1000 documents, we define num_list as 30 as sqrt(1000)≈30
        Returns:
            None
        """
        indexes = self.collection.list_indexes()
        
        if not any(index['name'] == index_name for index in indexes):
            indexDefs:List[any] = [
                {
                    "name": index_name, 
                    "key": 
                        {vector_field: "cosmosSearch"}, 
                    "cosmosSearchOptions": 
                        {"kind": "vector-ivf",
                        "similarity": "COS",
                        "dimensions": dimensions,
                        "numLists": num_lists}
                }
            ]
            
            self.database.command("createIndexes", self.collection_name, indexes = indexDefs)
        else:
            print("Index already exists")
        
    def get_documents(self, query: Dict[str, Any]) -> List[Document]:
        """Get documents from the MongoDB database.

        Args:
            query (Dict[str, Any]): The query to filter the documents.

        Returns:
            List[Document]: The list of documents that match the query.
        """
        documents = self.collection.find(query)
        return [Document(**document) for document in documents]

    def verify_file_exists(self, file_path: str) -> bool:
        """Verify if a file exists in the MongoDB database.

        Args:
            file_path (str): The path of the file.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return self.collection.find_one({"metadata.pdf_path": file_path}) is not None
    
    def similarity_search(self, 
                          query_embedding: List[float],
                          k: int = 4,
                          ) -> List[Document]:
        """Perform a similarity search on the MongoDB database.

        Args:
            query_embedding (List[float]): The embedding of the query.
            k (int, optional): The number of results to return. Defaults to 4.

        Returns:
            List[Document]: The list of documents that match the query.
        """
        query = [
            {
                "$search": {
                    "cosmosSearch": {
                        "vector": query_embedding,
                        "path": "vector_content",
                        "k": k,
                    },
                    "returnStoredSource": True
                }
            },
            {
                "$project": {
                    "simScore": {"$meta": "searchScore"},
                    "content": 1,
                    "metadata": 1,
                    "_id": 0
                }
            }
        ]
        results = self.collection.aggregate(query)
        
        return list(results)


class ChatClient:
    def __init__(self, 
                 azure_endpoint: str,
                 api_key: str,
                 api_version: str = "2024-08-01-preview",
                 chat_model: str = "gpt-4o-mini",
                 embedding_model: str = "text-embedding-3-small"
                 ) -> None:
    
        self.openai_client = AzureOpenAI(azure_endpoint=azure_endpoint, 
                                         api_version=api_version, 
                                         api_key=api_key)
        self.chat_model = chat_model
        self.embedding_model = embedding_model

    def chat_completion(self, 
                        user_prompt: str,
                        documents: List[Dict],
                        temperature: float = 1,
                        max_tokens: int = 1000,
                        top_p:float = 1,
                        ) -> str:
        """Create an instance of the AzureOpenAI class.

        Returns:
            AzureOpenAI: An instance of the AzureOpenAI class.
        """
        system_prompt = """You are an intelligent assistant for NLP books. You are designed to provide helpful answers to user questions about NLP in your database.
    You are friendly, helpful, and informative and can be lighthearted. Be concise in your responses, but still friendly.
    - Only answer questions related to the information provided below.
    - The documents provided specify the page on which the document provided is extracted. Ensure that the page are provided in the response."""
        messages = [{'role': 'system', 'content': system_prompt}]

        messages.append({'role': 'user', 'content': user_prompt})

        for result in documents:
            rag_content = f"Page: {result['metadata']['page_span']}\n\n Content:{result['content']}"
            messages.append({'role': 'system', 'content': rag_content})

        response = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
            )

        return response.choices[0].message.content
    
    def generate_embeddings(self, 
                            text: str
                            ) -> List[float]:
        """Generate embeddings for a given text.

        Args:
            text (str): The text to generate embeddings for.

        Returns:
            List[float]: The embeddings for the given text.
        """
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
            )
        return response.data[0].embedding


class PDFChunk:
    def __init__(self, 
                 pdf_path: str):
        self.pdf_path = pdf_path
        self.document = pymupdf.open(pdf_path)

    def extract_and_chunk_pdf(self,
                              chunk_size: int = 520,
                              chunk_overlap: int = 20
                              ) -> List[Document]:
        """
        Extract text from a PDF file and split it into chunks across pages while maintaining page metadata.
        
        Args:
            chunk_size (int): Maximum number of tokens per chunk
            chunk_overlap (int): Number of tokens to overlap between chunks
        
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing chunks and their metadata
        """

        try:
            doc = self.doc
            # Initialize the text splitter
            text_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )            
            # Initialize list to store chunks with metadata
            chunks_with_metadata = []
            
            # Create a list of tuples containing (text, page_number)
            text_with_pages: List[Tuple[str, int]] = []
            
            # First, collect all text with corresponding page numbers
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():  # Skip empty pages
                    text_with_pages.append((text, page_num + 1))
            
            # Concatenate all text while keeping track of page boundaries
            full_text = ""
            page_boundaries = []  # List of (character_index, page_number)
            current_char_index = 0
            
            for text, page_num in text_with_pages:
                full_text += text
                page_boundaries.append((current_char_index, page_num))
                current_char_index += len(text)
            
            # Split the concatenated text into chunks
            chunks = text_splitter.split_text(full_text)
            
            # Process each chunk to determine which pages it spans
            current_pos = 0
            for chunk in chunks:
                chunk_start = full_text.find(chunk, current_pos)
                chunk_end = chunk_start + len(chunk)
                
                # Find which pages this chunk belongs to
                chunk_pages = set()
                for i, (boundary_pos, page_num) in enumerate(page_boundaries):
                    next_boundary = len(full_text) if i == len(page_boundaries) - 1 else page_boundaries[i + 1][0]
                    
                    # If there's any overlap between the chunk and this page's text
                    if chunk_start < next_boundary and chunk_end > boundary_pos:
                        chunk_pages.add(page_num)
                
                chunk_data = Document(
                    content= chunk,
                    metadata= {
                        "pages": sorted(list(chunk_pages)),  # List of pages this chunk spans
                        "page_span": f"{min(chunk_pages)}-{max(chunk_pages)}" if len(chunk_pages) > 1 else str(min(chunk_pages)),
                        "pdf_path": self.pdf_path,
                        "total_pages": len(doc),
                    }
                )
                chunks_with_metadata.append(chunk_data)
                current_pos = chunk_start + 1  # Update position for next search
            
            doc.close()
            return chunks_with_metadata
        
        except Exception as e:
            print(f"Error processing PDF {self.pdf_path}: {str(e)}")
            raise