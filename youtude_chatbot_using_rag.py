
import os


"""##install libraries"""

!pip install -q youtube-transcript-api langchain-community langchain-openai \ faiss-cpu tiktoken python-dotenv

!pip show youtube-transcript-api

from youtube_transcript_api import YouTubeTranscriptApi
print(YouTubeTranscriptApi)

from youtube_transcript_api import YouTubeTranscriptApi

YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])

!pip uninstall youtube-transcript-api -y
!pip uninstall youtube_transcript_api -y

!pip install --upgrade youtube-transcript-api

from youtube_transcript_api import YouTubeTranscriptApi

print(dir(YouTubeTranscriptApi))

!pip install langchain_groq

"""##import libraries"""

from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

"""##1.Indexing -> YoutubeTranscript"""

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

video_id = "VITy7mDXHsQ"

try:
  api = YouTubeTranscriptApi() # Instantiate the API
  transcript_list_obj = api.list(video_id) # Call instance method

  # Find the English transcript and then fetch its data
  transcript = transcript_list_obj.find_transcript(language_codes=['en'])
  data = transcript.fetch()

  text = " ".join(chunk.text for chunk in data)
  print(text)

except TranscriptsDisabled:
  print("No caption available for this video")
except Exception as e:
  print(f"An error occurred: {e}")

print(data)

print(len(text))

"""##1. indexing -> Text Splitting"""

splitter = RecursiveCharacterTextSplitter(chunk_size = 100 , chunk_overlap = 20)
chunks = splitter.create_documents([text])

len(chunks)

chunks[167]

"""##1. indexing -> Embedding & Vector Store FAISS"""



!pip install huggingface_hub

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
import os
from typing import List
import numpy as np # Import numpy

# Instantiate the HuggingFace Inference Client
client = InferenceClient(api_key=os.environ["HF_TOKEN"])

# Define the model ID for embeddings
model_id = "BAAI/bge-small-en-v1.5"

# Define a custom Embeddings class to integrate with HuggingFace InferenceClient
class CustomHuggingFaceInferenceAPIEmbeddings(Embeddings):
    """Custom Embeddings class to use HuggingFace Inference Client for embeddings."""
    def __init__(self, client: InferenceClient, model_id: str):
        self.client = client
        self.model_id = model_id

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # The InferenceClient's feature_extraction method can handle a list of strings.
        # It typically returns a numpy array, which we convert to a list of lists.
        # Ensure the output is a list of lists for compatibility with FAISS.
        embeddings_list = []
        for text_chunk in texts:
            try:
                embedding = self.client.feature_extraction(text_chunk, model=self.model_id)
                if isinstance(embedding, np.ndarray):
                    embeddings_list.append(embedding.tolist())
                else:
                    # Handle cases where the API might return non-numpy format
                    embeddings_list.append(embedding) # Assume it's already a list or convertible
            except Exception as e:
                print(f"Error generating embedding for text chunk: {text_chunk[:50]}... Error: {e}")
                embeddings_list.append([0.0] * 768) # Append a zero vector or handle error as appropriate
        return embeddings_list

    def embed_query(self, text: str) -> List[float]:
        # For a single query, it returns a single embedding (numpy array).
        # Convert it to a list.
        embedding = self.client.feature_extraction(text, model=self.model_id)
        return embedding.tolist()

# Instantiate our custom embeddings class
hf_embeddings_model = CustomHuggingFaceInferenceAPIEmbeddings(client, model_id)

# Now, create the FAISS vector store using the documents and our embedding model.
# FAISS.from_documents will internally call embed_documents on hf_embeddings_model.
vector_store = FAISS.from_documents(chunks, hf_embeddings_model)

vector_store.index_to_docstore_id

"""##2.Retriever"""

retriever = vector_store.as_retriever(search_type="similarity",search_kwargs = {"k":4})

retriever

retriever.invoke("who is Pakeeza")



"""##3.Augumentation"""

model = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"),
        model = "openai/gpt-oss-120b")

prompt = PromptTemplate(
    template = """
     You are helpful AI assistant.Answer only from the provided trascript content.
     If the only is unsufficient , just say I dont know.

     {context}
     Question:{question}
    """,
    input_variables=["context","question"]

)

#page content concatenate
context_text = "/n/n".join(doc.page_content for doc in retriever.invoke("is nuclear function discuss in this video?"))

context_text

question = "is nuclear function discuss in this video?" # Define the question variable
final_prompt =  prompt.invoke({"context" : context_text , "question" : question})

final_prompt



"""##4.Generation"""

result = model.invoke(final_prompt)
print(result.content)



"""##5.Build chain"""

from langchain_core.runnables import RunnablePassthrough , RunnableLambda , RunnableParallel
from langchain_core.output_parsers import StrOutputParser

def format_doc(retrieve_docs):
  return "/n/n".join(doc.page_content for doc in retrieve_docs)

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_doc),
    "question":RunnablePassthrough()
})

parallel_chain.invoke("Who is Demis")

parser = StrOutputParser()

main_chain = parallel_chain | prompt | model| parser

main_chain.invoke("Who is Pakeeza")



