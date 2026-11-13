from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os 

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

#Indexing 
#Step 1 getting the transcript of the Vdo !

try:
    video_id = 'JinpVA6p8Mo' # this is the id of yt vdo
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])

    #these transcripts are in the list of dictioneries that's why in the next code we 
    # are joining those 

    transcript = " ".join(chunk.text for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print('No captions avalable for this video')


# Chunking (Text Splitter )

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
chunks = splitter.create_documents([transcript])

print(len(chunks))

# Embedding and vectore storing using FIASS(by meta

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)
vector_store = FAISS.from_documents(chunks,embeddings)

# this is the code to see how the chunks embedded id looks in the vector data base
# and it's id(done using gpt model's)

vector_store.index_to_docstore_id

for vector_id, doc_id in vector_store.index_to_docstore_id.items():
    print(f"Vector ID: {vector_id} | Document ID: {doc_id}")

# Indexing part is completed 

#Step 2 Retriver

retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs ={"k":4}) #k means number of documents  and similarty is thye pf search 

#step 2 completed 

#step 3 augmentation 

llm = ChatGroq(model="openai/gpt-oss-20b",temperature= 0.2)

prompt = PromptTemplate(
    template= """
    You are a helpful assistent.
    Answer ONLY from the provided transcript context.
    if the context is insufficient, just say you don't know.

    {context}
    question : {question}
    """,
    input_variables={'context','question'}
)

question = 'Are we making a iceball in this vdo if not then what are we making'
retrieved_docs = retriever.invoke(question)

context_text = "\n \n".join(doc.page_content for doc in retrieved_docs)

#sending eveything thing to the main prompt and eveything will go to the llm next

final_promt = prompt.invoke({"context" : context_text, "question":question})

#Step 3 is completed Augmentation

#step 4 Generation 

answer = llm.invoke(final_promt)
print(answer.content)

#RAG process is completed now we are building chains 

from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

#making first parllel chain 

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
parser = StrOutputParser()

# connecting 2 chins 

main_chain = parallel_chain | prompt | llm | parser
main_chain.invoke("summerize this whole vdo ?  ")