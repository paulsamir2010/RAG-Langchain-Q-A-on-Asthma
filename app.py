import streamlit as st

# Must be the first Streamlit command
# st.set_page_config(page_title="RAG QA on Asthma", page_icon="🤖")
st.set_page_config(page_title="RAG QA on Asthma", page_icon="🫁")

# Import required libraries
import tiktoken
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from operator import itemgetter
import tempfile
import os
import locale
import yaml
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import (
    CrossEncoderReranker,
    LLMChainFilter,
)
from langchain.retrievers import ContextualCompressionRetriever


import logging
import sys

# Configure locale and download NLTK data
locale.getpreferredencoding = lambda: "UTF-8"
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chatbot_debug.log"),
        logging.StreamHandler(sys.stdout),  # This will show logs in Colab
    ],
)

logger = logging.getLogger(__name__)


# Log startup
logger.info("=== Application Starting ===")


# Load API credentials
with open("chatgpt_api_credentials.yml", "r") as file:
    api_creds = yaml.safe_load(file)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = api_creds["openai_key"]
OPENAI_KEY = api_creds["openai_key"]

# Initialize OpenAI embedding model
openai_embed_model = OpenAIEmbeddings(
    model="text-embedding-3-small", openai_api_key=OPENAI_KEY
)

st.title("Welcome to File QA RAG Chatbot 🤖")

# Load and process document
website = "https://my.clevelandclinic.org/health/diseases/6424-asthma"


def load_document(loader_class, website_url):
    loader = loader_class([website_url])
    return loader.load()


selenium_loader_doc = load_document(SeleniumURLLoader, website)

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1450, chunk_overlap=200)
chunked_docs = splitter.split_documents(selenium_loader_doc)

# Create and load vector DB
chroma_db = Chroma.from_documents(
    documents=chunked_docs,
    collection_name="rag_asthma_db",
    embedding=openai_embed_model,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="./asthma_db",
)

chroma_db = Chroma(
    persist_directory="./asthma_db",
    collection_name="rag_asthma_db",
    embedding_function=openai_embed_model,
)

# Initialize ChatGPT
chatgpt = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_KEY
)

# Set up retrieval pipeline
similarity_retriever = chroma_db.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)

_filter = LLMChainFilter.from_llm(llm=chatgpt)
compressor_retriever = ContextualCompressionRetriever(
    base_compressor=_filter, base_retriever=similarity_retriever
)

reranker = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")
reranker_compressor = CrossEncoderReranker(model=reranker, top_n=3)
final_retriever = ContextualCompressionRetriever(
    base_compressor=reranker_compressor, base_retriever=compressor_retriever
)

# Set up QA template
qa_template = """You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If no context is present or if you don't know the answer, just say that you don't know.
            Do not make up the answer unless it is there in the provided context.
            Give a detailed answer with regard to the question.

            Question: {question}
            Context: {context}
            Answer:"""

qa_prompt = ChatPromptTemplate.from_template(qa_template)


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


# Stream Handler for live updates
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Create QA RAG Chain
qa_rag_chain = (
    {
        "context": itemgetter("question") | final_retriever | format_docs,
        "question": itemgetter("question"),
    }
    | qa_prompt
    | chatgpt
)

# Initialize chat history
streamlit_msg_history = StreamlitChatMessageHistory(key="langchain_messages")
if len(streamlit_msg_history.messages) == 0:
    streamlit_msg_history.add_ai_message("Please ask your question?")

for msg in streamlit_msg_history.messages:
    st.chat_message(msg.type).write(msg.content)


# Post Message Handler for sources
class PostMessageHandler(BaseCallbackHandler):
    def __init__(self, msg: st.write):
        BaseCallbackHandler.__init__(self)
        self.msg = msg
        self.sources = []

    def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
        source_ids = []
        for d in documents:
            metadata = {
                "source": d.metadata["source"],
                "page": d.metadata["page"],
                "content": d.page_content[:200],
            }
            idx = (metadata["source"], metadata["page"])
            if idx not in source_ids:
                source_ids.append(idx)
                self.sources.append(metadata)

    def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
        if len(self.sources):
            st.markdown("__Sources:__ " + "\n")
            st.dataframe(data=pd.DataFrame(self.sources[:3]), width=1000)


# THIS ONE WORKED
# if user_prompt := st.chat_input():
#     logger.info(f"Received user prompt: {user_prompt}")
#     st.chat_message("human").write(user_prompt)

#     with st.chat_message("ai"):
#         try:
#             logger.info("Setting up stream handler...")
#             stream_handler = StreamHandler(st.empty())
#             sources_container = st.write("")
#             pm_handler = PostMessageHandler(sources_container)
#             config = {"callbacks": [stream_handler, pm_handler]}

#             logger.info("Starting response generation...")
#             with st.spinner("Thinking..."):
#                 try:
#                     # Test the retriever
#                     logger.info("Testing retriever...")
#                     context = final_retriever.invoke(user_prompt)
#                     if not context:
#                         raise ValueError("No context retrieved from documents")
#                     logger.info(f"Retrieved {len(context)} context documents")

#                     # Format the context
#                     formatted_context = format_docs(context)
#                     logger.info("Context formatted successfully")

#                     # Prepare the complete prompt
#                     full_prompt = {
#                         "question": user_prompt,
#                         "context": formatted_context,
#                     }
#                     logger.info("Prompt prepared")

#                     # Send to ChatGPT
#                     logger.info("Sending to ChatGPT...")
#                     response = chatgpt.invoke(
#                         qa_prompt.format_prompt(**full_prompt).to_string()
#                     )
#                     logger.info("Received response from ChatGPT")

#                     # Display the response
#                     st.write(response)
#                     logger.info("Response displayed")

#                 except Exception as e:
#                     logger.error(
#                         f"Error in response generation: {str(e)}", exc_info=True
#                     )
#                     st.error(f"Error: {str(e)}")
#                     # Display more detailed error information
#                     st.error("Detailed error information has been logged")
#                     raise

#         except Exception as e:
#             logger.error(f"Error in chat interface: {str(e)}", exc_info=True)
#             st.error("An error occurred. Please check the logs for details.")

# Add this at the bottom to display any errors that occur during startup
# if st.session_state.get("startup_error"):
#     st.error(f"Startup Error: {st.session_state.startup_error}")


def test_components():
    try:
        # Test retriever
        test_docs = final_retriever.invoke("What is asthma?")
        logger.info(f"Retriever test: Got {len(test_docs)} documents")

        # Test ChatGPT
        test_response = chatgpt.invoke("Hello, how are you?")
        logger.info(f"ChatGPT test: {test_response}")

        logger.info("All components tested successfully")
    except Exception as e:
        logger.error(f"Component test failed: {str(e)}")


# LATEST SUGGESTION
if user_prompt := st.chat_input():
    logger.info(f"Received user prompt: {user_prompt}")
    st.chat_message("human").write(user_prompt)

    with st.chat_message("ai"):
        try:
            logger.info("Setting up stream handler...")
            stream_handler = StreamHandler(st.empty())
            sources_container = st.write("")
            pm_handler = PostMessageHandler(sources_container)
            config = {"callbacks": [stream_handler, pm_handler]}

            logger.info("Starting response generation...")
            with st.spinner("Thinking..."):
                try:
                    # Get context using retriever
                    logger.info("Retrieving context...")
                    context = final_retriever.invoke(user_prompt)
                    if not context:
                        raise ValueError("No context retrieved from documents")
                    logger.info(f"Retrieved {len(context)} context documents")

                    # Format the context
                    formatted_context = format_docs(context)
                    logger.info("Context formatted successfully")

                    # Prepare the complete prompt
                    full_prompt = {
                        "question": user_prompt,
                        "context": formatted_context,
                    }
                    logger.info("Prompt prepared")

                    # Send to ChatGPT
                    logger.info("Sending to ChatGPT...")
                    response = chatgpt.invoke(
                        qa_prompt.format_prompt(**full_prompt).to_string()
                    )
                    logger.info("Received response from ChatGPT")

                    # Display the response
                    st.write(response.content)  # Add this line to show the response

                    # Display sources
                    if context:
                        st.markdown("__Sources:__ ")
                        sources_df = pd.DataFrame(
                            [
                                {
                                    "source": doc.metadata["source"],
                                    "page": doc.metadata.get("page", "N/A"),
                                    "content": doc.page_content[:200] + "...",
                                }
                                for doc in context[:3]  # Show top 3 sources
                            ]
                        )
                        st.dataframe(sources_df, width=1000)

                except Exception as e:
                    logger.error(
                        f"Error in response generation: {str(e)}", exc_info=True
                    )
                    st.error(f"Error: {str(e)}")
                    st.error("Detailed error information has been logged")
                    raise

        except Exception as e:
            logger.error(f"Error in chat interface: {str(e)}", exc_info=True)
            st.error("An error occurred. Please check the logs for details.")


# Add this at the bottom to display any errors that occur during startup
if st.session_state.get("startup_error"):
    st.error(f"Startup Error: {st.session_state.startup_error}")

# COMBINEd new suggestion

# In your chat interface section:
# if user_prompt := st.chat_input():
#     logger.info(f"Received user prompt: {user_prompt}")
#     st.chat_message("human").write(user_prompt)

#     with st.chat_message("ai"):
#         try:
#             logger.info("Setting up stream handler...")
#             stream_handler = StreamHandler(st.empty())
#             sources_container = st.write("")
#             pm_handler = PostMessageHandler(sources_container)
#             config = {"callbacks": [stream_handler, pm_handler]}

#             logger.info("Starting response generation...")
#             with st.spinner("Thinking..."):
#                 try:
#                     # Use the RAG chain for retrieval and response
#                     logger.info("Invoking RAG chain...")
#                     response = qa_rag_chain.invoke({"question": user_prompt}, config)
#                     logger.info("RAG chain completed successfully")

#                     if not response:
#                         raise ValueError("No response generated from RAG chain")

#                     # The sources will be displayed by the PostMessageHandler
#                     logger.info("Response and sources displayed")

#                 except Exception as e:
#                     logger.error(f"Error in RAG chain: {str(e)}", exc_info=True)
#                     st.error(f"Error: {str(e)}")
#                     st.error("Please try asking your question again.")
#                     raise

#         except Exception as e:
#             logger.error(f"Error in chat interface: {str(e)}", exc_info=True)
#             st.error("An error occurred. Please check the logs for details.")

# # Add this at the bottom to display any errors that occur during startup
# if st.session_state.get("startup_error"):
#     st.error(f"Startup Error: {st.session_state.startup_error}")


# if user_prompt := st.chat_input():
#     logger.info(f"Received user prompt: {user_prompt}")
#     st.chat_message("human").write(user_prompt)

#     with st.chat_message("ai"):
#         try:
#             logger.info("Setting up stream handler...")
#             stream_handler = StreamHandler(st.empty())
#             sources_container = st.write("")
#             pm_handler = PostMessageHandler(sources_container)
#             config = {"callbacks": [stream_handler, pm_handler]}

#             logger.info("Starting response generation...")
#             with st.spinner("Thinking..."):
#                 try:
#                     # Log retriever operation
#                     logger.info("Retrieving context from vector store...")
#                     context = final_retriever.invoke(user_prompt)
#                     logger.info(f"Retrieved {len(context)} context documents")

#                     # Log ChatGPT operation
#                     logger.info("Sending request to ChatGPT...")
#                     response = qa_rag_chain.invoke({"question": user_prompt}, config)
#                     logger.info(f"Received response from ChatGPT: {response}")

#                     logging.debug("Processing prompt...")

#                 except Exception as e:
#                     logger.error(
#                         f"Error during response generation: {str(e)}", exc_info=True
#                     )
#                     st.error(f"Error generating response: {str(e)}")
#                     raise e

#             if not response:
#                 logger.error("No response generated")
#                 st.error(
#                     "I apologize, but I couldn't generate a response. Please try again."
#                 )

#         except Exception as e:
#             logger.error(f"Error in chat interface: {str(e)}", exc_info=True)
#             st.error(f"An error occurred: {str(e)}")
#             st.error("Please try asking your question again.")
