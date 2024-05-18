from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

MAX_TOKENS = 1024
MAX_CHAT_HISTORY = 10
VERBOSE = False

embeddings = OpenAIEmbeddings()

faiss = FAISS.load_local("faiss", embeddings, allow_dangerous_deserialization=True)

llm = ChatOpenAI(max_tokens=MAX_TOKENS)

chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=faiss.as_retriever(), verbose=VERBOSE
)

chat_history = list()

while True:
    question = input("You: ")
    resp = chain.invoke({"question": question, "chat_history": chat_history})
    answer = resp["answer"]
    print("AI: ", answer)
    chat_history.append((question, answer))
    if len(chat_history) > MAX_CHAT_HISTORY:
        chat_history.pop(0)
