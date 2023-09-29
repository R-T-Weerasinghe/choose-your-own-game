from langchain.memory import CassandraChatMessageHistory, ConversationBufferMemory
from langchain.llms import OpenAI
from langchain import LLMChain, PromptTemplate

from connect_database import session, ASTRA_DB_KEYSPACE, OPENAI_API_KEY
from template import template
message_history = CassandraChatMessageHistory(
    session_id="mysession",
    session=session,
    keyspace=ASTRA_DB_KEYSPACE,
    ttl_seconds=3600
)

message_history.clear()

cass_buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=message_history
)

prompt = PromptTemplate(
    input_variables=["chat_history", "human_history"],
    template=template
)

llm = OpenAI(openai_api_key=OPENAI_API_KEY)
llm_chain = LLMChain(
    llm=llm,
    prompt="prompt",
    memory=cass_buffer_memory
)
