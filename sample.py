import streamlit as st
import time
import functools, operator, requests, os, json
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.messages import BaseMessage, HumanMessage,AIMessage,FunctionMessage,SystemMessage
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END,MessageGraph
from langchain.tools import tool
import json
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict,Union
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
import os
from langchain_community.document_loaders import JSONLoader
from langchain.vectorstores import Chroma,Pinecone
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
import pinecone
# from langchain_pinecone import PineconeVectorStore
import time
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
#from typing_extensions import TypedDict
from typing import List,TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain.schema.runnable import RunnablePassthrough
import ast
from stories_json import product_json,product_json1,product_json2
from openai import OpenAI
from langchain_pinecone import PineconeVectorStore
import time
load_dotenv()

li=[]

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ['PINECONE_API_KEY'] = os.getenv("PINECONE_API_KEY")

#llm = ChatOpenAI(model="gpt-4o")
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
index_name = "langgraphserverless-index"
# #JsonLoade with the file name
# #loader = JSONLoader(file_path="product_data1.json")
# loader = TextLoader(file_path="output.txt",encoding='utf8')
# data = loader.load()

# #Split the text into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500,length_function=len,
#     is_separator_regex=False,)
# docs = text_splitter.split_documents(data)
# print(len(docs))
# #Use the OpenAIEmbeddings
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# # vectordb = Chroma.from_documents(documents=docs, 
# #                                     embedding=embeddings,
# #                                     persist_directory=persist_directory,
# #                                     collection_name="proinfoforlanggraph",
# #                                     )
# # # persiste the db to disk
# # vectordb.persist()
# vectorstore_from_docs = PineconeVectorStore.from_documents(
#     docs,
#     index_name=index_name,
#     embedding=embeddings
# )
# print("Embeddings have been generated")

@tool("refine_userquery_analyst", return_direct=False)
def refine_userquery_analyst(query: str):
    """when user query comes the first time,Check the query thoroughly and rewrite the query according
      to product description"""

    prompt = f'''
    refine a user query {query} and align it based on the product description listed below.

    product description: {product_json2}.
    
    
        '''
#find a semanticaly matching keyword from a query and replace it with a product descrption word and reproduce the query once again.
    client = OpenAI(api_key='sk-proj-Jwb75cA0ToYsqpV2DWU7T3BlbkFJ3UOkbFALzDcqKKirpxiS')
    # Send the prompt to the OpenAI API and get the response
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "system",
            "content": prompt
        
            },
            # {
            #   "role": "user",
            #   "content": "Write a SQL query which computes the average total order value for all orders on 2023-04-01."
            # }
        ],
        temperature=0.0,
        max_tokens=500,
        top_p=1
        )

    # Extract the generated text from the response
    generated_text = response.choices[0].message.content
    return generated_text



@tool("forecast_calculation")
def forecast_calculation(query: str) -> str:
    """Use this for forecasting the product data if days and SKU number is mentioned"""
    print("this is query from forecast calculation",query)
    return "from the forecast calculation"

@tool("fetch_product_review")
def fetch_product_review(query: str) -> str:
    """Use this if the user query is related to the product review, including positive review, critical review, negative review,competitior reviews."""
    print("this is content from fetch product review", query)
    #print("this is product name",productname)
    #llm = ChatOpenAI()
    llm = ChatOpenAI(model="gpt-4o")
    #llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    # vectorstore = PineconeVectorStore(index_name, embeddings)
    # retriever = vectorstore.as_retriever(search_type="similarity")
    vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding = embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity")
    #retriever = vectorstore_from_docs.as_retriever(search_type="similarity",search_kwargs={"k": 4})
    # compressor = LLMChainExtractor.from_llm(llm)
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=retriever
    # )
    
    prompt = '''You are an assistant for question-answering tasks.
      if the product comparison is asked in terms of either positive or negative review then first take review from the 
       particular SKY bestchoice product and compare it against same product's SKY competitiors product and put the comparisons
         reviews point by point.Do not consider other SKY products number other than the given SKY number.
       1. *Product Information Extraction:*
               - Summarize reviews into Positive, Neutral, and Negative aspects for both the queried product and its competitor(s).
        2. *User Query Understanding:*
            - Identify the main concern or query focus (e.g., product quality, price comparison, features).
            - Determine if the query requires comparison with a competitor or specific details about the product.
        3. *Expert Analysis and Response Generation:*
            - Compare the product and competitor based on the extracted information.
            - Highlight the queried product's strengths and unique features.
            - Address any negative aspects with potential solutions or explanations.
            - Provide a balanced view, acknowledging competitor strengths but emphasizing the queried product's value.
        4. *Response Customization:*
            - Tailor responses to directly address the user's specific concerns or queries.
            - Use persuasive language to emphasize benefits and mitigate concerns based on the summarized reviews.
            - Offer additional recommendations or alternatives if appropriate.
        5. *Continuous Learning:*
            - Update the response strategy based on user feedback and new product information.
            - Encourage users to ask more detailed questions for personalized advice.
        Question: {question} 

        Context: {context} 

        Answer:'''
    # prompt=''' You are an AI-assistant for the business operations team at the BestChoice Products.
    #   Answer the question based on your business acumen. 
    # Question: {question} 

    # Context: {context} 

    # Answer:        
    # '''
    # if the product comparison is asked in terms of either positive or negative
    #     review then take review from the {productname} bestchoice product and than take review of {productname} competitors and compare it
    #     and put the comparisons reviews point by point. 
    #     If comparison is asked:
    #     take {productname} review and compare it with {productname} competitors and put the comparison reviews point by point.
    #     If comparison is not asked:
    #     take {productname} review and put the either negative or positve reviews point by point whatever is asked.
    #     Try to answer the question in detail. If the question cannot be answered, answer with "I don't know".
    custom_rag_prompt = PromptTemplate.from_template(prompt)
    def format_docs(docs):
        #print("this is docs from format docs", docs)
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": lambda x: retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    respon=rag_chain.invoke(query)
    
    return respon

class AgentState(TypedDict):
    # The annotation tells the graph that new messages will always
    # be added to the current states
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # The 'next' field indicates where to route to next
    next: str

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

members = ["refine_userquery_analyst","forecast_calculation", "fetch_product_review"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers:  {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status."
     "When finished,respond with FINISH."
)

options = ["FINISH"] + members
# Using openai function calling can make output parsing easier for us
function_def = {
    "name": "route",
    "description": "Select the next role based on refine query agent.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        (
            "system",
            "Given the conversation above, who should act next?"
            " Or should we FINISH? Select one of: {options}",
        ),
    ]
).partial(options=str(options), members=", ".join(members))

supervisor_chain = (
    prompt
    | llm.bind_functions(functions=[function_def], function_call="route")
    | JsonOutputFunctionsParser()
)

forecast_agent = create_agent(llm, [forecast_calculation], "You are responsible for forecasting the product data.")
forecasting_node = functools.partial(agent_node, agent=forecast_agent, name="forecast_calculation")
# search_engine_agent = create_agent(llm, [web_search], "You are a web search engine.")
# search_engine_node = functools.partial(agent_node, agent=search_engine_agent, name="Search_Engine")
product_agent = create_agent(llm,[fetch_product_review],"You are responsible to give product review either negative or positive")
product_node = functools.partial(agent_node, agent=product_agent, name="fetch_product_review")
# twitter_operator_agent = create_agent(llm, [write_tweet], "You are responsible for writing a tweet based on the content given.")
# twitter_operator_node = functools.partial(agent_node, agent=twitter_operator_agent, name="Twitter_Writer")
refine_userquery_agent = create_agent(llm,[refine_userquery_analyst],"when user query comes first time,You are responsible to refine the user query based on the product information")
refine_userquery_node = functools.partial(agent_node, agent=refine_userquery_agent, name="refine_userquery_analyst")

workflow = StateGraph(AgentState)

workflow.add_node("refine_userquery_analyst", refine_userquery_node)
workflow.add_node("forecast_calculation", forecasting_node)
workflow.add_node("fetch_product_review", product_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

#workflow.set_entry_point("refine_userquery_analyst")
workflow.set_entry_point("supervisor")

graph = workflow.compile()

# # Function to simulate getting intermediate messages and the final answer
# def get_answer_with_intermediate_steps(question):

    
#     intermediate_steps = [
#         "Analyzing the question...",
#         "Gathering relevant information...",
#         "Processing the data...",
#         "Formulating the answer...",
#     ]
#     answer = "This is the final answer to your question."

#     return intermediate_steps, answer

# Streamlit App
st.title("Question and Answer Application")

# Input for question
query = st.text_input("Enter your question:")

if query:
    # Display the question
    st.write(f"**Question:** {query}")

    # Get the intermediate steps and the final answer
    #intermediate_steps, final_answer = get_answer_with_intermediate_steps(question)
    
    # Display intermediate messages with spinner
    st.write("**Processing steps:**")
    with st.spinner("Processing..."):
        #for step in intermediate_steps:
        def checkquery(query:str) -> str:

            prompt = f'''Your objective is to check a user query {query} and if it contains sky number pattern like SKY1935,SKY6333 or 
                    SKY 904 either in uppercase of lowercase then provide response as a "Yes".if not present in the query provide response in terms of "No".Dont give any extra words 

                    Example:
                    SKY6333
                    SKY1935
                    SKY 904
                    SKY 123
                    sky6333
                    sky1234
                    sky1935
                    '''
            client = OpenAI(api_key='sk-proj-Jwb75cA0ToYsqpV2DWU7T3BlbkFJ3UOkbFALzDcqKKirpxiS')
            # Send the prompt to the OpenAI API and get the response
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                {
                    "role": "system",
                    "content": prompt
                
                },
                # {
                #   "role": "user",
                #   "content": "Write a SQL query which computes the average total order value for all orders on 2023-04-01."
                # }
                ],
                temperature=0.0,
                max_tokens=500,
                top_p=1
            )

            # Extract the generated text from the response
            generated_text = response.choices[0].message.content
            print("this is generated text",generated_text)
            if generated_text=="No":
                return "Write a query again with SKU number"
            else:
                return generated_text
    
        answer=checkquery(query)
        #print(answer)
        count=0
        if answer=="Yes":
            config = {"recursion_limit": 20}
            for s in graph.stream(
                {
                    "messages": [
                        HumanMessage(content=query)
                    ]
                }, config=config
            ):
                if "__end__" not in s:
                    for key,value in s.items():
                        #st.write(key)
                        if "refine_userquery_analyst" in key:
                            count+=1
                            st.write(str(count)+"."+"**We are refining your query**")
                            st.write(value['messages'][0].content)
                        if "fetch_product_review" in key:
                            count+=1
                            st.write(str(count)+"."+"**We are fetching product review**")
                            st.write(value['messages'][0].content)
                    li.append(s)
                    #st.write(s)
                    #st.write("----")
                    #time.sleep(1)
            # final_response = graph.invoke(
            # {
            #     "messages": [
            #         HumanMessage(content=query)
            #     ]
            # }, config=config
            # )
            st.write("**Answer:**")
            dans=li[-2]
            #print(dans['fetch_product_review']['messages'][0].content)
            #st.write(dans['fetch_product_review']['messages'][0].content)
            st.write(dans)
    # Display the final answer
    # st.write("**Answer:**")
    # st.write(final_answer)
