import os
import re
import smtplib
from email.mime.text import MIMEText
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.embeddings import OllamaEmbeddings
import chainlit as cl

load_dotenv()

prompt_template = """ 
    You are a legal assistant specializing in domestic violence cases. Your role is to provide concise and clear case summaries and also the arguments which will the junior lawers to win the case, generate legally sound arguments, and reference relevant articles and sections based on user prompts. Focus only on the legal context and respond accurately, avoiding overly complex legal jargon. If the user concludes the task or changes the topic, gracefully end the interaction. This tool is designed to assist junior lawyers, so ensure that all explanations are straightforward and educational.
    Context: {context} Question: {question}
    Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])  
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=db.as_retriever(), chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def load_llm():
    groqllm = ChatGroq(
        model="llama3-8b-8192", temperature=0
    )
    return groqllm

#def qa_bot():
    #data = PyPDFLoader('doc_samp1.pdf')
    #loader = data.load()
    #chunk = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    #splitdocs = chunk.split_documents(loader)
    #index_name = "lawai"
    #db = PineconeVectorStore.from_documents(splitdocs[:5], OllamaEmbeddings(model="mxbai-embed-large"), index_name=index_name)
    #llm = load_llm()
    #qa_prompt = set_custom_prompt()
    #qa = retrieval_qa_chain(llm, qa_prompt, db)
    #return qa

def qa_bot():
    data = PyPDFLoader('Dhillip_Combined_PDF_Data.pdf')
    loader = data.load()
    chunk = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)
    splitdocs = chunk.split_documents(loader)
    index_name = "llmlaw"
    db = PineconeVectorStore.from_documents(splitdocs, OllamaEmbeddings(model="mxbai-embed-large"), index_name=index_name)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

# def send_notification(email, message):
#     smtp_server = 'smtp.gmail.com'
#     smtp_port = 587
#     smtp_username = 'badrisrp3836@gmail.com'
#     smtp_password = 'hngb nzfa prsd adcy'

#     subject = 'Suicidal Attempt Detected'
#     body = f"Conversation related to suicidal attempts:\n\n{message}"

#     msg = MIMEText(body)
#     msg['Subject'] = subject
#     msg['From'] = 'therapybot@example.com'
#     msg['To'] = email

#     with smtplib.SMTP(smtp_server, smtp_port) as server:
#         server.starttls()
#         server.login(smtp_username, smtp_password)
#         server.sendmail(msg['From'], msg['To'], msg.as_string())


# @cl.on_chat_start
# async def start():
#     chain = qa_bot()
#     msg = cl.Message(content="Starting the bot...")
#     await msg.send()
#     msg.content = "Hi, Welcome to the Legal Assistant Bot. What is your query?"
#     await msg.update()

#     cl.user_session.set("chain", chain)

# suicidal_keywords = ['suicide', 'self-harm', 'end my life', 'suicidal thoughts', 'kill myself']

# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain")
#     if chain is None:
#         return

#     try:
#         for keyword in suicidal_keywords:
#             if keyword in message.content.lower():
#                 email = 'dhillipkumar2001@gmail.com'
#                 send_notification(email, message.content)
#                 break

#         res = await chain.acall({'query': message.content})
#         answer = res['result']
#         await cl.Message(content=answer).send()

#     except Exception as e:
#         await cl.Message(content=f"An error occurred: {e}").send()

# def send_notification(email, user_message, model_output):
#     smtp_server = 'smtp.gmail.com'
#     smtp_port = 587
#     smtp_username = 'badrisrp3836@gmail.com'
#     smtp_password = 'hngb nzfa prsd adcy'

#     subject = 'Potential Concerning Conversation Detected'
#     body = f"User message:\n{user_message}\n\nModel's response:\n{model_output}"

#     msg = MIMEText(body)
#     msg['Subject'] = subject
#     msg['From'] = 'therapybot@example.com'
#     msg['To'] = email

#     with smtplib.SMTP(smtp_server, smtp_port) as server:
#         server.starttls()
#         server.login(smtp_username, smtp_password)
#         server.sendmail(msg['From'], msg['To'], msg.as_string())

# @cl.on_chat_start
# async def start():
#     chain = qa_bot()
#     msg = cl.Message(content="Starting the bot...")
#     await msg.send()
#     msg.content = "Hi, Welcome to the Legal Assistant Bot. What is your query?"
#     await msg.update()

#     cl.user_session.set("chain", chain)

# suicidal_keywords = ['suicide', 'self-harm', 'end my life', 'suicidal thoughts', 'kill myself']

# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain")
#     if chain is None:
#         return

#     try:
#         res = await chain.acall({'query': message.content})
#         answer = res['result']

#         for keyword in suicidal_keywords:
#             if keyword in message.content.lower():
#                 email = 'dhillipkumar2001@gmail.com'
#                 send_notification(email, message.content, answer)
#                 break

#         await cl.Message(content=answer).send()

#     except Exception as e:
#         await cl.Message(content=f"An error occurred: {e}").send()

def extract_email(text):
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    return match.group(0) if match else None

def send_notification(email, user_message, model_output):
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'badrisrp3836@gmail.com'
    smtp_password = 'hngb nzfa prsd adcy'

    subject = 'Requested Information from Legal Assistant Bot'
    body = f"User message:\n{user_message}\n\nModel's response:\n{model_output}"

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'legalassistant@example.com'
    msg['To'] = email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(msg['From'], msg['To'], msg.as_string())

@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to the Legal Assistant Bot. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if chain is None:
        return

    try:
        res = await chain.acall({'query': message.content})
        answer = res['result']

        # Check if the user wants to send an email
        if "send" in message.content.lower() and "email" in message.content.lower():
            email = extract_email(message.content)
            if email:
                send_notification(email, message.content, answer)
                await cl.Message(content=f"I've sent the information to {email}. Here's my response:").send()
            else:
                await cl.Message(content="I couldn't find a valid email address in your message. Please provide a valid email if you want me to send the information.").send()

        await cl.Message(content=answer).send()

    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}").send()