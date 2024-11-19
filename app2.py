import os
import re
import smtplib
from email.mime.text import MIMEText
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.embeddings import OllamaEmbeddings
import chainlit as cl

load_dotenv()

prompt_template = """ 
    You are a legal assistant specializing in domestic violence cases. Your role is to provide the arguments for the requested scenario and concise and clear previous related case summaries, which will help the junior lawyers to win the case, generate legally sound arguments, and reference relevant articles and sections based on user prompts. Focus only on the legal context and respond accurately, avoiding overly complex legal jargon. If the user concludes the task or changes the topic, gracefully end the interaction. This tool is designed to assist junior lawyers, so ensure that all explanations are straightforward and educational.
    Context: {context} Question: {question}
    Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])  
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=db.as_retriever(search_kwargs={"k": 5}), chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def load_llm():
    return ChatGroq(model="llama3-8b-8192", temperature=0.5)

def qa_bot():
    index_name = "lawllm2"
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    
    # Load the existing Pinecone index
    db = PineconeVectorStore.from_existing_index(index_name, embeddings)
    
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

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
    body = f"User query:\n{user_message}\n\nModel's response:\n{model_output}"

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
    cl.user_session.set("last_response", None)  # Initialize last_response
    cl.user_session.set("last_query", None)  # Initialize last_query

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    if chain is None:
        return

    try:
        # Check if the user wants to send an email
        if "send" in message.content.lower() and "email" in message.content.lower():
            email = extract_email(message.content)
            last_response = cl.user_session.get("last_response")
            last_query = cl.user_session.get("last_query")
            
            if email and last_response and last_query:
                send_notification(email, last_query, last_response)
                await cl.Message(content=f"I've sent the previous response to {email}.").send()
            elif not email:
                await cl.Message(content="I couldn't find a valid email address in your message. Please provide a valid email if you want me to send the information.").send()
            else:
                await cl.Message(content="I don't have any previous response or query to send. Please ask a question first.").send()
        else:
            # Normal query processing
            res = await chain.acall({'query': message.content})
            answer = res['result']
            cl.user_session.set("last_response", answer)  # Store the response
            cl.user_session.set("last_query", message.content)  # Store the user's query
            await cl.Message(content=answer).send()

    except Exception as e:
        await cl.Message(content=f"An error occurred: {e}").send()

if __name__ == "__main__":
    cl.run(main)
