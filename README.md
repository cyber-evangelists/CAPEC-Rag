# RAG System Using LLaMA 3.2 with knowledge base of Threatmon-feeds-IOC

## Introduction

This project implements a Retrieval-Augmented Generation (RAG) based system where users ask question related to the CAPEC dataset and it will response to the query. This project is completed using Web Sockets Fast API.

## Setting Up

Follow these steps to set up and run the project:

1. Clone the repository:

   ```
   git clone https://github.com/cyber-evangelists/CAPEC-Rag.git
   ```

2. Navigate to the project root directory:

   ```
   CAPEC-Rag
   ```

3. Make sure that the docker is installed on your system:

   ```
   docker --version
   ```

   If docker is not installed, run the following command:

   ```
   sudo apt install docker
   ```

4. In the same directory, create a file name `.env` and add following API key

   ```
   GROQ_API_KEY=your_api_key
   ```

   replace your_api_key with groq API key

5. Build the docker environment::

   ```
   docker compose up --build
   ```

6. Access the graio app by pasting this URL:

   ```
   http://localhost:7860/
   ```

7. Enter text to search for the query
