import streamlit as st
import os
import sys
import json
sys.path.append(os.path.abspath('../../'))
from tasks.task_3.task_3 import DocumentProcessor
from tasks.task_4.task_4 import EmbeddingClient
from tasks.task_5.task_5 import ChromaCollectionCreator

from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI

class QuizGenerator:
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        """
        Initializes the QuizGenerator with a required topic, the number of questions for the quiz,
        and an optional vectorstore for querying related information.

        :param topic: A string representing the required topic of the quiz.
        :param num_questions: An integer representing the number of questions to generate for the quiz, up to a maximum of 10.
        :param vectorstore: An optional vectorstore instance (e.g., ChromaDB) to be used for querying information related to the quiz topic.
        """
        if not topic:
            self.topic = "General Knowledge"
        else:
            self.topic = topic

        if num_questions > 10:
            raise ValueError("Number of questions cannot exceed 10.")
        self.num_questions = num_questions

        self.vectorstore = vectorstore
        self.llm = None
        self.question_bank = [] # Initialize the question bank to store questions
        self.system_template = """
            You are a subject matter expert on the topic: {topic}
            
            Follow the instructions to create a quiz question:
            1. Generate a question based on the topic provided and context as key "question"
            2. Provide 4 multiple choice answers to the question as a list of key-value pairs "choices"
            3. Provide the correct answer for the question from the list of answers as key "answer"
            4. Provide an explanation as to why the answer is correct as key "explanation"
            
            You must respond as a JSON object with the following structure **only**:
            
            {{
                "question": "<question>",
                "choices": [
                    {{"key": "A", "value": "<choice>"}},
                    {{"key": "B", "value": "<choice>"}},
                    {{"key": "C", "value": "<choice>"}},
                    {{"key": "D", "value": "<choice>"}}
                ],
                "answer": "<answer key from choices list>",
                "explanation": "<explanation as to why the answer is correct>"
            }}
            
            Do **not** include anything else other than the above JSON object.
            
            Context: {context}
            """
    
    def init_llm(self):
        """
        Initializes and configures the Large Language Model (LLM) for generating quiz questions.

        This method should handle any setup required to interact with the LLM, including authentication,
        setting up any necessary parameters, or selecting a specific model.

        :return: An instance or configuration for the LLM.
        """
        self.llm = VertexAI(
            model_name = "gemini-pro",
            temperature = 0.8, # Increased for less deterministic questions 
            max_output_tokens = 500
        )

    def generate_question_with_vectorstore(self):
        """
        Generates a quiz question based on the topic provided using a vectorstore

        :return: A JSON object representing the generated quiz question.
        """
        # if not self.llm:
        #     self.init_llm()
        # if not self.vectorstore:
        #     raise ValueError("Vectorstore not provided.")
        
        # from langchain_core.runnables import RunnablePassthrough, RunnableParallel

        # # Enable a Retriever
        # retriever = self.vectorstore.as_retriever()
        
        # # Use the system template to create a PromptTemplate
        # prompt = PromptTemplate.from_template(self.system_template)
        
        # # RunnableParallel allows Retriever to get relevant documents
        # # RunnablePassthrough allows chain.invoke to send self.topic to LLM
        # setup_and_retrieval = RunnableParallel(
        #     {"context": retriever, "topic": RunnablePassthrough()}
        # )
        # # Create a chain with the Retriever, PromptTemplate, and LLM
        # chain = setup_and_retrieval | prompt | self.llm 

        # # Invoke the chain with the topic as input
        # response = chain.invoke(self.topic)
        # return response
        
        # Raise an error if the vectorstore is not initialized on the class
        if not self.llm or not self.vectorstore:
            raise Exception("LLM or vectorstore is not initialized.")
        
        from langchain_core.runnables import RunnablePassthrough, RunnableParallel

        # Enable a Retriever using the as_retriever() method on the VectorStore object
        # Use the vectorstore as the retriever initialized on the class
        relevant_docs = self.vectorstore.query_chroma_collection(self.topic)

        # Determine how many documents are available
        available_docs_count = len(relevant_docs)

        # Set n_results based on the available documents
        n_results = min(available_docs_count, self.num_questions)
        
        # Check if relevant_docs is not empty
        if not relevant_docs:
            raise Exception("No relevant documents found for the given topic.")
        
        # Use the .from_template method on the PromptTemplate class and pass in the system template
        prompt = f"""
            You are a subject matter expert on the topic: {self.topic}.

            Generate a quiz question based on the topic provided and provide the result in **only** the following JSON format:

            {{
                "question": "<question>",
                "choices": [
                    {{"key": "A", "value": "<choice 1>"}},
                    {{"key": "B", "value": "<choice 2>"}},
                    {{"key": "C", "value": "<choice 3>"}},
                    {{"key": "D", "value": "<choice 4>"}}
                ],
                "answer": "<answer key from choices list>",
                "explanation": "<explanation as to why the answer is correct>"
            }}

            Do **not** include anything else other than the above JSON object.

            Context: {relevant_docs[:n_results]}
            """

        # Invoke the chain with the topic as input
        response = self.llm.invoke(prompt)

        return response

    def generate_quiz(self) -> list:
        """
        Task: Generate a list of unique quiz questions based on the specified topic and number of questions.

        This method orchestrates the quiz generation process by utilizing the `generate_question_with_vectorstore` method to generate each question and the `validate_question` method to ensure its uniqueness before adding it to the quiz.

        Steps:
            1. Initialize an empty list to store the unique quiz questions.
            2. Loop through the desired number of questions (`num_questions`), generating each question via `generate_question_with_vectorstore`.
            3. For each generated question, validate its uniqueness using `validate_question`.
            4. If the question is unique, add it to the quiz; if not, attempt to generate a new question (consider implementing a retry limit).
            5. Return the compiled list of unique quiz questions.

        Returns:
        - A list of dictionaries, where each dictionary represents a unique quiz question generated based on the topic.

        Note: This method relies on `generate_question_with_vectorstore` for question generation and `validate_question` for ensuring question uniqueness. Ensure `question_bank` is properly initialized and managed.
        """
        self.question_bank = [] # Reset the question bank
        retry_limit = 5  # Set a retry limit
        
        # Adjust the number of requested results
        # n_results = min(self.num_questions, len(self.question_bank))  # Ensure we don't exceed available questions
        print(self.num_questions)
        for i in range(self.num_questions):
            for attempt in range(retry_limit):
                if len(self.question_bank) >= self.num_questions:
                    break
                
                question_str = self.generate_question_with_vectorstore()  # Use class method to generate question
                if not question_str.strip():
                    print(f"Attempt {attempt + 1} failed to generate a question: Empty response.")
                    continue

                import re
                def extract_json_content(response):
                    """Extract JSON content from a string that may contain code block markers."""
                    # Remove code block markers (e.g., ```json ... ```)
                    return re.sub(r'```(?:json)?\n|```', '', response).strip()
                
                
                # Clean the response
                cleaned_response = extract_json_content(question_str)
                # print("Response Content:", repr(cleaned_response))
                
                try:
                    question = json.loads(cleaned_response)  # Convert the JSON String to a dictionary
                    # question_json = json.loads(question_str)
                    # print("Converted JSON:", question_json)  # Output the JSON object
                except json.JSONDecodeError as e:
                    print("\nFailed to decode question JSON.", e)
                    print(cleaned_response)
                    continue  # Skip this iteration if JSON decoding fails

                # Validate the question using the validate_question method
                if self.validate_question(question):
                    print("\nSuccessfully generated unique question ", i)
                    self.question_bank.append(question)  # Add the valid and unique question to the bank
                else:
                    print("\n\nDuplicate or invalid question detected.", question)

        return self.question_bank

    def validate_question(self, question: dict) -> bool:
        """
        Task: Validate a quiz question for uniqueness within the generated quiz.

        This method checks if the provided question (as a dictionary) is unique based on its text content compared to previously generated questions stored in `question_bank`. The goal is to ensure that no duplicate questions are added to the quiz.

        Steps:
            1. Extract the question text from the provided dictionary.
            2. Iterate over the existing questions in `question_bank` and compare their texts to the current question's text.
            3. If a duplicate is found, return False to indicate the question is not unique.
            4. If no duplicates are found, return True, indicating the question is unique and can be added to the quiz.

        Parameters:
        - question: A dictionary representing the generated quiz question, expected to contain at least a "question" key.

        Returns:
        - A boolean value: True if the question is unique, False otherwise.

        Note: This method assumes `question` is a valid dictionary and `question_bank` has been properly initialized.
        """
        if not self.question_bank:  # Check if question_bank is empty
            return True  # If empty, the question is unique by default
        
        question_text = question.get("question")  # Extract the question text from the dictionary
        
        # Consider missing 'question' key as invalid in the dict object
        # Check if a question with the same text already exists in the self.question_bank
        for existing_question in self.question_bank:  # Iterate over existing questions
            if existing_question.get("question") == question_text:  # Compare texts
                return False  # Return False if a duplicate is found
            
        return True  # Return True if no duplicates are found

# Test Generating the Quiz
if __name__ == "__main__":
    
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "",
        "location": "us-central1"
    }
    
    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")
        processor = DocumentProcessor()
        processor.ingest_documents()
    
        embed_client = EmbeddingClient(**embed_config) # Initialize from Task 4
    
        chroma_creator = ChromaCollectionCreator(processor, embed_client)
    
        question = None
        question_bank = None
    
        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
            
            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()
                
                st.write(topic_input)
                
                # Test the Quiz Generator
                generator = QuizGenerator(topic_input, questions, chroma_creator)
                generator.init_llm()
                question_bank = generator.generate_quiz()
                question = question_bank[0]

    if question_bank:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Question: ")
            for question in question_bank:
                st.write(question)