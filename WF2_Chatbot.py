# Import pysqlite3 and swap it with the default sqlite3 library
import sys
__import__('pysqlite3')  # Corrected this line
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import streamlit as st
import openai
import numpy as np
from threading import Lock
import json
import os
import time
# Load environment variables and set OpenAI key
openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"]

# Initialize ChromaDB
CHROMA_DATA_PATH = 'chromadb_WF2_chatbot/'
COLLECTION_NAME = "document_embeddings"
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key, 
    model_name="text-embedding-ada-002",
    dimensions=1536
)
collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=openai_ef)

# Set up Streamlit app
st.set_page_config(page_title="Quiz Chatbot", page_icon=":books:", layout="wide")

# Define questions and right answers
questions = [
    {
        "scenario": "Scenario 1",
        "question": "You are planning to fast for Ramadan, and your community has announced the sighting of the new crescent moon. However, your friend in another city claims they haven't seen the moon yet. How would you respond to your friend based on what you've learned about the different moon sightings in different places?",
        "right_answer": "Explain how moon sightings may differ between locations due to local visibility and different time zones."
    },
    {
        "scenario": "Scenario 2",
        "question": "It's the start of a new Islamic month, and you want to perform the recommended actions. You have a busy school schedule but still want to follow the sunnah. What are two simple actions you could do at the beginning of the new month, and why are they beneficial?",
        "right_answer": "Mention charity and reciting a dua or prayer as recommended actions."
    },
    {
        "scenario": "Scenario 3",
        "question": "Your classmate asks why Muslims use a lunar calendar instead of the solar calendar, especially since solar dates stay the same every year. How would you explain the benefit of using a lunar calendar in Islam, particularly for occasions like Ramadan or Hajj?",
        "right_answer": "Explain how the lunar calendar allows occasions like Ramadan to rotate through different seasons."
    },
    {
        "scenario": "Scenario 4",
        "question": "Imagine it's a clear night, and you see the new crescent moon for the first time. You remember learning about the phases of the moon and the significance in Islam. What would be a good way to reflect on this moment, and which dua could you recite? Why is this a meaningful time?",
        "right_answer": "Mention reflection on Allah's creation and reciting a specific dua for the new crescent moon."
    },
    {
        "scenario": "Scenario 5",
        "question": "You’re participating in a group discussion at the mosque about the differences in moonsighting rulings among scholars. One person mentions that if a city to the east sees the moon, another city to the west should follow. Can you explain Ayatullah Sistani’s opinion on the unity of horizons and when you can follow the moon sighting of another city?",
        "right_answer": "Explain Ayatullah Sistani's view on the unity of horizons and its implications for moon sightings."
    }
]

# Define system prompt
system_prompt = """
    Role: As a proficient educational assistant dedicated to supporting learners, your primary responsibilities include providing targeted feedback based solely on the information from Module 6F, Lesson 06 on moonsighting. Your goal is to assess understanding and guide the student effectively through structured feedback and hints without deviating from the source material.

    Tasks:
    1. Critical Analysis and Feedback:
        - Assess each student's response individually based on the concepts covered in Module 6F, Lesson 06, to evaluate their understanding.
        - Provide concise, targeted feedback to confirm, correct, or enhance understanding, strictly following the information from Module 6F, Lesson 06.
        - Ensure feedback directly reflects the terminology and explanations from Module 6F, Lesson 06, avoiding any additional general knowledge or interpretations not found in the lesson.
        - Use simple, clear language to maintain a supportive and educational tone.
    
    Handling Inquiries:
    1. For critiquing responses:
        - Offer direct feedback using only the information from Module 6F, Lesson 06. Avoid summarizing assessments or introducing unrelated information.
        - Provide concise additional explanations to enhance clarity or address missing details, referring strictly to Module 6F, Lesson 06.
        - Correct inaccuracies and guide students back to relevant concepts from Module 6F, Lesson 06 when responses are off-topic or incorrect.
        - Employ guided questions and additional information from Module 6F, Lesson 06 as necessary for follow-up queries or corrections.
    
    Response Guidelines:
    1. Ensure all feedback is accurate and exclusively supported by Module 6F, Lesson 06.
    2. Provide corrective guidance and additional information if responses misinterpret a concept, using only Module 6F, Lesson 06.
    3. Use concise questions and dialogue to encourage critical thinking, strictly adhering to Module 6F, Lesson 06.
    4. Maintain a supportive and educational tone, using simple language and practical examples from Module 6F, Lesson 06.
    5. Aim for engagement through direct and educational feedback, adhering strictly to Module 6F, Lesson 06 without summarizing or providing extraneous details.
    6. Avoid explicitly mentioning the source of information; act as if Module 6F, Lesson 06 is the inherent source of truth.
    7. In instances where the student provides an incomprehensible answer, avoid interpreting the answer—respond solely based on known concepts in Module 6F, Lesson 06.
    8. When the name "Muhammad" is mentioned, add "(saww)" immediately after it.

    Question Structure and Scenarios:
    - The assessment consists of 5 scenario-based questions that should be asked in a specific, predefined order:
    {questions as outlined above}

    Attempts and Feedback:
    - The student has 3 attempts to answer each question correctly.
    - For each incorrect or partially correct answer, provide subtle guidance, nudges, or prompts to encourage deeper thinking without directly revealing the answer.
    - Be strict in evaluating responses. Only fully correct answers are accepted. If the response is partial or lacks key elements, prompt the student for further clarification before moving on.
    - After 3 unsuccessful attempts, provide the correct answer along with directions to review specific content from Module 6F, Lesson 06 for further understanding.

    Correct Answers:
    - When the student provides a correct answer, acknowledge it positively, add relevant insights if needed, and then proceed to the next question.
    
    Overall Goal:
    - Your objective is to assess the student’s understanding of Module 6F, Lesson 06 while guiding them to recall, connect, and articulate their knowledge effectively.
"""

# Function to get embeddings for a given text
def get_embedding(text):
    response = openai.Embedding.create(model="text-embedding-ada-002", input=text)
    return np.array(response['data'][0]['embedding'])

# Function to query ChromaDB for the most relevant context based on the question
def retrieve_context(query):
    # Query ChromaDB with the question text
    results = collection.query(query_texts=[query], n_results=3)
    
    # # Log the results structure to understand any issues
    # st.write("ChromaDB Results:", results)

    # Initialize an empty context text
    context_text = ""

    # Check if the results contain the "metadatas" key as expected
    if "metadatas" in results:
        # Extract "text" from each metadata entry if available
        context_text = " ".join(
            metadata.get("text", "") for metadata in results["metadatas"] if isinstance(metadata, dict)
        )
    else:
        context_text = "No relevant context found."

    return context_text

# Function to generate feedback based on the student's answer and relevant context
def generate_feedback(question_data, user_answer, attempt_number):
    # Retrieve context from ChromaDB
    context_text = retrieve_context(question_data["question"])

    # Refine prompt to emphasize structured feedback with subtle guidance and hints
    prompt = f"""
    You are assessing a student's understanding of Module 6F, Lesson 06 on moonsighting. Here’s the scenario and question:
    
    Scenario: {question_data["scenario"]}
    Question: {question_data["question"]}
    Right Answer: {question_data["right_answer"]}
    
    Lesson Context (from Module 6F, Lesson 06): {context_text}
    
    Student's Answer: {user_answer}
    Attempt Number: {attempt_number}

    Task:
    1. Analyze the student's response carefully, referencing only the information in Module 6F, Lesson 06.
    2. If the response is correct, acknowledge it positively and proceed.
    3. If the response is partially correct, provide subtle hints to encourage deeper thinking.
    4. If the response is incorrect, provide a gentle nudge or guiding question without directly revealing the answer.
    5. After 3 unsuccessful attempts, reveal the correct answer and suggest reviewing specific concepts from Module 6F, Lesson 06.

    Your response should be educational, supportive, and strictly adhere to the lesson content. Avoid any unrelated information or outside knowledge.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message["content"]

# Main function to display the assessment instructions
def display_instructions():
    st.title("Educational Assistant Chatbot🤖")
    st.write("Hey! I will help you assess and guide on Module 6F, Lesson 06 about Moonsighting.")
    
    st.subheader("Assessment Instructions:")
    st.write("""
    - **Question Structure**: The assessment has 5 scenario-based questions, asked in a specific order.
    - **Attempts and Feedback**: You have 3 attempts to answer each question correctly.
    - **Strictness**: Only fully correct answers are accepted; partial answers will prompt for further clarification.
    - **Final Explanation**: After 3 unsuccessful attempts, I will provide the correct answer and suggest reviewing the lesson.
    - **Correct Answers**: Correct answers will be acknowledged, and we will move on to the next question.
    """)

    # Button to start the assessment
    if st.button("Start Assessment"):
        st.session_state["page"] = "assessment"  # Switch to the assessment page

# Function to simulate typing effect
def simulate_typing(text, delay=0.01):
    container = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        container.markdown(displayed_text)
        time.sleep(delay)

# Main function to display the quiz
def display_quiz():
    # Fetch current question data
    current_question = questions[st.session_state["current_question_index"]]

    # Display question
    st.write(f"### {current_question['scenario']}")
    st.write(current_question["question"])

    # Display chat history if available
    for entry in st.session_state["chat_history"]:
        role = entry["role"]
        content = entry["content"]
        st.chat_message(role).write(content)

    # Display trial count above the input field
    st.write(f"**Trial: {st.session_state['attempts'] + 1} of 3**")

    # Handle single submission using st.chat_input with static label
    if user_input := st.chat_input("Type your answer here"):
        # Increment session state attempts immediately
        st.session_state["attempts"] += 1

        # Append user input to chat history
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)  # Display the user's input

        # Display a spinner while processing the answer
        with st.spinner('💭Checking your answer...'):
            time.sleep(0.5)  # Simulate delay for demonstration
            feedback = generate_feedback(current_question, user_input, st.session_state["attempts"])

        # Append feedback to chat history
        st.session_state["chat_history"].append({"role": "assistant", "content": feedback})
        with st.chat_message("assistant"):
            simulate_typing(feedback)  # Use typing simulation for assistant response

        # Process attempts and correct answers
        if "correct" in feedback.lower():
            st.session_state["show_proceed_button"] = True
        elif st.session_state["attempts"] >= 3:
            # After 3 attempts, reveal the correct answer and suggest reviewing the lesson
            correct_answer_feedback = f"The correct answer is: {current_question['right_answer']}. Please review Module 6F, Lesson 06."
            st.session_state["chat_history"].append({"role": "assistant", "content": correct_answer_feedback})
            with st.chat_message("assistant"):
                simulate_typing(correct_answer_feedback)
            st.session_state["show_proceed_button"] = True
        else:
            st.session_state["show_proceed_button"] = False

    # Display "Proceed to the next question" button if answer is correct or 3 attempts reached
    if st.session_state.get("show_proceed_button", False) and st.button("Proceed to the next question"):
        st.session_state["current_question_index"] += 1
        st.session_state["attempts"] = 0
        st.session_state["show_proceed_button"] = False  # Hide button after moving to the next question
        st.session_state["chat_history"] = []  # Clear chat history for the next question

    # Always display the Restart Quiz button at the bottom
    if st.button("Restart Quiz"):
        st.session_state["page"] = "instructions"
        st.session_state["current_question_index"] = 0
        st.session_state["attempts"] = 0
        st.session_state["chat_history"] = []
        st.session_state["show_proceed_button"] = False

# Run the app with a page-based structure
def main():
    # Initialize session state variables
    if "page" not in st.session_state:
        st.session_state["page"] = "instructions"
    if "current_question_index" not in st.session_state:
        st.session_state["current_question_index"] = 0
    if "attempts" not in st.session_state:
        st.session_state["attempts"] = 0
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "show_proceed_button" not in st.session_state:
        st.session_state["show_proceed_button"] = False

    # Display either the instructions or the assessment quiz
    if st.session_state["page"] == "instructions":
        display_instructions()
    else:
        display_quiz()

# Run the app
if __name__ == "__main__":
    main()
