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
from sklearn.metrics.pairwise import cosine_similarity

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
st.set_page_config(page_title="Quiz Chatbot", page_icon=":books:")

# Define questions and right answers
questions = [
    {
        "scenario_number": "Scenario 1",
        "scenario": "You are planning to fast for Ramadan, and your community has announced the sighting of the new crescent moon. However, your friend in another city claims they haven't seen the moon yet.",
        "question": "How would you respond to your friend based on what you've learned about the different moon sightings in different places?",
        "key_points": [
            "Moon sighting times vary across different locations",
            "Local conditions and visibility can affect moon sightings",
            "Time zones influence sighting reports in different areas"
        ],
        "example_response": "The new crescent moon may not be visible everywhere on the same evening. Due to local conditions and time zones, one city may see the moon while another city may not.",
        "context": "The Islamic lunar calendar depends on the phases of the moon, and the new month starts with the sighting of the crescent moon. Since cities to the west may see the moon after eastern locations, the sighting can vary regionally."
    },
    {
        "scenario_number": "Scenario 2",
        "scenario": "It's the start of a new Islamic month, and you want to perform the recommended actions. You have a busy school schedule but still want to follow the sunnah.",
        "question": "What are two simple actions you could do at the beginning of the new month, and why are they beneficial?",
        "key_points": [
            "Giving charity as a recommended act",
            "Reciting a dua or prayer",
            "Explanation of benefits such as blessings, protection from misdeeds, and spiritual growth"
        ],
        "example_response": "I could give charity and recite a dua. Giving charity helps protect against harm, while the dua brings blessings for the new month.",
        "context": "Recommended actions include charity, prayer, and reflection on personal growth as the new month is a time for renewing commitment and dedication to Allah."
    },
    {
        "scenario_number": "Scenario 3",
        "scenario": "Your classmate asks why Muslims use a lunar calendar instead of the solar calendar, especially since solar dates stay the same every year.",
        "question": "How would you explain the benefit of using a lunar calendar in Islam, particularly for occasions like Ramadan or Hajj?",
        "key_points": [
            "The lunar calendar causes Islamic events to move through different seasons",
            "Using the lunar calendar allows for varied seasonal experiences",
            "The lunar months begin with the sighting of the crescent moon"
        ],
        "example_response": "The lunar calendar lets Islamic events like Ramadan and Hajj rotate through all seasons. This way, Muslims can experience these occasions in different weather conditions and day lengths over time.",
        "context": "The lunar calendar is about 11 days shorter than the solar calendar, causing Islamic events to shift through different seasons, allowing diverse experiences for Muslims."
    },
    {
        "scenario_number": "Scenario 4",
        "scenario": "Imagine it's a clear night, and you see the new crescent moon for the first time. You remember learning about the phases of the moon and the significance in Islam.",
        "question": "What would be a good way to reflect on this moment, and which dua could you recite? Why is this a meaningful time?",
        "key_points": [
            "Expressing gratitude and making a dua upon seeing the new crescent moon",
            "Reciting the dua narrated by Imam Zayn al-Abidin (a) for moonsighting",
            "Reflecting on the passage of time and personal growth as Muslims"
        ],
        "example_response": "I would make a dua, reflecting on the start of the new month as a chance to improve myself and draw closer to Allah. The dua of Imam Zayn al-Abidin (a) emphasizes gratitude and repentance.",
        "context": "Seeing the crescent moon symbolizes the start of a new month. It’s a moment to reflect on time, and we can recite a dua, such as the one from Imam Zayn al-Abidin, to seek blessings and guidance."
    },
    {
        "scenario_number": "Scenario 5",
        "scenario": "You’re participating in a group discussion at the mosque about the differences in moonsighting rulings among scholars. One person mentions that if a city to the east sees the moon, another city to the west should follow.",
        "question": "Can you explain Ayatullah Sistani’s opinion on the unity of horizons and when you can follow the moon sighting of another city?",
        "key_points": [
            "Ayatullah Sistani’s ruling on unity of horizons",
            "Conditions under which one city can follow another's moon sighting",
            "Cities with similar horizons can follow each other's moon sightings"
        ],
        "example_response": "According to Ayatullah Sistani, cities with similar horizons can follow each other in moon sighting. If one city to the east sees the moon, a city to the west may also observe the new month.",
        "context": "Ayatullah Sistani supports the unity of horizons, allowing cities with similar geographic horizons to follow each other's sightings, which considers Earth’s spherical nature and rotation."
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
from sklearn.metrics.pairwise import cosine_similarity

def generate_feedback(question_data, user_answer, attempt_number):
    context_text = retrieve_context(question_data["question"])

    # Define hint level based on the attempt number
    if attempt_number == 1:
        hint_level = "Provide high-level hints to encourage exploration without direct answers."
    elif attempt_number == 2:
        hint_level = "Provide specific hints or reference missing key points indirectly."
    elif attempt_number == 3:
        hint_level = "Provide a full answer and suggest reviewing specific content from Module 6F, Lesson 06."

    # Construct prompt for GPT-4o to handle all feedback
    prompt = f"""
    You are assessing a student's understanding of Module 6F, Lesson 06 on moonsighting. Here’s the scenario and question:
    
    Scenario: {question_data["scenario"]}
    Question: {question_data["question"]}
    Key Points: {', '.join(question_data['key_points'])}
    Lesson Context: {context_text}
    
    Student's Answer: {user_answer}
    Attempt Number: {attempt_number}

    Task:
    1. Determine if the student's answer is fully correct by checking if it covers all key points.
    2. If the answer covers all key points, ALWAYS state "This answer is fully correct."
    3. If the answer is partially correct (some key points are covered), provide subtle hints to encourage deeper thinking, without stating it’s fully correct.
    4. If the answer is incorrect (no key points are covered), provide a gentle nudge or guiding question without directly revealing the answer.
    5. After 3 unsuccessful attempts, reveal the correct answer and suggest reviewing specific concepts from Module 6F, Lesson 06.
    6. Always respond in first person to maintain a supportive and educational tone.

    Respond educationally and supportively, strictly adhering to the lesson content and avoiding unrelated information or outside knowledge.
    """

    # Generate feedback using GPT-4o
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    feedback = response.choices[0].message["content"]

    return feedback


def match_key_points(user_answer, key_points, threshold=0.7):
    user_embedding = get_embedding(user_answer)
    matched_points = []

    for point in key_points:
        point_embedding = get_embedding(point)
        similarity = cosine_similarity(user_embedding.reshape(1, -1), point_embedding.reshape(1, -1))[0][0]
        
        if similarity > threshold:
            matched_points.append(point)

    return matched_points

# Main function to display the assessment instructions
def display_instructions():
    st.title("Educational Assistant Chatbot 🤖")
    st.write("*Hey! I will help you assess and guide on Module 6F, Lesson 06 about Moonsighting.*")
    st.subheader("Assessment Instructions:")
    
    st.write("""
    - **Question Structure**: The assessment has *5 scenario-based questions*, asked in a specific order.
    - **Attempts and Feedback**: You have *3 attempts* to answer each question correctly.
    - **Strictness**: Only *fully correct answers* are accepted; partial answers will prompt for further clarification.
    - **Final Explanation**: After *3 unsuccessful attempts*, I will provide the correct answer and suggest reviewing the lesson.
    - **Correct Answers**: *Correct answers* will be acknowledged, and we will move on to the next question.
    """)

    st.markdown("<br>", unsafe_allow_html=True) 

    # Creating columns to center the button
    col1, col2, col3 = st.columns([5, 3, 5])  # Middle column is wider

    with col2:
        # Centering the button in the middle column
        if st.button("Start Assessment", key="start_assessment", on_click=start_assessment):
            pass  # `on_click` will handle the transition

def start_assessment():
    st.session_state["page"] = "assessment"

# Function to display progress in the sidebar
def display_sidebar_progress():
    progress = st.session_state.get("progress", {"correct_answers": 0, "attempts_per_question": {}})
    st.sidebar.header("Your Progress")
    st.sidebar.metric("Questions Answered Correctly", progress["correct_answers"])
    st.sidebar.button("Restart Quiz", on_click=restart_quiz)

# Function to simulate typing effect
def simulate_typing(text, delay=0.01):
    container = st.empty()
    displayed_text = ""
    for char in text:
        displayed_text += char
        container.markdown(displayed_text)
        time.sleep(delay)

# Function to handle proceeding to the next question
def proceed_to_next_question():
    st.session_state["current_question_index"] += 1
    st.session_state["attempts"] = 0
    st.session_state["show_proceed_button"] = False

# Function to handle returning to the previous question
def return_to_previous_question():
    if st.session_state["current_question_index"] > 0:
        st.session_state["current_question_index"] -= 1
        st.session_state["attempts"] = st.session_state["attempts_per_question"].get(st.session_state["current_question_index"], 0)
        st.session_state["show_proceed_button"] = True  # Disable input for previous question

# Function to handle resuming the most recent question
def resume_current_question():
    st.session_state["current_question_index"] = st.session_state["most_recent_question_index"]
    st.session_state["attempts"] = st.session_state["attempts_per_question"].get(st.session_state["current_question_index"], 0)
    st.session_state["show_proceed_button"] = st.session_state["button_states"].get(st.session_state["current_question_index"], False)

# Function to handle restarting the quiz
def restart_quiz():
    st.session_state["page"] = "instructions"
    st.session_state["current_question_index"] = 0
    st.session_state["most_recent_question_index"] = 0
    st.session_state["attempts"] = 0
    st.session_state["progress"] = {"correct_answers": 0, "attempts_per_question": {}}
    st.session_state["chat_histories"] = {}
    st.session_state["button_states"] = {}
    st.session_state["attempts_per_question"] = {}
    st.session_state["show_proceed_button"] = False
    st.session_state["question_completed"] = {}

# Main function to display the quiz
def display_quiz():
    display_sidebar_progress()  # Show progress in sidebar

    feedback = None

    # Get the current question index and data
    current_index = st.session_state["current_question_index"]
    current_question = questions[current_index]

    # Initialize chat histories and button state for the current question if not present
    if current_index not in st.session_state["chat_histories"]:
        st.session_state["chat_histories"][current_index] = []
    if current_index not in st.session_state["button_states"]:
        st.session_state["button_states"][current_index] = False
    if current_index not in st.session_state["attempts_per_question"]:
        st.session_state["attempts_per_question"][current_index] = 0

    # Restore the button state and attempts for the current question
    st.session_state["show_proceed_button"] = st.session_state["button_states"][current_index]
    st.session_state["attempts"] = st.session_state["attempts_per_question"][current_index]

    # Update the most recent question index if we're on the latest question
    if current_index >= st.session_state["most_recent_question_index"]:
        st.session_state["most_recent_question_index"] = current_index

    # Display question
    st.write(f"### {current_question['scenario_number']}")
    st.write(f"*{current_question['scenario']}*")
    st.write(f"**{current_question['question']}**")

    # Display chat history for the current question
    for entry in st.session_state["chat_histories"][current_index]:
        role = entry["role"]
        content = entry["content"]
        trial_count = entry.get("trial_count", None)
        
        # Display trial count above each user response
        if role == "user" and trial_count:
            st.write(f"**Trial: {trial_count} of 3**")
        
        # Display message
        st.chat_message(role).write(content)

    # Check if attempts have reached the maximum or if answer is fully correct
    if st.session_state["attempts"] >= 3 or st.session_state["question_completed"].get(current_index, False):
        st.error("This question is complete. Please proceed to the next question or review the lesson.", icon="❗")
        st.session_state["show_proceed_button"] = True  # Set the proceed button to display
    else:
        # Show the chat input only if the question is not complete
        if st.session_state["attempts"] < 3 and not st.session_state["question_completed"].get(current_index, False):
            # Display chat input if answer is not fully correct and attempts are below 3
            if user_input := st.chat_input("Type your answer here"):
                # Display trial count above the user's response for the first and subsequent trials
                st.write(f"**Trial: {st.session_state['attempts'] + 1} of 3**")

                # Increment session state attempts immediately
                st.session_state["attempts"] += 1
                st.session_state["attempts_per_question"][current_index] = st.session_state["attempts"]

                # Append user input to chat history for the current question with trial count
                st.session_state["chat_histories"][current_index].append({
                    "role": "user", 
                    "content": user_input,
                    "trial_count": st.session_state["attempts"]
                })
                st.chat_message("user").write(user_input)  # Display the user's input

                # Display a spinner while processing the answer
                with st.spinner('💭 Checking your answer...'):
                    time.sleep(0.5)  # Simulate delay for demonstration
                    feedback = generate_feedback(current_question, user_input, st.session_state["attempts"])

                # Append feedback to chat history for the current question
                st.session_state["chat_histories"][current_index].append({"role": "assistant", "content": feedback})
                with st.chat_message("assistant"):
                    simulate_typing(feedback)  # Use typing simulation for assistant response

                # Process attempts and correct answers
                if "This answer is fully correct" in feedback:
                    st.session_state["show_proceed_button"] = True
                    st.session_state["progress"]["correct_answers"] += 1
                    st.session_state["question_completed"][current_index] = True  # Mark question as completed

                elif st.session_state["attempts"] >= 3:
                    # After 3 attempts, let GPT-4 suggest review materials
                    feedback = generate_feedback(current_question, user_input, st.session_state["attempts"])
                    st.session_state["chat_histories"][current_index].append({"role": "assistant", "content": feedback})
                    
                    with st.chat_message("assistant"):
                        simulate_typing(feedback)
                    st.session_state["show_proceed_button"] = True
                    st.session_state["question_completed"][current_index] = True  # Mark question as completed
                else:
                    st.session_state["show_proceed_button"] = False

                # Save the button state for the current question
                st.session_state["button_states"][current_index] = st.session_state["show_proceed_button"]

    # Display success message if the user got the answer right on the first try
    if st.session_state["attempts"] == 1 and feedback and "This answer is fully correct" in feedback:
        st.success("Great job! You got it right on the first try! 🌟")

    # Display "Proceed to the next question" button if answer is correct or 3 attempts reached
    if st.session_state.get("show_proceed_button", False) and current_index == st.session_state["most_recent_question_index"]:
        st.button("Proceed to the next question", key="proceed_next", on_click=proceed_to_next_question)

    # Display "Return to previous question" button if not on the first question
    if current_index > 0:
        st.button("Return to previous question", key="return_previous", on_click=return_to_previous_question)

    # Display "Resume Current Question" button only if the user is on a previous question
    if current_index < st.session_state["most_recent_question_index"]:
        st.button("Resume Current Question", key="resume_current", on_click=resume_current_question)



# Run the app with a page-based structure
def main():
    if "page" not in st.session_state:
        st.session_state["page"] = "instructions"
    if "progress" not in st.session_state:
        st.session_state["progress"] = {"correct_answers": 0, "attempts_per_question": {}}
    if "current_question_index" not in st.session_state:
        st.session_state["current_question_index"] = 0
    if "most_recent_question_index" not in st.session_state:
        st.session_state["most_recent_question_index"] = 0  # Track the last active question
    if "attempts" not in st.session_state:
        st.session_state["attempts"] = 0
    if "chat_histories" not in st.session_state:
        st.session_state["chat_histories"] = {}
    if "button_states" not in st.session_state:
        st.session_state["button_states"] = {}
    if "attempts_per_question" not in st.session_state:
        st.session_state["attempts_per_question"] = {}
    if "show_proceed_button" not in st.session_state:
        st.session_state["show_proceed_button"] = False
    if "question_completed" not in st.session_state:
        st.session_state["question_completed"] = {}

    # Decide which page to display
    if st.session_state["page"] == "instructions":
        display_instructions()
    else:
        display_quiz()



# Run the app
if __name__ == "__main__":
    main()
