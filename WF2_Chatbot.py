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
import base64
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
import io

# Load environment variables and set OpenAI key
openai.api_key = st.secrets["openai"]["OPENAI_API_KEY"]

# Initialize ChromaDB
CHROMA_DATA_PATH = "chromadb_WF2_chatbot/"
COLLECTION_NAME = "document_embeddings"

# Initialize ChromaDB client and embedding function
client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai.api_key,
    model_name="text-embedding-ada-002"
)
collection = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=openai_ef)


# Set up Streamlit app
st.set_page_config(page_title="Quiz Chatbot", page_icon=":books:", layout="wide")

# Function to convert an image file to a base64 string
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Paths to the avatar images (Replace with your actual paths)
user_avatar_path = "female-avatar.png"
assistant_avatar_path = "avatar_open.png"

# Load avatar images (assuming you have two images for simplicity)
avatar_open_path = "avatar_open.png"
avatar_closed_path = "avatar_closed.png"

# Convert images to base64
user_avatar_base64 = image_to_base64(user_avatar_path)
assistant_avatar_base64 = image_to_base64(assistant_avatar_path)

# CSS for chat UI
st.markdown("""
    <style>
    .chat-container {
        height: 100px;
        overflow-y: auto;
        padding: 0px;
        border-radius: 0px;
        background-color: #f5f5f5;
    }
    .user-message {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        margin: 10px 0;
    }
    .assistant-message {
        display: flex;
        justify-content: flex-start;
        align-items: center;
        margin: 10px 0;
    }
    .message-bubble {
        padding: 20px;
        border-radius: 15px;
        margin: 0px;
        max-width: 70%;
    }
    .user-bubble {
        background-color: #DCF8C6;
        color: black;
    }
    .assistant-bubble {
        background-color: #ADE8F4;
        color: black;
    }
    .user-avatar {
        width: 80px;
        height: 80px;
        border-radius: 20%;
        margin: 5px;
    }
    .assistant-avatar {
        width: 250px;
        height: 250px;
        border-radius: 10%;
        margin: 5px;
    }
    .stButton button {
        width: 100%;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

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
        "context": "The Islamic lunar calendar depends on the phases of the moon, and the new month starts with the sighting of the crescent moon. Since cities to the west may see the moon after eastern locations, the sighting can vary regionally.",
        "specific_section": "The Phases of the Moon"
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
        "context": "Recommended actions include charity, prayer, and reflection on personal growth as the new month is a time for renewing commitment and dedication to Allah.",
        "specific_section": "Recommended actions when sighting the new moon"
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
        "context": "The lunar calendar is about 11 days shorter than the solar calendar, causing Islamic events to shift through different seasons, allowing diverse experiences for Muslims.",
         "specific_section": "Benefit of a lunar calendar"
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
        "context": "Seeing the crescent moon symbolizes the start of a new month. It’s a moment to reflect on time, and we can recite a dua, such as the one from Imam Zayn al-Abidin, to seek blessings and guidance.",
        "specific_section": "Faith in action"
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
        "context": "Ayatullah Sistani supports the unity of horizons, allowing cities with similar geographic horizons to follow each other's sightings, which considers Earth’s spherical nature and rotation.",
        "specific_section": "Differences of opinion"
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
    9. Determine if the student's answer is fully correct by checking if it covers all key points.
    10. If the answer covers all key points, state first "This answer is fully correct" AT ALL TIMES. 
    11. If the answer is partially correct (some key points are covered), provide subtle hints to encourage deeper thinking, without stating it’s fully correct.
    12. If the answer is incorrect (no key points are covered), provide a gentle nudge or guiding question without directly revealing the answer.
    13. After 3 unsuccessful attempts, reveal the correct answer and suggest reviewing the SPECIFIC sub topic under {specific_section} that covers the question from Module 6F, Lesson 06 Moonsighting
    14. Always respond in first person to maintain a supportive and educational tone.
"""


# Function to get embeddings for a given text
def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")  # Preprocess text
    response = client.embeddings.create(input=[text], model=model)  # Call OpenAI API
    return np.array(response.data[0].embedding)

# Function to query ChromaDB for the most relevant context based on the question
def retrieve_context(query, specific_section=None):
    # Query ChromaDB with the question text and specific section/subtopic if provided
    query_texts = [query]
    if specific_section:
        query_texts.append(specific_section)

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


def generate_feedback(question_data, user_answer, attempt_number):
    context_text = retrieve_context(question_data["question"])

    # Define hint level based on the attempt number
    if attempt_number == 1:
        hint_level = "Provide high-level hints to encourage exploration without direct answers."
    elif attempt_number == 2:
        hint_level = "Provide specific hints or reference missing key points indirectly."
    elif attempt_number == 3:
        # If it’s the third attempt, get specific section from ChromaDB
        specific_section = question_data.get("specific_section", "General Guidance")
        context_text = retrieve_context(question_data["question"], specific_section=specific_section)
        hint_level = "Provide a full answer and suggest reviewing specific content or subtopic that contain the answer for the question from Module 6F, Lesson 06."

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
    2. If the answer covers all key points, state "This answer is fully correct" AT ALL TIMES.
    3. If the answer is partially correct (some key points are covered), provide subtle hints to encourage deeper thinking, without stating it’s fully correct.
    4. If the answer is incorrect (no key points are covered), provide a gentle nudge or guiding question without directly revealing the answer.
    5. After 3 unsuccessful attempts, reveal the correct answer and suggest reviewing the SPECIFIC sub topic that covers the question from Module 6F, Lesson 06 Moonsighting
    6. Always respond in first person to maintain a supportive and educational tone.



    Respond educationally and supportively, strictly adhering to the lesson content and avoiding unrelated information or outside knowledge.
    
    {hint_level}
    """

    # Generate feedback using GPT-4o
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    feedback = response.choices[0].message.content

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
    st.title("MCE Chatbot")
    st.write("*Salaam alaykum! I will help you assess and guide on Module 6F, Lesson 06 about Moonsighting.*")
    st.subheader("Assessment Instructions:")

    st.markdown("""
    1. **Question Structure**: The assessment has 5 scenario-based questions. Questions will be asked in a specific order.
    """, unsafe_allow_html=True)
    st.markdown("""
    2. **Attempts and Feedback**: You have 3 attempts to answer each question correctly.
        - *If your response is incorrect or only partially correct*, I will provide subtle guidance, nudges, or prompts to encourage you to think deeper without directly revealing the answer.
        - *Strictness*: I will be strict in evaluating your responses, considering only complete answers as fully correct. If your answer is partial, I will ask for additional details or clarification before moving to the next question.
        - *Final Explanation*: After 3 unsuccessful attempts, I will provide the correct answer along with directions to review specific lesson content in more detail.
    """, unsafe_allow_html=True)
    st.markdown("""
    3. **Correct Answers**: When you provide a correct answer, I will acknowledge it, offer additional insights if needed, and then proceed to the next question.
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True) 

    # Creating columns to center the button
    col1, col2, col3 = st.columns([5, 3, 5])  # Middle column is wider

    with col2:
        # Centering the button in the middle column
        if st.button("Start Assessment", key="start_assessment", on_click=start_assessment):
            pass  # `on_click` will handle the transition

def start_assessment():
    st.session_state["page"] = "assessment"


# Function to generate speech from text
def generate_speech(text, voice="alloy"):
    try:
        # Create a temporary file for the audio output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file_path = tmp_file.name

        # Stream the audio directly to the temporary file
        response = openai.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )
        response.stream_to_file(tmp_file_path)

        # Return the temporary file path for playback
        return tmp_file_path
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None
    

# Function to display progress in the sidebar
def display_sidebar_progress():
    progress = st.session_state.get("progress", {"correct_answers": 0, "attempts_per_question": {}})
    st.sidebar.header("⏳ Your Progress", divider='blue')
    st.sidebar.metric("Questions Answered Correctly:", progress["correct_answers"])
    
    st.sidebar.write("")
    st.sidebar.write("")

    # Settings
    st.sidebar.header("⚙️ Settings", divider='blue')

    # Text Size
    with st.sidebar.expander("🗚 Text Size", expanded=True):
        text_size = st.selectbox("Select text size:", ["Small", "Medium", "Large", "Extra Large"], index=1)

    # TTS settings
    with st.sidebar.expander("🔊 Text-to-Speech", expanded=True):
        enable_tts = st.sidebar.toggle("Enable Text-to-Speech", value=False, key="enable_tts")

        if enable_tts:
            # Voice selection
            voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            selected_voice = st.sidebar.selectbox(
                "Select Voice",
                voice_options,
                index=0,
                key="tts_voice"
            )
            

    # Apply selected text size
    font_size_map = {
        "Small": "14px",
        "Medium": "16px",
        "Large": "18px",
        "Extra Large": "20px"
    }
    selected_font_size = font_size_map[text_size]

    # Inject dynamic CSS to control text size based on user selection
    st.markdown(
        f"""
        <style>
        .appview-container, .element-container, .stText, .stMarkdown p {{
            font-size: {selected_font_size} !important;
        }}
        .stButton button {{
            font-size: {selected_font_size} !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")
    st.sidebar.write("")

    st.sidebar.button("Restart Quiz", icon="🔄", on_click=restart_quiz)

# Function to simulate typing effect
def simulate_typing_with_moving_lips(text, delay=0.01, batch_size=10, lip_sync_interval=2):
    """
    Simulates typing text with moving lips by alternating between two avatar images.
    
    Parameters:
        text (str): The text to display as typing.
        delay (float): Time delay between updates for typing effect.
        batch_size (int): Number of characters to display at each update.
        lip_sync_interval (int): How often to toggle mouth position, in terms of display updates.
    """
    container = st.empty()
    displayed_text = ""
    open_mouth = True  # Toggle mouth position

    # Initialize a counter to control lip sync frequency
    lip_sync_counter = 0

    # Loop through the text in chunks defined by batch_size
    for i in range(0, len(text), batch_size):
        # Append the next chunk of text to displayed_text
        displayed_text += text[i:i+batch_size]

        # Toggle avatar image based on the lip_sync_counter
        if lip_sync_counter % lip_sync_interval == 0:
            open_mouth = not open_mouth

        avatar_img = avatar_open_path if open_mouth else avatar_closed_path

        # Display the avatar with the current state of text
        container.markdown(
            f"""
            <div class="assistant-message">
                <img src="data:image/png;base64,{image_to_base64(avatar_img)}" class="assistant-avatar" alt="Assistant Avatar">
                <div class="message-bubble assistant-bubble">{displayed_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Increment the lip sync counter
        lip_sync_counter += 1

        # Adjust speed for text typing and animation
        time.sleep(delay)

    # Ensure mouth is closed after speaking
    container.markdown(
        f"""
        <div class="assistant-message">
            <img src="data:image/png;base64,{image_to_base64(avatar_closed_path)}" class="assistant-avatar" alt="Assistant Avatar">
            <div class="message-bubble assistant-bubble">{displayed_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Generate and play speech if TTS is enabled
    if st.session_state.get("enable_tts", False):
        speech_file = generate_speech(
            text,
            voice=st.session_state.get("tts_voice", "alloy")
        )
        if speech_file:
            with open(speech_file, "rb") as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format="audio/mp3")
            # Clean up the temporary file
            os.unlink(speech_file)



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

# Function to handle completing the quiz
def display_results():
    # Display the score
    score = st.session_state["progress"]["correct_answers"]
    st.title("Assessment Results")
    st.write(f"You got {score} out of {len(questions)} questions correct.")
    st.write("")
    st.write("")

    # Display all chat history across questions
    st.subheader("Review your assessment:")
    st.divider()
    for i, question in enumerate(questions):
        st.write(f"### {question['scenario_number']}")
        st.write(f"*{question['scenario']}*")
        st.write(f"**{question['question']}**")
        
        # Display chat history for each question
        for entry in st.session_state["chat_histories"].get(i, []):
            role = entry["role"]
            content = entry["content"]
            if role == "user":
                st.write(f"**You**: {content}")
            else:
                st.write(f"**Assistant**: {content}")

        st.divider()
    

    # Add an "Exit" button at the bottom
    if st.button("Exit", on_click=exit_quiz):
        st.write("Returning to the instruction page...")  # Optional message

def complete_quiz():
    st.session_state["page"] = "complete"


def exit_quiz():
    # Clear all session state variables to reset the quiz
    for key in st.session_state.keys():
        del st.session_state[key]
    
    # Redirect to the instructions page
    st.session_state["page"] = "instructions"



def display_chat_history(chat_history):
    for entry in chat_history:
        role = entry["role"]
        content = entry["content"]
        attempt_count = entry.get("attempt_count", None)
        
        # Display attempt count if present
        if role == "user" and attempt_count:
            st.markdown(f"**Attempt: {attempt_count} of 3**")
        
        # Display message with enhanced UI
        if role == "user":
            st.markdown(
                f"""
                <div class="user-message">
                    <div class="message-bubble user-bubble">{content}</div>
                    <img src="data:image/png;base64,{user_avatar_base64}" class="user-avatar" alt="User Avatar">
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="assistant-message">
                    <img src="data:image/png;base64,{assistant_avatar_base64}" class="assistant-avatar" alt="Assistant Avatar">
                    <div class="message-bubble assistant-bubble">{content}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )

            # Generate and play speech for assistant messages if TTS is enabled
            if st.session_state.get("enable_tts", False) and not entry.get("tts_played", False):
                speech_file = generate_speech(
                    content,
                    voice=st.session_state.get("tts_voice", "alloy")
                )
                if speech_file:
                    with open(speech_file, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/mp3")
                    # Clean up the temporary file
                    os.unlink(speech_file)
                    # Mark this message as played
                    entry["tts_played"] = True


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
    st.write("")
    st.write(f"**{current_question['question']}**")

    # Display chat history using display_chat_history() (initial full history display)
    display_chat_history(st.session_state["chat_histories"][current_index])

    # Check if attempts have reached the maximum or if answer is fully correct
    if st.session_state["attempts"] >= 3 or st.session_state["question_completed"].get(current_index, False):
        st.error("This question is complete. Please proceed to the next question or review the lesson.", icon="❗")
        st.session_state["show_proceed_button"] = True  # Set the proceed button to display
    else:
        # Show the chat input only if the question is not complete
        if st.session_state["attempts"] < 3 and not st.session_state["question_completed"].get(current_index, False):
            # Display chat input if answer is not fully correct and attempts are below 3
            if user_input := st.chat_input("Type your answer here"):
                # Increment session state attempts immediately
                st.session_state["attempts"] += 1
                st.session_state["attempts_per_question"][current_index] = st.session_state["attempts"]

                # Append user input to chat history for the current question with attempt count
                st.session_state["chat_histories"][current_index].append({
                    "role": "user", 
                    "content": user_input,
                    "attempt_count": st.session_state["attempts"]
                })

                # Display the new user input directly without re-rendering the full history
                display_chat_history([{
                    "role": "user",
                    "content": user_input,
                    "attempt_count": st.session_state["attempts"]
                }])

                # Display a spinner while processing the answer
                with st.spinner('💭 Checking your answer...'):
                    time.sleep(0.5)  # Simulate delay for demonstration
                    feedback = generate_feedback(current_question, user_input, st.session_state["attempts"])

                # Append only the new feedback to chat history for the current question
                st.session_state["chat_histories"][current_index].append({"role": "assistant", "content": feedback})

                # Use simulate_typing_with_moving_lips for the assistant's response
                simulate_typing_with_moving_lips(feedback)

                # Process attempts and correct answers
                if "fully correct" in feedback:
                    st.session_state["show_proceed_button"] = True
                    st.session_state["progress"]["correct_answers"] += 1
                    st.session_state["question_completed"][current_index] = True  # Mark question as completed

                elif st.session_state["attempts"] >= 3:
                    # After 3 attempts, suggest review materials
                    st.session_state["show_proceed_button"] = True
                    st.session_state["question_completed"][current_index] = True  # Mark question as completed
                else:
                    st.session_state["show_proceed_button"] = False

                # Save the button state for the current question
                st.session_state["button_states"][current_index] = st.session_state["show_proceed_button"]

    # Display success message if the user got the answer right on the first try
    if st.session_state["attempts"] == 1 and feedback and "fully correct" in feedback:
        st.success("Great job! You got it right on the first try! 🌟")

    # Display "Proceed to the next question" button if answer is correct or 3 attempts reached
    if st.session_state.get("show_proceed_button", False) and current_index == st.session_state["most_recent_question_index"]:
        # Check if we are on the last question
        if current_index == len(questions) - 1:
            # Only display "Complete Quiz" button if answer is fully correct or attempts reached 3
            if st.session_state["question_completed"].get(current_index, False) or st.session_state["attempts"] >= 3:
                st.button("Complete Quiz", on_click=complete_quiz)
        else:
            # Otherwise, show the "Proceed to the next question" button for intermediate questions
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
    elif st.session_state["page"] == "assessment":
        display_quiz()
    elif st.session_state["page"] == "complete":
        display_results()

# Run the app
if __name__ == "__main__":
    main()
