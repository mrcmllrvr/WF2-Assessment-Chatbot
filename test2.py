# (missing audio player and play voice over button when changing the text size)

# Import pysqlite3 and swap it with the default sqlite3 library
import sys
__import__('pysqlite3')  # Corrected this line
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
import streamlit as st
import openai
import numpy as np
from threading import Lock
import os
import time
import base64
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import tempfile
from mutagen.mp3 import MP3
import threading


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


# Paths to the wf and tarbiyah logo
logo_tarbiyah_path = "logo-tarbiyah.png"


# Load avatar images (assuming you have two images for simplicity)
avatar_open_path = "avatar_open.png"
avatar_closed_path = "avatar_closed.png"

# Ensure paths to avatars are properly encoded
open_avatar_base64 = image_to_base64(avatar_open_path)
closed_avatar_base64 = image_to_base64(avatar_closed_path)



# CSS for chat UI
st.markdown("""
    <style>
    .chat-container {
        height: 0px;
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
        padding: 15px;
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
       "scenario_number": "Scenario 1", # Originally Q1 
       "scenario": "The New Islamic Month",
       "question": "Why is it so important for Muslims to know when a new Islamic month begins?",
       "key_points": [
            "Express understanding that the start date determines when religious obligations/occasions occur. (Religious obligations/practices tied to specific months (like Ramadhan)"
            "Express understanding that the start date determines when religious obligations/occasions occur.  (Important religious occasions/commemorations)",
            "Express understanding that the start date determines when religious obligations/occasions occur. (Essential religious days during the month)",
       ],
       "partial_answer": [
           "Gives a vague answer about religious importance without specifics",
           "Only mentions calendar dates without connecting to religious significance",
           "Only gives examples without explaining the importance",
           "Focuses only on one type of occasion (e.g., only fasting) without showing broader understanding",
       ],
       "incorrect_answer": [
           "Focuses only on moon sighting methods",
           "Discusses lunar vs solar calendar differences",
           "Gives cultural rather than religious reasons",
           "Mentions unrelated benefits of Islamic calendar",
       ],
       "note": "Any 1 of the key points is considered right answer",
       "learn_more": "https://tarbiyah.education/topic/module-6f-06-differences-of-opinion/?tb_action=complete&prev_step_id=40648"
   },
   {
       "scenario_number": "Scenario 2", # Originally Q3
       "scenario": "Your parents heard that your madrasah lesson was about moonsighting. They want to know what you learned about establishing that a new lunar month has begun.",
       "question": "According to Ayatullah Sistani, there are 4 ways. Can you tell them 2?",
       "key_points": [
           "Personal sighting (seeing the crescent moon with your own eyes)",
           "Testimony of a reliable group (e.g., trusted local Shia community/mosque/organization)",
           "Testimony of two adil (trusted) people who have seen the moon",
           "Completion of 30 days of the current month"
       ],
       "partial_answer": [
           "Lists one correct way precisely and one vaguely",
           "Lists one correct way only",
           "Uses imprecise language but shows understanding (e.g., 'when good people see it' instead of 'two adil people')",
           "Gives correct examples but doesn't clearly state the principle"
       ],
       "incorrect_answer": [
           "Lists no correct ways",
           "Confuses different methods",
           "Mentions unrelated criteria",
           "Misunderstands what makes testimony acceptable",
           "Confuses individual vs group testimony requirements"
       ],
       "note": "Any 2 of the key points are considered right answer",
       "learn_more": "https://tarbiyah.education/topic/module-6f-06-benefit-of-a-lunar-calendar/?tb_action=complete&prev_step_id=40606"
   },
   {
       "scenario_number": "Scenario 3", # Originally Q2
       "scenario": "Your classmate asks you why Muslims use a lunar calendar instead of the solar calendar, especially since the number of days in a month is fixed in a solar calendar.",
       "question": "What is one important benefit of using the lunar calendar for Islamic occasions like Ramadhan and Hajj?",
       "key_points": [
            "Clearly explain that Islamic occasions move through different seasons over the years",
            "Include reference to either Ramadhan, Hajj, or both as examples",
            "Show understanding that this movement is due to the lunar calendar",
       ],
        "partial_answer": [
            "Only mentions that dates change without explaining seasonal movement",
            "Explains seasonal changes but doesn't connect it to Islamic occasions",
            "Makes the point about different regions experiencing occasions in different seasons without mentioning the yearly progression"
        ],
        "incorrect_answer": [
            "Only discusses fixed vs changing month lengths",
            "Focuses on moon sighting rules rather than seasonal benefits",
            "Gives unrelated benefits of the lunar calendar",
            "Suggests that lunar calendar was chosen for ease of use"
        ],
       "learn_more": "https://tarbiyah.education/topic/module-6f-06-recommended-actions-when-sighting-the-new-moon/"
   },
   {
       "scenario_number": "Scenario 4", # Originally Q4
       "scenario": "When you learn that a new Islamic month has begun,",
       "question": "name any TWO mustahabb (recommended) actions that you should try to perform on the first night.",
       "key_points": [
            "Giving charity",
            "Praying 2 rak'ah prayer",
            "Reciting the du'a for seeing the new crescent"
        ],
        "partial_answer": [
            "Lists only one correct specific action",
            "Lists two but one is too vague (e.g., 'pray' instead of 'pray 2 rak'ah')",
            "Uses imprecise language (e.g., 'give money' instead of 'give charity/sadaqah')"
        ],
        "incorrect_answer": [
            "Lists no correct actions",
            "Lists completely different actions",
            "Lists actions too vaguely to be identifiable",
            "Confuses these with actions for other occasions"
        ],
       "note": "Any 2 of the key points are considered right answer",
       "learn_more": "https://tarbiyah.education/topic/module-6f-06-ways-to-tell-it-is-the-new-month/?tb_action=complete&prev_step_id=40617"
   },
   {
       "scenario_number": "Scenario 5", # Originally Q5
       "scenario": "Your local Islamic community has confirmed the sighting of the new crescent moon marking the start of Ramadhan. Your friend Bilal lives in another city that shares the same horizon with your city.",
       "question": "According to Ayatullah Sistani's ruling, can Bilal start his Ramadhan fasts based on your city's moon sighting? Explain your answer.",
       "key_points": [
           "Yes, Bilal can rely on the sighting from your city because they share the same horizon",
           "The first of the month established in one city is automatically established in other cities that share the same horizon",
           ],
        "partial_answer": [
            "Only mentions that Bilal can rely on the sighting without explaining the horizon connection",
            "Only mentions shared horizon without explicitly stating that the month is established",
            "Gives correct conclusion but incorrect reasoning",
            "Does not put 'Yes' but the explanation is correct"
        ],
        "incorrect_answer": [
            "States that each city must sight independently",
            "States that shared horizon is not relevant",
            "Confuses geographical proximity with horizon unity",
            "Suggests additional requirements beyond shared horizon"
        ],
       "note": "Any 2 of the correct answers is considered right answer",
       "learn_more": "https://tarbiyah.education/topic/module-6f-06-intro/?tb_action=complete&prev_step_id=40639"
   },
]


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
    # Initialize session storage for previous answers if not already done
    if "previous_answers" not in st.session_state or not isinstance(st.session_state["previous_answers"], list):
        st.session_state["previous_answers"] = []

    # Store the most recent answer 
    st.session_state["most_recent_answer"] = user_answer

    # Add the current user answer to session-based memory, avoiding duplicates
    if user_answer.strip() and user_answer not in st.session_state["previous_answers"]:
        st.session_state["previous_answers"].append(user_answer)

    # Combine all previous answers for cumulative evaluation
    all_answers = " ".join(st.session_state["previous_answers"]).replace("\n", " ")



    # Generate dynamic assessment criteria based on question's note
    if "note" in question_data and "Any 2 of the key points are considered right answer" in question_data["note"]:
        assessment_criteria = (
            f"If any 2 key points are covered in the student's {all_answers} across attempts, the answer is right answer. ",
            # f"If any partial answers are covered in the student's {all_answers}, the answer is partially correct.",
            # f"If incorrect answers are covered in the student's {all_answers}, the answer is incorrect."
            "If no key points are covered, the answer is incorrect."
        )
    elif "note" in question_data and "Any 1 of the key points is considered right answer" in question_data["note"]:
        assessment_criteria = (
            f"If any key point is covered in the student's {all_answers} accross attempts, the answer is right answer."
            # f"If any partial answers are covered in the student's {all_answers}, the answer is partially correct.",
            # f"If incorrect answers are covered in the student's {all_answers}, the answer is incorrect."
            "If no key points are covered, the answer is incorrect."
        )
    else: # all key points are needed
        assessment_criteria = (
            f"If all key points are covered in {all_answers} across attempts, the answer is right answer. "
            # f"If any partial answers are covered in the student's {all_answers}, the answer is partially correct.",
            # f"If incorrect answers are covered in the student's {all_answers}, the answer is incorrect."
            "If some key points are covered, the answer is partially correct."
            "If no key points are covered, the answer is incorrect."
        )

    # Dynamically generate the lesson link
    lesson_link = f"[here]({question_data['learn_more']})"
    # lesson_link = f"[here]({question_data['learn_more']})"    
    # st.write(f"Debug: Rendering Markdown as: {lesson_link}") 

    # Retrieve lesson context
    context_text = retrieve_context(question_data["question"])

    # Adjust hint level based on the attempt number
    if attempt_number == 2:
        hint_level = "Provide specific hints or reference missing key points indirectly. DO NOT GIVE THE ANSWER AWAY IF THE ANSWER IS PARTIALLY CORRECT OR INCORRECT - JUST PROVIDE FEEDBACK."
    else:  # attempt_number >= 3
        hint_level = f"Provide the complete answer and direct the student to review this topic at: {lesson_link}"

    system_prompt = f"""
    ### CORE RULES
    - Evaluate ALL cumulative answers: {all_answers}
    - If attempt < 3: NO answers revealed
    - NEVER use correct answers as hints
    - Correct answer MUST start with "This is the right answer"
    - Add "(saww)" after "Muhammad"
    - Maintain supportive, educational tone
    - Base feedback strictly on Module 6F, Lesson 06

    ### REQUIRED RESPONSE FORMAT
    CORRECT ANSWER:
    ```
    This is the right answer! [One sentence acknowledging the correct element mentioned in the student's response], because [one sentence explaining why this element is important].

    To expand your understanding, [one sentence about related concepts]. Your learning journey continues here: {lesson_link}
    ```

    PARTIALLY CORRECT (attempts < 3):
    ```
    You're making progress! You've covered [X] out of [Y] points. [One sentence acknowledging what they understand].

    Let's explore further: [One focused question about a broader concept]. Keep building your understanding by reviewing the lesson materials - you're on the right track!
    ```

    INCORRECT (attempts < 3):
    ```
    Keep going! Your answer needs some adjustment.

    Let's think about: [One broad question about the main lesson theme]. You can strengthen your understanding by reviewing the core concepts from our lesson. I'm here to help you learn!
    ```

    THIRD ATTEMPT:
    ```
    Thank you for your efforts! Let's review the complete explanation together:

    [Full answer]

    Key Points to Remember:
    [List points]

    Continue your learning journey here: {lesson_link}
    Remember, understanding takes time, and each attempt helps build your knowledge.
    ```

    ### CONTEXT
    Scenario: {question_data["scenario"]}
    Question: {question_data["question"]}
    Criteria: {assessment_criteria}
    Current Attempt: {attempt_number}

    ### FEEDBACK PRINCIPLES
    1. Protection:
    - No direct/indirect answer hints
    - No revealing terminology
    - No specific corrections

    2. Guidance:
    - Focus on demonstrated understanding
    - Use open-ended questions
    - Connect to lesson themes
    - Encourage deeper thinking

    3. Progress:
    - Track cumulative points covered
    - Acknowledge growth
    - Maintain forward momentum

    ### VERIFICATION
    - Follows exact response structure
    - Feedback reveals NO answers BUT gives subtle hints
    - No answers revealed before attempt 3
    - Uses appropriate guidance level
    - Links included when required
    - Maintains encouraging tone
    - Format matches attempt number
    - Progress accurately tracked

    """




    # Generate feedback using GPT-4o
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_answer}
        ]
    )

    feedback = response.choices[0].message.content
    #st.markdown(feedback, unsafe_allow_html=True)
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
    col1, col2 = st.columns([1, 4])

    with col1:
        st.image("avatar.png", use_container_width=True)

    with col2:
        st.title("MCE Know-Bot")
        st.write("*Salaam alaykum! I will help you assess and guide on Module 6F, Lesson 06 about Moonsighting.*")

    st.subheader("Guidelines:")

    st.markdown("""
    1. **Questions**: There are 5 scenario based questions for this topic. They will be asked in a specific order..
    """, unsafe_allow_html=True)
    st.markdown("""
    2. **Answers**: You have 3 attempts to answer each question correctly.
        - *If your response is correct*, I will acknowledge it, offer additional insights if needed, and then proceed to the next question.
        - *If your response is incorrect or partially correct*, I will provide subtle guidance, nudges, or prompts to encourage you to think deeper without directly revealing the answer.
        - I will be precise in evaluating your responses and will consider only complete answers as right answers. If you give me a partial answer, I will ask for additional details or clarification before moving to the next question.
        - If you are unable to answer correctly after 3 attempts, I will provide the correct answer along with directions to review specific lesson content in more detail.
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True) 

    # Creating columns to center the button
    col1, col2, col3 = st.columns([5, 3, 5])  # Middle column is wider

    with col2:
        # Centering the button in the middle column
        if st.button("Start Assessment", key="start_assessment", on_click=start_assessment):
            pass  # on_click will handle the transition


def start_assessment():
    st.session_state["page"] = "assessment"


# Function to generate speech from text
def generate_speech(text, voice="nova"):
    if not text or len(text.strip()) == 0:
        st.error("Cannot generate audio for an empty text.")
        return None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file_path = tmp_file.name
            response = openai.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=text,
            )
            response.stream_to_file(tmp_file_path)
            return tmp_file_path
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error generating speech: {str(e)}")
        return None

    
# Function to display progress in the sidebar
def display_sidebar_progress():
    st.sidebar.markdown(
    """
    <hr style="margin: 0px 0 20px 0; border: 1px solid #ddd;" />
    """,
    unsafe_allow_html=True,
    )
    st.sidebar.image("logo-wf.png", use_container_width=True)

    st.sidebar.markdown(
    """
    <hr style="margin: 5px 0 15px 0; border: 1px solid #ddd;" />
    """,
    unsafe_allow_html=True,
    )
    st.sidebar.image("logo-tarbiyah.png", use_container_width=True)
    st.sidebar.markdown(
    """
    <hr style="margin: 0px 0 60px 0; border: 1px solid #ddd;" />
    """,
    unsafe_allow_html=True,
    )

    progress = st.session_state.get("progress", {"correct_answers": 0, "attempts_per_question": {}})
    st.sidebar.header("⏳ Your Progress", divider='blue')
    st.sidebar.metric("Questions Answered Correctly:", progress["correct_answers"])

    # Settings
    st.sidebar.header("⚙️ Settings", divider='blue')

    # Text Size
    with st.sidebar.expander("🗚 Text Size", expanded=True):
        text_size = st.selectbox("Select text size:", ["Small", "Medium", "Large", "Extra Large"], index=1)
        st.session_state["text_size"] = text_size  # Store selected text size in session state

    # TTS settings
    with st.sidebar.expander("🔊 Text-to-Speech", expanded=True):
        enable_tts = st.sidebar.toggle("Enable Text-to-Speech", value=False, key="enable_tts")
            

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
    
    
    st.sidebar.markdown(
    """
    <hr style="margin: 60px 0 10px 0; border: 1px solid #ddd;" />
    """,
    unsafe_allow_html=True,
)
    st.sidebar.button("Restart Quiz", on_click=restart_quiz)


# Function to simulate typing effect
def simulate_typing_with_moving_lips(text, delay=0.01, batch_size=20, lip_sync_interval=4, container=None):
    if container is None:
        container = st.empty()
        
    displayed_text = ""
    open_mouth = True
    lip_sync_counter = 0

    for i in range(0, len(text), batch_size):
        displayed_text += text[i:i+batch_size]

        if lip_sync_counter % lip_sync_interval == 0:
            open_mouth = not open_mouth

        avatar_img = open_avatar_base64 if open_mouth else closed_avatar_base64

        container.markdown(
            f"""
            <div class="assistant-message">
                <img src="data:image/png;base64,{avatar_img}" class="assistant-avatar" alt="Assistant Avatar">
                <div class="message-bubble assistant-bubble">{displayed_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        lip_sync_counter += 1
        time.sleep(delay)

    # Ensure mouth is closed after speaking
    container.markdown(
        f"""
        <div class="assistant-message">
            <img src="data:image/png;base64,{closed_avatar_base64}" class="assistant-avatar" alt="Assistant Avatar">
            <div class="message-bubble assistant-bubble">{displayed_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_avatar_with_audio_and_typing(audio_path, text, duration):

     # Get the selected font size from session state
    font_size_map = {
        "Small": "14px",
        "Medium": "16px",
        "Large": "18px",
        "Extra Large": "20px"
    }
    selected_font_size = font_size_map[st.session_state.get("text_size", "Medium")]  # Default to "Medium" if not set

    # Check if the voice-over was successfully generated
    if audio_path:
        # Convert audio to Base64
        with open(audio_path, "rb") as audio_file:
            audio_base64 = base64.b64encode(audio_file.read()).decode()

    # HTML and JavaScript for syncing
    html_code = f"""
    <div id="chat-container" style="display: flex; flex-direction: column; align-items: flex-start; font-family: 'serif', sans-serif; line-height: 1.5; color: #333333; font-size: {selected_font_size};">
        <div style="display: flex; align-items: center; width: 100%; margin-bottom: 20px;">
            <img id="avatar-img" src="data:image/png;base64,{closed_avatar_base64}" style="width: 250px; height: 250px; margin: 5px; border-radius: 10%;">
            <div id="text-box" style="background-color: #ADE8F4; color: black; padding: 15px; border-radius: 15px; max-width: 70%; display: flex; justify-content: flex-start; align-items: center; margin: 10px 0; font-size: {selected_font_size};">
                {text}
            </div>
        </div>
        <audio id="audio-player" controls style="width: 100%; max-width: 600px; margin-bottom: 10px;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        <button id="play-animation" style="padding: 10px 20px; background-color: #f63366; color: white; border: none; border-radius: 15px; cursor: pointer; font-size: {selected_font_size};">
            Play voice-over
        </button>
    </div>
    <script>
        document.addEventListener("DOMContentLoaded", () => {{
            const playButton = document.getElementById("play-animation");
            const avatarImg = document.getElementById("avatar-img");
            const audio = document.getElementById("audio-player");
            let openMouth = true;

            playButton.addEventListener("click", () => {{
                audio.currentTime = 0;
                audio.play();

                // Lips animation during audio playback
                const lipsSyncInterval = setInterval(() => {{
                    if (audio.paused || audio.ended) {{
                        clearInterval(lipsSyncInterval);
                        avatarImg.src = "data:image/png;base64,{closed_avatar_base64}";
                    }} else {{
                        avatarImg.src = openMouth
                            ? "data:image/png;base64,{open_avatar_base64}"
                            : "data:image/png;base64,{closed_avatar_base64}";
                        openMouth = !openMouth;
                    }}
                }}, 200);  // Toggle every 200ms
            }});
        }});
    </script>
    """
    # Inject the HTML into Streamlit
    st.components.v1.html(html_code, height=500)


# Function to handle proceeding to the next question
def proceed_to_next_question():
    # Increment the question index
    st.session_state["current_question_index"] += 1
    st.session_state["attempts"] = 0

    # Clear chat history for the new question
    st.session_state["chat_history"] = []

    # Reset any other session variables for the new question if needed
    st.session_state["speech_file"] = None
    st.session_state["feedback"] = None



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


# <img src="data:image/png;base64,{user_avatar_base64}" class="user-avatar" alt="User Avatar"> # user avatar
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
                    
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="assistant-message">
                    <img src="data:image/png;base64,{closed_avatar_base64}" class="assistant-avatar" alt="Assistant Avatar">
                    <div class="message-bubble assistant-bubble">{content}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )


# Main function to display the quiz
def display_quiz():
    display_sidebar_progress()  # Show progress in sidebar

    # Get the current question index and data
    current_index = st.session_state["current_question_index"]
    current_question = questions[current_index]

    # Ensure session state variables exist
    if current_index not in st.session_state["chat_histories"]:
        st.session_state["chat_histories"][current_index] = []
    if current_index not in st.session_state["button_states"]:
        st.session_state["button_states"][current_index] = False
    if current_index not in st.session_state["attempts_per_question"]:
        st.session_state["attempts_per_question"][current_index] = 0
    if "feedback" not in st.session_state:
        st.session_state["feedback"] = None
    if "speech_file" not in st.session_state:
        st.session_state["speech_file"] = None
    if "feedback_rendered" not in st.session_state:
        st.session_state["feedback_rendered"] = {}  # Track rendering for all questions
    if current_index not in st.session_state["feedback_rendered"]:
        st.session_state["feedback_rendered"][current_index] = False

    # Restore the button state and attempts for the current question
    st.session_state["show_proceed_button"] = st.session_state["button_states"][current_index]
    st.session_state["attempts"] = st.session_state["attempts_per_question"][current_index]

    # # Update the most recent question index if we're on the latest question
    # if current_index >= st.session_state["most_recent_question_index"]:
    #     st.session_state["most_recent_question_index"] = current_index

    st.title("MCE Know-Bot")

    col1, col2 = st.columns([1,4])

    with col1:
        st.image("avatar.png", use_container_width=True)

    with col2:
        # Display question
        st.write(f"### {current_question['scenario_number']}")
        st.write(f"*{current_question['scenario']}*")
        st.write("")
        st.write(f"**{current_question['question']}**")

    # Display chat history using display_chat_history() (initial full history display)
    display_chat_history(st.session_state["chat_histories"][current_index])

    # Check if attempts have reached the maximum or if answer is right answer
    if st.session_state["attempts"] >= 3 or st.session_state["question_completed"].get(current_index, False):
        st.error("This question is complete. Please proceed to the next question or review the lesson.", icon="❗")
        st.session_state["show_proceed_button"] = True  # Set the proceed button to display
    else:
        # Show the chat input only if the question is not complete
        if st.session_state["attempts"] < 3 and not st.session_state["question_completed"].get(current_index, False):
            # Display chat input if answer is not right answer and attempts are below 3
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

                # # Create a placeholder for the assistant's response
                # assistant_response_placeholder = st.empty()

                # Display a spinner while processing the answer
                with st.spinner('💭 Checking your answer...'):
                    time.sleep(0.5)  # Simulate delay for demonstration
                    feedback = generate_feedback(current_question, user_input, st.session_state["attempts"])
                    st.markdown(feedback, unsafe_allow_html=True)

                # Append feedback to chat history
                st.session_state["chat_histories"][current_index].append({"role": "assistant", "content": feedback})

                # # Initial rendering: Show the typing animation (avatar + text response)
                # with assistant_response_placeholder.container():
                #     simulate_typing_with_moving_lips(feedback)

                # Save feedback in session state
                st.session_state["feedback"] = feedback
                st.session_state["speech_file"] = None  # Reset speech file for new attempt
                st.session_state["feedback_rendered"][current_index] = False  # Reset rendering flag

                # Check if TTS is enabled
                if st.session_state.get("enable_tts", False):  
                    with st.spinner('🔊 Generating voice-over...'):
                        # Generate speech for the feedback
                        speech_file = generate_speech(feedback, voice=st.session_state.get("tts_voice", "nova"))

                    if speech_file and os.path.exists(speech_file):
                        st.session_state["speech_file"] = speech_file
                    else:
                        st.error("Audio file is missing or not accessible.")

                    # if speech_file and os.path.exists(speech_file):
                    #     try:
                    #         # Get audio duration using mutagen
                    #         try:
                    #             audio = MP3(speech_file)
                    #             audio_duration = audio.info.length
                    #         except Exception:
                    #             info = mediainfo(speech_file)
                    #             audio_duration = float(info['duration'])

                    #         # Append audio and avatar below the response bubble
                    #         with assistant_response_placeholder.container():
                    #             display_avatar_with_audio_and_typing(speech_file, feedback, audio_duration)

                    #     except Exception as e:
                    #         st.error(f"Audio error: {e}")
                    # else:
                    #     st.error("Audio file is missing or not accessible.")

                else:
                    simulate_typing_with_moving_lips(feedback)

                # Process attempts and correct answers
                if "right answer" in feedback:
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

    # Render the "Play Audio with Animation" button and feedback
    if st.session_state.get("feedback") and st.session_state.get("speech_file"):
        # Only render once per question
        if not st.session_state["feedback_rendered"][current_index]:
            speech_file = st.session_state["speech_file"]
            feedback = st.session_state["feedback"]
            display_avatar_with_audio_and_typing(speech_file, feedback, 0)
            st.session_state["feedback_rendered"][current_index] = True

    # Display success message
    if st.session_state["attempts"] == 1 and st.session_state.get("feedback") and "right answer" in st.session_state["feedback"]:
        st.success("Great job! You got it right on the first try! 🌟")


    # Display "Proceed to the next question" button if answer is correct or 3 attempts reached
    if st.session_state.get("show_proceed_button", False):
        # Check if we are on the last question
        if current_index == len(questions) - 1:
            # Only display "Complete Quiz" button if answer is right answer or attempts reached 3
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
