import streamlit as st
import nltk
import re
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="StudentGPT",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)


# =========================================================
# NLTK SETUP
# =========================================================

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# =========================================================
# FAQ DATASET
# =========================================================

faqs = [
    {
        "question": "What are the college working hours?",
        "answer": "The college working hours are from 9 AM to 5 PM."
    },
    {
        "question": "How can I apply for admission?",
        "answer": "You can apply for admission through the college admission portal."
    },
    {
        "question": "What documents are required for admission?",
        "answer": "You need academic certificates, identity proof, photographs, and other required documents."
    },
    {
        "question": "How can I pay my college fees?",
        "answer": "You can pay your college fees through the college's online payment system."
    },
    {
        "question": "How can I contact the college?",
        "answer": "You can contact the college through the official college office or contact number."
    },
    {
        "question": "Where is the admission office?",
        "answer": "The admission office is located in the main administrative building."
    },
    {
        "question": "How can I get my student ID card?",
        "answer": "You can collect your student ID card from the student administration office."
    },
    {
        "question": "How can I check my exam timetable?",
        "answer": "The exam timetable is available through the college student portal."
    },
    {
        "question": "When are the exams conducted?",
        "answer": "The examination dates are announced by the college examination department."
    },
    {
        "question": "How can I get my result?",
        "answer": "You can check your result through the college student portal."
    }
]


# =========================================================
# NLP PREPROCESSING
# =========================================================

def preprocess_text(text):

    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove empty tokens
    tokens = [word for word in tokens if word.strip()]

    return " ".join(tokens)


# =========================================================
# PREPROCESS FAQ QUESTIONS
# =========================================================

faq_questions = [
    preprocess_text(faq["question"])
    for faq in faqs
]


# =========================================================
# TF-IDF
# =========================================================

vectorizer = TfidfVectorizer(
    stop_words="english"
)

faq_vectors = vectorizer.fit_transform(
    faq_questions
)


# =========================================================
# FIND BEST ANSWER
# =========================================================

def find_best_answer(user_question):

    processed_question = preprocess_text(
        user_question
    )

    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[
        0
    ][best_match_index]

    # Minimum similarity required
    if best_score < 0.20:
        return None

    return faqs[
        best_match_index
    ]["answer"]


# =========================================================
# CHATBOT RESPONSE
# =========================================================

def chatbot_response(user_message):

    message = user_message.lower().strip()

    # -----------------------------
    # GREETINGS
    # -----------------------------

    greetings = [
        "hi",
        "hello",
        "hey",
        "hii",
        "hiii",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    if message in greetings:

        return (
            "👋 Hello! Welcome to StudentGPT. "
            "How can I help you today?"
        )


    # -----------------------------
    # THANKS
    # -----------------------------

    thanks = [
        "thanks",
        "thank you",
        "thankyou",
        "thx",
        "thanks a lot"
    ]

    if message in thanks:

        return (
            "😊 You're welcome! "
            "I'm happy to help."
        )


    # -----------------------------
    # GOODBYE
    # -----------------------------

    goodbye = [
        "bye",
        "goodbye",
        "see you",
        "see you later"
    ]

    if message in goodbye:

        return (
            "👋 Goodbye! Have a great day!"
        )


    # -----------------------------
    # HELP
    # -----------------------------

    help_messages = [
        "help",
        "what can you do",
        "what can you help me with",
        "what can i ask"
    ]

    if message in help_messages:

        return """I can help you with college-related questions such as:

🎓 Admission

📄 Required documents

💳 College fees

🪪 Student ID card

📝 Exam timetable

📊 Results

🏢 Admission office

📞 Contact information

⏰ College working hours"""


    # -----------------------------
    # FAQ MATCHING
    # -----------------------------

    answer = find_best_answer(
        user_message
    )

    if answer:
        return answer


    # -----------------------------
    # UNKNOWN QUESTION
    # -----------------------------

    return (
        "🤔 I couldn't find a relevant answer "
        "in my FAQ database.\n\n"
        "Try asking about admission, fees, "
        "exams, results, student ID cards, "
        "working hours, or college contact information."
    )


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =========================================
       MAIN BACKGROUND
       ========================================= */

    .stApp {

        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(124, 58, 237, 0.18),
                transparent 30%
            ),

            radial-gradient(
                circle at 90% 80%,
                rgba(59, 130, 246, 0.14),
                transparent 30%
            ),

            linear-gradient(
                135deg,
                #070B14,
                #0B1020,
                #080D18
            );

        color: #F5F7FA;

    }


    /* =========================================
       TITLE
       ========================================= */

    .main-title {

        text-align: center;

        font-size: 38px;

        font-weight: 800;

        color: #FFFFFF;

        letter-spacing: 1px;

        margin-top: 10px;

        margin-bottom: 4px;

        text-shadow:
            0 0 15px rgba(139, 92, 246, 0.5);

        animation: titleGlow 3s infinite alternate;

    }


    @keyframes titleGlow {

        from {
            text-shadow:
                0 0 8px rgba(139, 92, 246, 0.3);
        }

        to {
            text-shadow:
                0 0 25px rgba(139, 92, 246, 0.8);
        }

    }


    /* =========================================
       SUBTITLE
       ========================================= */

    .subtitle {

        text-align: center;

        color: #A7B0C0;

        font-size: 14px;

        margin-bottom: 8px;

    }


    /* =========================================
       ONLINE STATUS
       ========================================= */

    .status {

        text-align: center;

        color: #A7F3D0;

        font-size: 13px;

        margin-bottom: 25px;

    }


    .status-dot {

        display: inline-block;

        width: 9px;

        height: 9px;

        background: #22C55E;

        border-radius: 50%;

        margin-right: 6px;

        box-shadow:
            0 0 10px #22C55E;

        animation: pulse 1.5s infinite;

    }


    @keyframes pulse {

        0% {
            box-shadow:
                0 0 4px #22C55E;
        }

        50% {
            box-shadow:
                0 0 16px #22C55E;
        }

        100% {
            box-shadow:
                0 0 4px #22C55E;
        }

    }


    /* =========================================
       STREAMLIT CHAT MESSAGE
       ========================================= */

    [data-testid="stChatMessage"] {

        background:
            rgba(32, 41, 56, 0.65);

        border:
            1px solid
            rgba(255,255,255,0.06);

        border-radius: 18px;

        padding: 10px 14px;

        margin-bottom: 12px;

        box-shadow:
            0 8px 25px
            rgba(0,0,0,0.18);

        animation:
            messageIn 0.35s ease;

        transition:
            transform 0.2s ease,
            border 0.2s ease;

    }


    [data-testid="stChatMessage"]:hover {

        transform:
            translateY(-2px);

        border:
            1px solid
            rgba(139,92,246,0.25);

    }


    @keyframes messageIn {

        from {

            opacity: 0;

            transform:
                translateY(12px);

        }

        to {

            opacity: 1;

            transform:
                translateY(0);

        }

    }


    /* =========================================
       CHAT TEXT
       ========================================= */

    [data-testid="stChatMessage"] p {

        color: #F5F7FA !important;

        font-size: 15px;

        line-height: 1.7;

    }


    [data-testid="stChatMessage"] strong {

        color: #A78BFA !important;

    }


    /* =========================================
       SIDEBAR
       ========================================= */

    section[data-testid="stSidebar"] {

        background:
            linear-gradient(
                180deg,
                #111827,
                #080D18
            );

        border-right:
            1px solid
            rgba(255,255,255,0.08);

    }


    section[data-testid="stSidebar"] * {

        color: #E5E7EB;

    }


    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {

        color: #FFFFFF;

    }


    /* =========================================
       SIDEBAR CARDS
       ========================================= */

    .sidebar-card {

        background:
            rgba(255,255,255,0.04);

        border:
            1px solid
            rgba(255,255,255,0.06);

        padding: 12px;

        border-radius: 12px;

        margin-bottom: 10px;

        transition:
            all 0.2s ease;

    }


    .sidebar-card:hover {

        background:
            rgba(124,58,237,0.12);

        border:
            1px solid
            rgba(124,58,237,0.3);

        transform:
            translateX(4px);

    }


    /* =========================================
       CHAT INPUT
       ========================================= */

    div[data-testid="stChatInput"] {

        border-radius: 18px;

    }


    div[data-testid="stChatInput"] textarea {

        background-color:
            #151C2A !important;

        color:
            #FFFFFF !important;

        border:
            1px solid
            #30394D !important;

        border-radius:
            16px !important;

        font-size:
            15px !important;

    }


    div[data-testid="stChatInput"] textarea::placeholder {

        color:
            #7F8AA3 !important;

    }


    div[data-testid="stChatInput"] textarea:focus {

        border:
            1px solid
            #8B5CF6 !important;

        box-shadow:
            0 0 0 2px
            rgba(139,92,246,0.15)
            !important;

    }


    /* =========================================
       BUTTON
       ========================================= */

    .stButton > button {

        background:
            linear-gradient(
                135deg,
                #7C3AED,
                #6366F1
            );

        color:
            #FFFFFF;

        border:
            none;

        border-radius:
            12px;

        font-weight:
            600;

        transition:
            all 0.25s ease;

    }


    .stButton > button:hover {

        transform:
            translateY(-2px);

        box-shadow:
            0 8px 25px
            rgba(124,58,237,0.40);

    }


    /* =========================================
       DIVIDER
       ========================================= */

    hr {

        border-color:
            rgba(255,255,255,0.08);

    }


    /* =========================================
       SCROLLBAR
       ========================================= */

    ::-webkit-scrollbar {

        width: 7px;

    }


    ::-webkit-scrollbar-track {

        background:
            #080D18;

    }


    ::-webkit-scrollbar-thumb {

        background:
            #374151;

        border-radius:
            10px;

    }


    ::-webkit-scrollbar-thumb:hover {

        background:
            #7C3AED;

    }


    /* =========================================
       MOBILE
       ========================================= */

    @media (max-width: 700px) {

        .main-title {

            font-size: 30px;

        }

    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🎓 StudentGPT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your intelligent college FAQ assistant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="status">
        <span class="status-dot"></span>
        StudentGPT is online
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 🎓 StudentGPT")

    st.markdown("### Your College Assistant")

    st.write(
        "Ask questions about college admission, "
        "fees, exams, results and student services."
    )

    st.divider()

    st.subheader("💡 Try asking")

    st.markdown(
        """
        <div class="sidebar-card">
        🎓 How can I apply for admission?
        </div>

        <div class="sidebar-card">
        📄 What documents are required?
        </div>

        <div class="sidebar-card">
        💳 How can I pay my college fees?
        </div>

        <div class="sidebar-card">
        📝 When are the exams conducted?
        </div>

        <div class="sidebar-card">
        📊 How can I get my result?
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.caption(
        "🧠 NLP + TF-IDF + Cosine Similarity"
    )

    st.caption(
        "Built with Python & Streamlit"
    )

    st.divider()

    if st.button(
        "🗑️ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = [
            {
                "role": "assistant",
                "content":
                "👋 Hello! Welcome to StudentGPT. "
                "How can I help you today?"
            }
        ]

        st.rerun()


# =========================================================
# INITIAL MESSAGE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = [

        {
            "role": "assistant",
            "content":
            "👋 Hello! Welcome to StudentGPT. "
            "How can I help you today?"
        }

    ]


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message(
            "user",
            avatar="👤"
        ):

            st.markdown(
                message["content"]
            )

    else:

        with st.chat_message(
            "assistant",
            avatar="🤖"
        ):

            st.markdown(
                message["content"]
            )


# =========================================================
# CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Message StudentGPT..."
)


# =========================================================
# PROCESS USER MESSAGE
# =========================================================

if user_input:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Generate answer
    response = chatbot_response(
        user_input
    )

    # Add bot response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    # Refresh
    st.rerun()