from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================
# AI MODEL CONFIGURATION
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "resnet50_radimagenet_layer3_layer4_best.pth"
)

MODEL_URL = (
    "https://github.com/nainidhadda/"
    "Bone-Tumour-Detection-Using-Multi-Source-Dataset-Integration/"
    "releases/download/v1.0-model/"
    "resnet50_radimagenet_layer3_layer4_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_INPUT_SIZE = 224

MODEL_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# LOAD TRAINED BONE TUMOR MODEL
# ============================================================

def ensure_model_exists():
    """
    Download the trained model from the GitHub Release
    if it is not already available locally.
    """

    if MODEL_PATH.exists():
        return

    import urllib.request

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with st.spinner(
        "Loading AI model for the first time..."
    ):
        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )


ensure_model_exists()


@st.cache_resource
def load_ai_model():
    """
    Load the trained ResNet50 model.

    The checkpoint contains the complete trained
    ResNet50 + RadImageNet model state.
    """

    model = models.resnet50(
        weights=None
    )

    # Binary classification:
    # 0 = Normal
    # 1 = Tumor
    model.fc = nn.Linear(
        in_features=2048,
        out_features=2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    return model


AI_MODEL = load_ai_model()


# ============================================================
# AI INFERENCE
# ============================================================

def ai_analysis(image):
    """
    Run the trained ResNet50 model on an uploaded X-ray.
    """

    image = image.convert("RGB")

    image_tensor = MODEL_TRANSFORM(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():
        outputs = AI_MODEL(image_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()

    if predicted_class == 1:
        prediction = (
            "Tumor detected — "
            "professional review required"
        )
    else:
        prediction = (
            "No tumor detected — "
            "professional review required"
        )

    return {
        "ai_prediction": prediction,
        "ai_confidence": (
            f"{confidence * 100:.2f}% model confidence"
        ),
        "ai_model": "ResNet50 + RadImageNet",
    }


# ============================================================
# APPLICATION CONTENT
# ============================================================

DISCLAIMER = (
    "This application is an AI-assisted research prototype. "
    "AI output is intended to support professional review "
    "and is not a medical diagnosis."
)


# Simple demo accounts for the college prototype.
DEMO_ACCOUNTS = {
    "user@bonewise.demo": {
        "password": "demo",
        "role": "User",
        "name": "Alex Rivera",
    },
    "doctor@bonewise.demo": {
        "password": "demo",
        "role": "Doctor",
        "name": "Dr. N. Rao",
    },
}


ANATOMY_FACTS = [
    (
        "Femur",
        "The femur is the longest and one of the strongest "
        "bones in the human body, running from the hip to the knee."
    ),
    (
        "Tibia",
        "The tibia, or shin bone, bears most of the body's "
        "weight and connects the knee to the ankle."
    ),
    (
        "Fibula",
        "The fibula runs alongside the tibia and mainly "
        "assists with muscle attachment and ankle stability."
    ),
    (
        "Humerus",
        "The humerus is the long bone of the upper arm, "
        "connecting the shoulder to the elbow."
    ),
    (
        "Spine",
        "The spine is made up of vertebrae that protect "
        "the spinal cord and support posture."
    ),
    (
        "Pelvis",
        "The pelvis connects the spine to the lower limbs "
        "and protects several internal organs."
    ),
    (
        "Joints",
        "Joints are points where two or more bones meet, "
        "allowing movement."
    ),
    (
        "Bone marrow",
        "Bone marrow, found inside many bones, is where "
        "the body produces blood cells."
    ),
]


BONE_HEALTH_FACTS = [
    (
        "Calcium",
        "Calcium is a key mineral for building and maintaining "
        "bone tissue throughout life."
    ),
    (
        "Vitamin D",
        "Vitamin D helps the body absorb calcium effectively."
    ),
    (
        "Weight-bearing activity",
        "Activities such as walking and strength training "
        "are generally associated with stronger bones."
    ),
    (
        "Balanced nutrition",
        "A varied, balanced diet supports overall skeletal health."
    ),
    (
        "Bone density",
        "Bone density naturally changes with age and is one "
        "factor clinicians may consider during evaluation."
    ),
    (
        "Healthy habits",
        "Avoiding smoking and limiting excessive alcohol intake "
        "support long-term bone health."
    ),
    (
        "Growth and aging",
        "Bone tissue is continuously broken down and rebuilt "
        "throughout life."
    ),
    (
        "General information only",
        "These notes are educational information and are not "
        "personalized medical advice."
    ),
]


# ============================================================
# PAGE DESIGN
# ============================================================

def configure_page():

    st.set_page_config(
        page_title="Bonewise | X-ray screening prototype",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>

        @import url(
            'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700'
            '&family=Newsreader:opsz,wght@6..72,500;6..72,600'
            '&display=swap'
        );

        :root {
            --ink:#292a26;
            --muted:#76766e;
            --line:#dedbd0;
            --ivory:#f5f2ea;
            --paper:#fffdf8;
            --sage:#66745d;
            --sage-light:#e6ebe1;
        }

        .stApp {
            background:var(--ivory);
            color:var(--ink);
        }

        [data-testid="stHeader"] {
            background:transparent;
        }

        [data-testid="stSidebar"] {
            background:#ebe8de;
            border-right:1px solid var(--line);
        }

        [data-testid="stSidebar"] * {
            color:var(--ink);
        }

        h1,
        h2,
        h3 {
            color:var(--ink);
            font-family:'Newsreader', Georgia, serif;
            letter-spacing:-.02em;
        }

        p,
        label,
        .stMarkdown,
        .stButton,
        .stTextInput,
        .stTextArea,
        .stSelectbox {
            font-family:'DM Sans', sans-serif;
        }

        .eyebrow {
            color:var(--sage);
            font:700 11px 'DM Sans',sans-serif;
            letter-spacing:.14em;
            text-transform:uppercase;
        }

        .lede {
            color:var(--muted);
            font-size:1.05rem;
            max-width:680px;
            line-height:1.6;
        }

        .paper {
            background:var(--paper);
            border:1px solid var(--line);
            padding:22px 24px;
            margin:10px 0 18px;
        }

        .metric {
            font:600 2rem 'Newsreader',serif;
            color:var(--ink);
        }

        .muted {
            color:var(--muted);
            font-size:.9rem;
        }

        .status {
            color:var(--sage);
            font-weight:600;
            font-size:.9rem;
        }

        .timeline {
            border-left:2px solid var(--line);
            margin:14px 0 4px 8px;
            padding-left:22px;
        }

        .timeline-row {
            position:relative;
            padding:4px 0 17px;
            color:var(--muted);
        }

        .timeline-row:before {
            content:'';
            position:absolute;
            left:-31px;
            top:5px;
            width:13px;
            height:13px;
            border-radius:50%;
            background:var(--line);
            border:3px solid var(--ivory);
        }

        .timeline-row.done {
            color:var(--ink);
            font-weight:600;
        }

        .timeline-row.done:before {
            background:var(--sage);
        }

        .disclaimer {
            border-left:3px solid #a6b29d;
            padding:10px 14px;
            color:var(--muted);
            font-size:.82rem;
            line-height:1.5;
            margin:20px 0;
        }

        .case-id {
            color:var(--sage);
            font-weight:700;
            letter-spacing:.05em;
        }

        .ai-panel {
            background:var(--paper);
            border:1px solid var(--line);
            padding:20px 22px;
            margin:12px 0 20px;
        }

        .ai-label {
            color:var(--muted);
            font-size:.76rem;
            text-transform:uppercase;
            letter-spacing:.08em;
            margin-bottom:5px;
        }

        .ai-result {
            color:var(--ink);
            font-family:'Newsreader', Georgia, serif;
            font-size:1.35rem;
            margin-bottom:14px;
        }

        .model-badge {
            display:inline-block;
            background:var(--sage-light);
            color:var(--sage);
            padding:5px 10px;
            font-size:.78rem;
            font-weight:600;
            margin-top:4px;
        }

        div.stButton > button {
            border:1px solid #b8b7ac;
            border-radius:2px;
            background:transparent;
            color:var(--ink);
            font-weight:600;
        }

        div.stButton > button:hover {
            border-color:var(--sage);
            color:var(--sage);
        }

        div.stButton > button[kind="primary"] {
            background:var(--sage);
            border-color:var(--sage);
            color:white;
        }

        [data-testid="stFileUploader"] {
            background:var(--paper);
            border:1px dashed #b9b5a8;
            padding:8px;
        }

        /* Login */

        .login-shell {
            display:flex;
            min-height:78vh;
        }

        .login-brand {
            flex:1.1;
            padding:48px 56px 0 4px;
        }

        .login-mark {
            font-family:'Newsreader',serif;
            font-size:2.6rem;
            letter-spacing:-.01em;
            margin-bottom:2px;
        }

        .login-tag {
            color:var(--muted);
            font-size:1.02rem;
            margin-bottom:28px;
        }

        .login-statement {
            max-width:400px;
            color:var(--ink);
            font-size:1.05rem;
            line-height:1.65;
            border-top:1px solid var(--line);
            padding-top:22px;
        }

        .login-demo-hint {
            margin-top:34px;
            color:var(--muted);
            font-size:.82rem;
            line-height:1.7;
            border-left:2px solid var(--line);
            padding-left:14px;
        }

        /* Statistics */

        .stat-row {
            display:flex;
            border-top:1px solid var(--line);
            border-bottom:1px solid var(--line);
            margin:18px 0 26px;
        }

        .stat-cell {
            flex:1;
            padding:16px 22px;
            border-right:1px solid var(--line);
        }

        .stat-cell:last-child {
            border-right:none;
        }

        .stat-label {
            color:var(--muted);
            font-size:.82rem;
            text-transform:uppercase;
            letter-spacing:.08em;
            margin-bottom:6px;
        }

        /* Flashcards */

        .flashcard {
            background:var(--paper);
            border:1px solid var(--line);
            padding:34px 30px;
            min-height:150px;
        }

        .flashcard-term {
            font-family:'Newsreader',serif;
            font-size:1.35rem;
            margin-bottom:10px;
        }

        .flashcard-body {
            color:var(--muted);
            line-height:1.6;
        }

        .flash-index {
            color:var(--muted);
            font-size:.82rem;
            text-align:center;
            margin-top:10px;
            letter-spacing:.05em;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DEMO IMAGE
# ============================================================

def demo_image_path():

    image_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "combined"
        / "test"
        / "images"
    )

    images = sorted(
        image_dir.glob("*.png")
    )

    return str(images[0]) if images else None


# ============================================================
# DEMO CASES
# ============================================================

def make_demo_cases():

    mock_result = {
        "ai_prediction": (
            "Demo case — upload an X-ray "
            "to run the trained model"
        ),
        "ai_confidence": (
            "Model inference is available "
            "for uploaded X-rays"
        ),
        "ai_model": "ResNet50 + RadImageNet",
    }

    return [

        {
            "id": "BW-2401",
            "submitted": "21 Sep 2026",
            "status": "Pending doctor review",
            "image": demo_image_path(),
            "image_bytes": None,
            **mock_result,
            "assessment": None,
            "comments": None,
            "review_date": None,
        },

        {
            "id": "BW-2398",
            "submitted": "19 Sep 2026",
            "status": "Doctor review completed",
            "image": None,
            "image_bytes": None,
            **mock_result,
            "assessment": (
                "Further clinical correlation recommended"
            ),
            "comments": (
                "This is a demonstration response. "
                "Please discuss any concerns with a "
                "qualified professional."
            ),
            "review_date": "20 Sep 2026",
        },

    ]


# ============================================================
# SESSION STATE
# ============================================================

def initialize_state():

    if "cases" not in st.session_state:
        st.session_state.cases = make_demo_cases()

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if "role" not in st.session_state:
        st.session_state.role = None

    if "user_name" not in st.session_state:
        st.session_state.user_name = None

    if "selected_case" not in st.session_state:
        st.session_state.selected_case = None

    if "anatomy_index" not in st.session_state:
        st.session_state.anatomy_index = 0

    if "health_index" not in st.session_state:
        st.session_state.health_index = 0


# ============================================================
# CASE TIMELINE
# ============================================================

def case_timeline(case):

    reviewed = (
        case["status"]
        == "Doctor review completed"
    )

    rows = [
        ("X-ray submitted", True),
        ("AI-assisted analysis", True),
        ("Doctor review", reviewed),
        ("Doctor response", reviewed),
    ]

    timeline_html = '<div class="timeline">'

    for label, done in rows:

        state_class = (
            "done"
            if done
            else ""
        )

        status_text = (
            ""
            if done
            else '<br><span class="muted">Pending</span>'
        )

        icon = "✓" if done else "○"

        timeline_html += (
            f'<div class="timeline-row {state_class}">'
            f'{icon} &nbsp;{label}'
            f'{status_text}'
            f'</div>'
        )

    timeline_html += "</div>"

    st.markdown(
        timeline_html,
        unsafe_allow_html=True
    )


# ============================================================
# BRAND
# ============================================================

def render_brand():

    st.markdown(
        '<div class="eyebrow">'
        'Bonewise / research prototype'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "<h1 style='margin-top:4px'>"
        "A calmer way to begin a review."
        "</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="lede">'
        'AI-assisted X-ray screening that keeps '
        'the doctor at the centre of every decision.'
        '</p>',
        unsafe_allow_html=True,
    )


# ============================================================
# LOGIN
# ============================================================

def render_login():

    left, right = st.columns(
        [1.1, 1],
        gap="large"
    )

    with left:

        st.markdown(
            """
            <div class="login-brand">

                <div class="login-mark">
                    Bonewise
                </div>

                <div class="login-tag">
                    AI-assisted X-ray review
                </div>

                <div class="login-statement">
                    Supporting clinician-led review
                    through machine learning.
                    Every case is screened, then reviewed
                    and decided by a doctor —
                    the AI never has the final word.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            '<div class="eyebrow" '
            'style="margin-top:12px;">'
            'Sign in'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            "<h3 style='margin-top:6px;'>"
            "Welcome back"
            "</h3>",
            unsafe_allow_html=True
        )

        email = st.text_input(
            "Email or username",
            placeholder="you@bonewise.demo"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="••••••••"
        )

        if st.button(
            "Sign in",
            type="primary"
        ):

            account = DEMO_ACCOUNTS.get(
                email.strip().lower()
            )

            if (
                account
                and account["password"] == password
            ):

                st.session_state.authenticated = True
                st.session_state.role = account["role"]
                st.session_state.user_name = account["name"]
                st.session_state.selected_case = None

                st.rerun()

            else:

                st.error(
                    "Those credentials weren't recognised. "
                    "Try one of the demo accounts below."
                )

        st.markdown(
            """
            <div class="login-demo-hint">
                Research prototype — demo accounts<br>
                user@bonewise.demo &nbsp;/&nbsp; demo<br>
                doctor@bonewise.demo &nbsp;/&nbsp; demo
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# FLASHCARDS
# ============================================================

def render_flashcard_deck(
    title,
    deck,
    state_key
):

    st.markdown(
        f"#### {title}"
    )

    index = st.session_state[state_key]

    term, body = deck[index]

    st.markdown(
        f'<div class="flashcard">'
        f'<div class="flashcard-term">'
        f'{term}'
        f'</div>'
        f'<div class="flashcard-body">'
        f'{body}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="flash-index">'
        f'{index + 1} / {len(deck)}'
        f'</div>',
        unsafe_allow_html=True
    )

    nav_left, nav_right = st.columns(2)

    with nav_left:

        if st.button(
            "← Previous",
            key=f"{state_key}-prev",
            disabled=index == 0
        ):

            st.session_state[state_key] -= 1
            st.rerun()

    with nav_right:

        if st.button(
            "Next →",
            key=f"{state_key}-next",
            disabled=index == len(deck) - 1
        ):

            st.session_state[state_key] += 1
            st.rerun()


# ============================================================
# HEALTH EDUCATION
# ============================================================

def render_health():

    st.markdown(
        "## Learn while you wait"
    )

    st.caption(
        "Short, general educational notes. "
        "These are not personalized medical advice."
    )

    deck_left, deck_right = st.columns(
        2,
        gap="large"
    )

    with deck_left:

        render_flashcard_deck(
            "Anatomy facts",
            ANATOMY_FACTS,
            "anatomy_index"
        )

    with deck_right:

        render_flashcard_deck(
            "Bone health",
            BONE_HEALTH_FACTS,
            "health_index"
        )


# ============================================================
# USER DASHBOARD
# ============================================================

def render_user():

    render_brand()

    st.markdown(
        '<div class="disclaimer">'
        '✦ &nbsp;'
        + DISCLAIMER
        + "</div>",
        unsafe_allow_html=True
    )

    first_name = (
        st.session_state.user_name.split()[0]
    )

    st.markdown(
        f"### Good to see you, {first_name}"
    )

    pending = next(
        (
            case
            for case in reversed(
                st.session_state.cases
            )
            if case["status"]
            != "Doctor review completed"
        ),
        None
    )

    if pending:

        st.markdown(
            '<div class="paper">'
            '<div class="eyebrow">'
            'Current case'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"### {pending['id']}"
        )

        st.markdown(
            f"<span class='muted'>"
            f"Submitted {pending['submitted']}"
            f"</span>",
            unsafe_allow_html=True
        )

        case_timeline(pending)

        if pending["assessment"]:

            st.markdown(
                "#### Doctor response"
            )

            st.markdown(
                f"**Assessment:** "
                f"{pending['assessment']}"
            )

            st.markdown(
                f"**Comments:** "
                f"{pending['comments']}"
            )

            st.caption(
                f"Reviewed {pending['review_date']}"
            )

        else:

            st.info(
                "Your case is waiting for "
                "professional review. You will "
                "see the doctor's response here."
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    else:

        st.info(
            "You do not have a case waiting for review."
        )

    # --------------------------------------------------------
    # X-RAY UPLOAD
    # --------------------------------------------------------

    st.markdown(
        "## Submit an X-ray"
    )

    st.caption(
        "PNG, JPG, and JPEG files are supported."
    )

    upload = st.file_uploader(
        "Choose an X-ray image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed"
    )

    if upload:

        st.image(
            upload,
            caption="Selected X-ray",
            width=420
        )

        if st.button(
            "Submit X-ray",
            type="primary"
        ):

            case_id = (
                f"BW-{datetime.now().strftime('%H%M%S')}"
            )

            uploaded_image = Image.open(
                upload
            )

            ai_result = ai_analysis(
                uploaded_image
            )

            st.session_state.cases.append(
                {
                    "id": case_id,
                    "submitted": datetime.now().strftime(
                        "%d %b %Y"
                    ),
                    "status": "Pending doctor review",
                    "image": None,
                    "image_bytes": upload.getvalue(),
                    **ai_result,
                    "assessment": None,
                    "comments": None,
                    "review_date": None,
                }
            )

            st.success(
                f"Your X-ray has been submitted. "
                f"Case ID: {case_id}"
            )

            st.rerun()

    # --------------------------------------------------------
    # PREVIOUS CASES
    # --------------------------------------------------------

    st.markdown(
        "## Previous cases"
    )

    for case in reversed(
        st.session_state.cases
    ):

        if (
            pending
            and case["id"] == pending["id"]
        ):
            continue

        st.markdown(
            f"""
            <div class='paper'>
                <span class='case-id'>
                    {case['id']}
                </span>
                &nbsp;
                <span class='muted'>
                    {case['submitted']}
                </span>
                <br>
                <span class='status'>
                    {case['status']}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    render_health()


# ============================================================
# DOCTOR DASHBOARD
# ============================================================

def render_doctor():

    render_brand()

    pending = [
        case
        for case in st.session_state.cases
        if case["status"]
        != "Doctor review completed"
    ]

    reviewed = [
        case
        for case in st.session_state.cases
        if case["status"]
        == "Doctor review completed"
    ]

    st.markdown(
        f"## Doctor workspace "
        f"&nbsp;·&nbsp; "
        f"{st.session_state.user_name}"
    )

    st.markdown(
        f"""
        <div class="stat-row">

            <div class="stat-cell">
                <div class="stat-label">
                    Waiting for review
                </div>
                <div class="metric">
                    {len(pending)}
                </div>
            </div>

            <div class="stat-cell">
                <div class="stat-label">
                    Reviewed cases
                </div>
                <div class="metric">
                    {len(reviewed)}
                </div>
            </div>

            <div class="stat-cell">
                <div class="stat-label">
                    Total cases
                </div>
                <div class="metric">
                    {len(st.session_state.cases)}
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "### Case list"
    )

    for case in st.session_state.cases:

        left, right = st.columns(
            [4, 1]
        )

        with left:

            st.markdown(
                f"**{case['id']}** "
                f"&nbsp; "
                f"<span class='muted'>"
                f"{case['submitted']}"
                f"</span><br>"
                f"<span class='status'>"
                f"{case['status']}"
                f"</span>",
                unsafe_allow_html=True,
            )

        with right:

            if st.button(
                "Open",
                key=f"open-{case['id']}"
            ):

                st.session_state.selected_case = (
                    case["id"]
                )

                st.rerun()

        st.divider()

    if st.session_state.selected_case:

        render_case_review(
            st.session_state.selected_case
        )


# ============================================================
# DOCTOR CASE REVIEW
# ============================================================

def render_case_review(case_id):

    case = next(
        case
        for case in st.session_state.cases
        if case["id"] == case_id
    )

    st.markdown(
        "## Case review"
    )

    st.markdown(
        f"<span class='case-id'>"
        f"{case['id']}"
        f"</span> "
        f"&nbsp; "
        f"<span class='muted'>"
        f"Submitted {case['submitted']}"
        f"</span>",
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1.15, 1]
    )

    # --------------------------------------------------------
    # X-RAY
    # --------------------------------------------------------

    with left:

        st.markdown(
            "### Uploaded X-ray"
        )

        if case["image_bytes"]:

            st.image(
                case["image_bytes"],
                use_container_width=True
            )

        elif (
            case["image"]
            and Path(case["image"]).exists()
        ):

            st.image(
                case["image"],
                use_container_width=True
            )

        else:

            st.info(
                "No image file is attached "
                "to this demonstration case."
            )

    # --------------------------------------------------------
    # AI ANALYSIS
    # --------------------------------------------------------

    with right:

        st.markdown(
            "### AI-Assisted Analysis"
        )

        st.caption(
            "This information supports review; "
            "it does not replace clinical judgment."
        )

        st.markdown(
            f"""
            <div class="ai-panel">

                <div class="ai-label">
                    Model
                </div>

                <div class="model-badge">
                    {case.get(
                        "ai_model",
                        "ResNet50 + RadImageNet"
                    )}
                </div>

                <br><br>

                <div class="ai-label">
                    AI prediction
                </div>

                <div class="ai-result">
                    {case.get(
                        "ai_prediction",
                        "No prediction available"
                    )}
                </div>

                <div class="ai-label">
                    Confidence
                </div>

                <div>
                    {case.get(
                        "ai_confidence",
                        "Not available"
                    )}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            "### Doctor review"
        )

        assessment = st.text_input(
            "Assessment",
            value=case["assessment"] or "",
            key=f"assessment-{case_id}"
        )

        comments = st.text_area(
            "Comments",
            value=case["comments"] or "",
            height=140,
            key=f"comments-{case_id}"
        )

        if st.button(
            "Submit review",
            type="primary",
            key=f"submit-{case_id}"
        ):

            if (
                not assessment.strip()
                or not comments.strip()
            ):

                st.error(
                    "Please add both an assessment "
                    "and comments before submitting."
                )

            else:

                case["assessment"] = (
                    assessment.strip()
                )

                case["comments"] = (
                    comments.strip()
                )

                case["review_date"] = (
                    datetime.now().strftime(
                        "%d %b %Y"
                    )
                )

                case["status"] = (
                    "Doctor review completed"
                )

                st.session_state.selected_case = None

                st.success(
                    "Review submitted. The response "
                    "is now visible in the User dashboard."
                )

                st.rerun()


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    configure_page()

    initialize_state()

    if not st.session_state.authenticated:

        render_login()

        return

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    with st.sidebar:

        st.markdown(
            "## Bonewise"
        )

        st.caption(
            "AI-assisted X-ray screening"
        )

        st.markdown(
            f"**{st.session_state.user_name}**"
        )

        st.caption(
            f"Signed in as "
            f"{st.session_state.role}"
        )

        st.markdown("---")

        st.caption(
            "Session-only demo data"
        )

        st.caption(
            "No diagnosis is made by this prototype."
        )

        st.markdown("---")

        if st.button(
            "Sign out"
        ):

            st.session_state.authenticated = False
            st.session_state.role = None
            st.session_state.user_name = None
            st.session_state.selected_case = None

            st.rerun()

    # --------------------------------------------------------
    # ROLE-BASED FRONTEND
    # --------------------------------------------------------

    if st.session_state.role == "User":

        render_user()

    else:

        render_doctor()


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
