from datetime import datetime
from pathlib import Path
import re
import io

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "resnet50_radimagenet_layer3_layer4_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@st.cache_resource
def load_bonewise_model():
    if not MODEL_PATH.exists():
        return None

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(2048, 2)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model

def predict_xray(image_bytes):
    model = load_bonewise_model()

    if model is None:
        return None

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = MODEL_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    predicted_class = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_class].item())

    label = "Cancer / Tumor" if predicted_class == 1 else "Normal"

    return {
        "label": label,
        "class": predicted_class,
        "confidence": confidence,
    }    
DISCLAIMER = (
    "This application is an AI-assisted research prototype. AI output is intended "
    "to support professional review and is not a medical diagnosis."
)

# Doctors have their own fixed accounts, kept separate from regular user sign-ups.
DOCTOR_ACCOUNTS = {
    "dr.rao@bonewise.demo": {"password": "demo", "name": "Dr. Meera Rao"},
    "dr.khan@bonewise.demo": {"password": "demo", "name": "Dr. Imran Khan"},
    "dr.iyer@bonewise.demo": {"password": "demo", "name": "Dr. Ananya Iyer"},
}

# One seeded demo user account. Anyone can also create their own from the login page.
DEMO_USER_ACCOUNT = {
    "user@bonewise.demo": {"password": "demo", "name": "Priya Sharma"},
}

ANATOMY_FACTS = [
    ("Femur", "The femur is the longest and one of the strongest bones in the human body, running from the hip to the knee."),
    ("Tibia", "The tibia, or shin bone, bears most of the body's weight and connects the knee to the ankle."),
    ("Fibula", "The fibula runs alongside the tibia and mainly helps with muscle attachment and ankle stability rather than weight-bearing."),
    ("Humerus", "The humerus is the long bone of the upper arm, connecting the shoulder to the elbow."),
    ("Spine", "The spine is made up of 33 individual vertebrae stacked to protect the spinal cord and support posture."),
    ("Pelvis", "The pelvis connects the spine to the lower limbs and protects several internal organs."),
    ("Joints", "Joints are the points where two or more bones meet, allowing movement while bones themselves stay rigid."),
    ("Bone marrow", "Bone marrow, found inside many bones, is where the body produces red and white blood cells."),
]

BONE_HEALTH_FACTS = [
    ("Calcium", "Calcium is a key mineral for building and maintaining bone tissue throughout life."),
    ("Vitamin D", "Vitamin D helps the body absorb calcium effectively. Sunlight exposure and diet both contribute to healthy levels."),
    ("Weight-bearing activity", "Regular weight-bearing activity, such as walking or strength training, is generally associated with stronger bones."),
    ("Balanced nutrition", "A varied, balanced diet supports overall skeletal health alongside many other bodily functions."),
    ("Bone density", "Bone density naturally changes with age, and is one of several factors clinicians may consider during an evaluation."),
    ("Healthy habits", "Avoiding smoking and limiting excessive alcohol intake are commonly cited as supportive of long-term bone health."),
    ("Growth and aging", "Bone tissue is continuously broken down and rebuilt throughout life, a process that shifts with age."),
    ("General information only", "These notes are general educational information, not personalized medical advice or a substitute for professional guidance."),
]

# --------------------------------------------------------------------------
# Frontend-only placeholders
# --------------------------------------------------------------------------

# Model inference and backend requests are intentionally disconnected for now.
# These lists keep the documentation preview usable until a real NLP service is added.
BONE_TERMS = ["tibia", "femur", "fibula", "humerus", "radius", "ulna", "wrist", "ankle", "hip", "shoulder", "clavicle", "spine", "patella", "pelvis"]
FINDING_TERMS = ["lesion", "fracture", "mass", "erosion", "sclerosis", "disruption", "irregularity", "deformity", "swelling"]
RECOMMEND_PATTERNS = [
    (r"further imaging", "Further imaging"),
    (r"clinical correlation", "Clinical correlation"),
    (r"follow[- ]?up", "Follow-up"),
    (r"biopsy", "Biopsy"),
    (r"\bmri\b", "MRI"),
    (r"\bct\b", "CT"),
]


def extract_notes_locally(notes: str) -> dict:
    lower = notes.lower()
    return {
        "location": ", ".join(term for term in BONE_TERMS if term in lower),
        "finding": ", ".join(term for term in FINDING_TERMS if term in lower),
        "recommendation": ", ".join(label for pattern, label in RECOMMEND_PATTERNS if re.search(pattern, notes, re.IGNORECASE)),
    }


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
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap');

        :root {
            --ink:#1d2a38; --muted:#667483; --line:#d7dee5; --ivory:#f4f7f9;
            --paper:#ffffff; --sage:#2d6f9f; --sage-light:#e7f0f7; --sage-dark:#245778;
        }

        /* ---------- base ---------- */
        .stApp { background:var(--ivory); color:var(--ink); }
        [data-testid="stHeader"] { background:transparent; }
        section.main > div.block-container { max-width:1080px; padding-top:2.4rem; }
        h1, h2, h3 { color:var(--ink); font-family:'Newsreader', Georgia, serif; letter-spacing:-.02em; }
        h1 { font-size:2.15rem; margin-bottom:.3rem; }
        h2 { font-size:1.5rem; margin-top:2.3rem; margin-bottom:.9rem; }
        h3 { font-size:1.2rem; }
        p, label, .stMarkdown, .stButton, .stTextInput, .stTextArea, .stSelectbox { font-family:'DM Sans', sans-serif; }
        hr { border-color:var(--line) !important; margin:0.4rem 0 !important; }

        /* ---------- sidebar ---------- */
        [data-testid="stSidebar"] { background:#eef2f5; border-right:1px solid var(--line); }
        [data-testid="stSidebar"] * { color:var(--ink); }
        [data-testid="stSidebar"] h2 { font-size:1.3rem; margin-top:0; }
        .role-chip {
            display:inline-block; background:var(--sage-light); color:var(--sage-dark);
            padding:3px 11px; border-radius:20px; font-size:.7rem; font-weight:700;
            letter-spacing:.06em; text-transform:uppercase; margin-top:6px;
        }

        /* ---------- text ---------- */
        .eyebrow { color:var(--sage); font:700 11px 'DM Sans',sans-serif; letter-spacing:.14em; text-transform:uppercase; }
        .lede { color:var(--muted); font-size:1.03rem; max-width:640px; line-height:1.6; margin-top:.4rem; }
        .muted { color:var(--muted); font-size:.9rem; }
        .status { color:var(--sage-dark); font-weight:600; font-size:.88rem; }
        .case-id { color:var(--sage-dark); font-weight:700; letter-spacing:.04em; font-size:.98rem; }

        /* ---------- panels ---------- */
        .paper { background:var(--paper); border:1px solid var(--line); border-radius:3px; padding:24px 26px; margin:6px 0 22px; }
        .metric { font:600 2.1rem 'Newsreader',serif; color:var(--ink); line-height:1.1; }

        /* ---------- timeline ---------- */
        .timeline { border-left:2px solid var(--line); margin:16px 0 4px 8px; padding-left:24px; }
        .timeline-row { position:relative; padding:3px 0 18px; color:var(--muted); font-size:.94rem; }
        .timeline-row:before {
            content:''; position:absolute; left:-33px; top:4px; width:13px; height:13px;
            border-radius:50%; background:var(--line); border:3px solid var(--ivory);
        }
        .timeline-row.done { color:var(--ink); font-weight:600; }
        .timeline-row.done:before { background:var(--sage); }

        /* ---------- disclaimer ---------- */
        .disclaimer {
            background:var(--sage-light); border-left:3px solid var(--sage);
            padding:12px 16px; color:var(--sage-dark); font-size:.82rem; line-height:1.55;
            margin:18px 0 26px; border-radius:0 3px 3px 0;
        }

        /* ---------- buttons ---------- */
        div.stButton > button {
            border:1px solid #b8b7ac; border-radius:3px; background:transparent;
            color:var(--ink); font-weight:600; padding:0.5rem 1.1rem;
            transition:border-color .15s ease, color .15s ease, background .15s ease;
        }
        div.stButton > button:hover { border-color:var(--sage); color:var(--sage-dark); }
        div.stButton > button[kind="primary"] { background:var(--sage); border-color:var(--sage); color:white; }
        div.stButton > button[kind="primary"]:hover { background:var(--sage-dark); border-color:var(--sage-dark); color:white; }

        /* ---------- inputs ---------- */
        [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea {
            border-radius:3px !important; border-color:var(--line) !important; background:var(--paper) !important;
        }
        [data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus {
            border-color:var(--sage) !important; box-shadow:0 0 0 1px var(--sage) !important;
        }
        [data-testid="stFileUploader"] { background:var(--paper); border:1px dashed #b9b5a8; border-radius:3px; padding:10px; }

        /* ---------- tabs ---------- */
        [data-baseweb="tab-list"] { gap:28px; border-bottom:1px solid var(--line); }
        [data-baseweb="tab"] { color:var(--muted); font-weight:600; padding:0 0 10px 0; }
        [data-baseweb="tab"] p { font-family:'DM Sans',sans-serif; font-size:.95rem; }
        [aria-selected="true"] { color:var(--ink) !important; }
        [data-baseweb="tab-highlight"] { background-color:var(--sage) !important; }
        [data-baseweb="tab-border"] { background-color:var(--line) !important; }

        /* ---------- expander (demo access) ---------- */
        [data-testid="stExpander"] { border:1px solid var(--line); border-radius:3px; background:transparent; }
        [data-testid="stExpander"] summary { font-size:.85rem; color:var(--muted); font-weight:600; }

        /* ---------- login ---------- */
        .login-brand { padding:36px 48px 0 4px; max-width:440px; }
        .login-mark { font-family:'Newsreader',serif; font-size:2.5rem; letter-spacing:-.01em; margin-bottom:12px; }
        .login-tagline { font-family:'Newsreader',serif; font-size:1.15rem; color:var(--ink); margin-bottom:12px; }
        .login-explainer { color:var(--muted); font-size:.95rem; line-height:1.6; margin-bottom:8px; }
        .step-row { display:flex; gap:16px; padding:16px 0; border-top:1px solid var(--line); }
        .step-number { font-family:'Newsreader',serif; color:var(--sage); font-size:1.05rem; min-width:26px; }
        .step-title { font-weight:600; font-size:.95rem; margin-bottom:2px; }
        .step-text { color:var(--muted); font-size:.87rem; line-height:1.5; }
        .login-footnote { margin-top:28px; color:var(--muted); font-size:.76rem; letter-spacing:.03em; }
        .login-panel { background:var(--paper); border:1px solid var(--line); border-radius:3px; padding:34px 36px; }
        .login-demo-hint { color:var(--muted); font-size:.82rem; line-height:1.7; }

        /* ---------- case list rows ---------- */
        .case-row { padding:16px 2px; border-bottom:1px solid var(--line); }
        .case-row:first-of-type { border-top:1px solid var(--line); }

        /* ---------- report / analysis blocks ---------- */
        .report, .analysis-block {
            background:var(--paper); border:1px solid var(--line); border-top:3px solid var(--sage);
            border-radius:0 3px 3px 3px; padding:24px 26px; margin:12px 0;
        }
        .report-label, .analysis-label { color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; margin-top:14px; }
        .report-value, .analysis-value { font-size:1.0rem; margin-top:4px; line-height:1.5; }

        /* ---------- flashcards ---------- */
        .flashcard { background:var(--paper); border:1px solid var(--line); border-top:3px solid var(--sage); border-radius:0 3px 3px 3px; padding:32px 28px; min-height:150px; }
        .flashcard-term { font-family:'Newsreader',serif; font-size:1.3rem; margin-bottom:10px; }
        .flashcard-body { color:var(--muted); line-height:1.6; font-size:.95rem; }
        .flash-index { color:var(--muted); font-size:.8rem; text-align:center; margin-top:10px; letter-spacing:.05em; }

        /* ---------- stat row ---------- */
        .stat-row { display:flex; border-top:1px solid var(--line); border-bottom:1px solid var(--line); margin:14px 0 30px; }
        .stat-cell { flex:1; padding:18px 22px; border-right:1px solid var(--line); }
        .stat-cell:last-child { border-right:none; }
        .stat-label { color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; margin-bottom:6px; }
        .clinician-brand { padding:8px 0 18px; border-bottom:1px solid var(--line); margin-bottom:18px; }
        .clinician-brand strong { display:block; font:600 1.55rem 'Newsreader',serif; }
        .clinician-brand span { color:var(--muted); font-size:.78rem; letter-spacing:.08em; text-transform:uppercase; }
        .section-kicker { color:var(--muted); font-size:.72rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin:20px 0 6px; }
        .worklist { background:var(--paper); border:1px solid var(--line); margin:12px 0 26px; }
        .worklist-head, .worklist-row { display:grid; grid-template-columns:1.35fr .85fr 1.3fr .95fr 1fr .7fr; gap:14px; align-items:center; padding:13px 16px; }
        .worklist-head { color:var(--muted); background:#f0eee7; font-size:.72rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
        .worklist-row { border-top:1px solid var(--line); font-size:.88rem; }
        .worklist-row strong { font-weight:600; }
        .status-dot { display:inline-flex; align-items:center; gap:7px; color:var(--sage-dark); font-size:.82rem; font-weight:600; }
        .status-dot:before { content:''; width:7px; height:7px; border-radius:50%; background:var(--sage); }
        .status-dot.reviewed:before { background:#9b9a91; }
        .case-header { border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:22px; }
        .case-meta { color:var(--muted); font-size:.9rem; margin-top:5px; }
        .image-stage { background:#1f2425; border:1px solid #303737; min-height:400px; display:flex; align-items:center; justify-content:center; padding:18px; }
        .final-impression { border-left:3px solid var(--sage); background:var(--sage-light); padding:12px 16px; margin:12px 0 20px; }
        .report-sheet { background:var(--paper); border:1px solid var(--line); padding:32px 38px; max-width:820px; }
        .report-sheet h2 { border-top:1px solid var(--line); padding-top:1rem; }
        .profile-row { display:flex; justify-content:space-between; gap:16px; border-top:1px solid var(--line); padding:16px 0; }
        @media (max-width: 850px) { .worklist { overflow-x:auto; } .worklist-head, .worklist-row { min-width:760px; } .report-sheet { padding:24px; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def demo_image_path():
    image_dir = PROJECT_ROOT / "data" / "processed" / "combined" / "test" / "images"
    images = sorted(image_dir.glob("*.png"))
    return str(images[0]) if images else None


def make_demo_cases():
    # Cases are created only when a logged-in user submits an X-ray.
    return []


STATUS_NEW = "NEW"
STATUS_IN_REVIEW = "IN REVIEW"
STATUS_REVIEWED = "REVIEWED"


def normalize_case_status(status):
    if status in {"Reviewed", STATUS_REVIEWED}:
        return STATUS_REVIEWED
    if status in {"Doctor Review", "In review", STATUS_IN_REVIEW}:
        return STATUS_IN_REVIEW
    return STATUS_NEW


def normalize_cases():
    for case in st.session_state.cases:
        case["status"] = normalize_case_status(case.get("status"))


def render_reset_cases(scope):
    reset_key = f"reset-cases-{scope}"
    confirm_key = f"confirm-reset-cases-{scope}"

    if st.button("Reset cases", key=reset_key):
        st.session_state[f"{reset_key}-confirm"] = True
        st.rerun()

    if st.session_state.get(f"{reset_key}-confirm", False):
        st.warning("This removes all cases currently stored in this frontend session.")
        confirmed = st.checkbox(
            "I understand that all current cases will be cleared.",
            key=confirm_key,
        )
        if st.button("Confirm reset", type="primary", key=f"{reset_key}-final"):
            if confirmed:
                st.session_state.cases = []
                st.session_state.selected_case = None
                st.session_state.selected_client = None
                st.session_state.selected_report = None
                st.session_state.open_user_case = None
                st.session_state.review_success_case = None
                st.session_state[f"{reset_key}-confirm"] = False
                st.success("All cases have been cleared.")
                st.rerun()
            else:
                st.error("Please confirm the reset first.")


def initialize_state():
    if "cases" not in st.session_state:
        st.session_state.cases = make_demo_cases()
    else:
        # Remove legacy seeded/demo cases from an older frontend version.
        st.session_state.cases = [
            case for case in st.session_state.cases
            if case.get("client_email")
        ]
    normalize_cases()
    if "review_success_case" not in st.session_state:
        st.session_state.review_success_case = None
    if "next_case_number" not in st.session_state:
        st.session_state.next_case_number = 2401
    if "next_patient_number" not in st.session_state:
        st.session_state.next_patient_number = 1001
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "role" not in st.session_state:
        st.session_state.role = None
    if "user_name" not in st.session_state:
        st.session_state.user_name = None
    if "user_email" not in st.session_state:
        st.session_state.user_email = None
    if "selected_case" not in st.session_state:
        st.session_state.selected_case = None
    if "open_user_case" not in st.session_state:
        st.session_state.open_user_case = None
    if "anatomy_index" not in st.session_state:
        st.session_state.anatomy_index = 0
    if "health_index" not in st.session_state:
        st.session_state.health_index = 0
    if "accounts" not in st.session_state:
        # Session-only registry of user accounts (doctors are kept separately, above).
        st.session_state.accounts = dict(DEMO_USER_ACCOUNT)
    if "uploader_version" not in st.session_state:
        st.session_state.uploader_version = 0
    if "doctor_section" not in st.session_state:
        st.session_state.doctor_section = "Overview"
    if "selected_client" not in st.session_state:
        st.session_state.selected_client = None
    if "selected_report" not in st.session_state:
        st.session_state.selected_report = None
def next_case_id():
    case_id = f"BW-{st.session_state.next_case_number}"
    st.session_state.next_case_number += 1
    return case_id


def next_patient_id():
    patient_id = f"P-{st.session_state.next_patient_number}"
    st.session_state.next_patient_number += 1
    return patient_id


def get_user_cases():
    email = (st.session_state.get("user_email") or "").strip().lower()
    return [case for case in st.session_state.cases if case.get("client_email") == email]


def get_patient_cases():
    return [case for case in st.session_state.cases if case.get("patient_added")]


def get_doctor_cases():
    doctor_name = st.session_state.get("user_name")
    return [
        case for case in st.session_state.cases
        if case.get("assigned_doctor") == doctor_name
    ]


def get_patient_id_for_email(email):
    email = (email or "").strip().lower()
    existing = next((case.get("patient_id") for case in st.session_state.cases
                     if case.get("patient_added") and case.get("client_email") == email and case.get("patient_id")), None)
    return existing


def find_account(email):
    email = email.strip().lower()
    if email in DOCTOR_ACCOUNTS:
        info = DOCTOR_ACCOUNTS[email]
        return {"password": info["password"], "name": info["name"], "role": "Doctor"}
    if email in st.session_state.accounts:
        info = st.session_state.accounts[email]
        return {"password": info["password"], "name": info["name"], "role": "User"}
    return None


def case_timeline(case):
    ai_completed = bool(case.get("ai_prediction"))

    timeline_items = [
        ("Case submitted", True),
        ("X-ray available", bool(case.get("image_bytes"))),
        ("AI-assisted analysis", ai_completed),
        ("Doctor review", case.get("reviewed_by") is not None),
        ("Clinical report", case.get("final_impression") is not None),
    ]

    for label, completed in timeline_items:
        status_class = "completed" if completed else "pending"
        status_text = "Completed" if completed else "Pending"

        st.markdown(
            f"""
            <div class="timeline-item">
                <div class="timeline-dot {status_class}"></div>
                <div class="timeline-content">
                    <div class="timeline-label">{label}</div>
                    <div class="timeline-status">{status_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_login():
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown(
            """
            <div class="login-brand">
                <div class="login-mark">Bonewise</div>
                <div class="login-tagline">AI-assisted bone imaging review</div>
                <div class="login-explainer">Review bone X-ray studies with machine learning support and clinician-led assessment.</div>
                <div class="login-footnote">Clinical research prototype. Demo data only.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="eyebrow">Account access</div>', unsafe_allow_html=True)
        sign_in_tab, create_tab = st.tabs(["Sign in", "Create account"])

        with sign_in_tab:
            st.markdown("<h3 style='margin-top:14px'>Welcome back</h3>", unsafe_allow_html=True)
            email = st.text_input("Email or username", placeholder="you@bonewise.demo", key="login-email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="login-password")
            if st.button("Sign in", type="primary", key="login-submit"):
                account = find_account(email)
                if account and account["password"] == password:
                    st.session_state.authenticated = True
                    st.session_state.role = account["role"]
                    st.session_state.user_name = account["name"]
                    st.session_state.user_email = email.strip().lower()
                    st.session_state.selected_case = None
                    st.session_state.open_user_case = None
                    st.rerun()
                else:
                    st.error("Those credentials weren't recognised. Try demo access below, or create an account.")
            with st.expander("Demo access"):
                st.markdown(
                    """
                    <div class="login-demo-hint">
                        user@bonewise.demo / demo<br>
                        dr.rao@bonewise.demo / demo<br>
                        dr.khan@bonewise.demo / demo<br>
                        dr.iyer@bonewise.demo / demo
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with create_tab:
            st.markdown("<h3 style='margin-top:14px'>Create an account</h3>", unsafe_allow_html=True)
            st.caption("New accounts here are set up as users. Doctor accounts are provided separately for this demo.")
            new_name = st.text_input("Full name", placeholder="Your name", key="signup-name")
            new_email = st.text_input("Email", placeholder="you@example.com", key="signup-email")
            new_password = st.text_input("Password", type="password", key="signup-password")
            if st.button("Create account", type="primary", key="signup-submit"):
                clean_email = new_email.strip().lower()
                if not new_name.strip() or not clean_email or not new_password:
                    st.error("Please fill in your name, email, and password.")
                elif clean_email in DOCTOR_ACCOUNTS or clean_email in st.session_state.accounts:
                    st.error("An account with that email already exists. Try signing in instead.")
                else:
                    st.session_state.accounts[clean_email] = {
                        "password": new_password,
                        "name": new_name.strip(),
                    }
                    st.session_state.authenticated = True
                    st.session_state.role = "User"
                    st.session_state.user_name = new_name.strip()
                    st.session_state.selected_case = None
                    st.session_state.open_user_case = None
                    st.success("Account created. Signing you in.")
                    st.rerun()
            st.caption("Accounts created here last for this session only.")
        st.markdown("</div>", unsafe_allow_html=True)


def render_flashcard_deck(title, deck, state_key):
    st.markdown(f"#### {title}")
    index = st.session_state[state_key]
    term, body = deck[index]
    st.markdown(
        f'<div class="flashcard"><div class="flashcard-term">{term}</div>'
        f'<div class="flashcard-body">{body}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="flash-index">{index + 1} / {len(deck)}</div>', unsafe_allow_html=True)
    nav_left, nav_right = st.columns(2)
    with nav_left:
        if st.button("← Previous", key=f"{state_key}-prev", disabled=index == 0):
            st.session_state[state_key] -= 1
            st.rerun()
    with nav_right:
        if st.button("Next →", key=f"{state_key}-next", disabled=index == len(deck) - 1):
            st.session_state[state_key] += 1
            st.rerun()


def render_health():
    st.markdown("## Learn while you wait")
    st.caption("Short, general educational notes. These are not personalized medical advice.")
    deck_left, deck_right = st.columns(2, gap="large")
    with deck_left:
        render_flashcard_deck("Anatomy facts", ANATOMY_FACTS, "anatomy_index")
    with deck_right:
        render_flashcard_deck("Bone health", BONE_HEALTH_FACTS, "health_index")


def render_doctor_report(case):
    st.markdown(
        f"""
        <div class="report">
            <div class="eyebrow">Doctor's response</div>
            <div class="report-label">Assessment</div>
            <div class="report-value">{case['assessment']}</div>
            <div class="report-label">Comments</div>
            <div class="report-value">{case['comments']}</div>
            <div class="report-label">Reviewed by</div>
            <div class="report-value">{case['reviewed_by'] or 'Reviewing doctor'}</div>
            <div class="report-label">Review date</div>
            <div class="report-value">{case['review_date']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_case_row(case, show_open_for_user=False, key_prefix="row", row_index=0):
    """One consistently aligned row for a case, with an optional Open action."""
    st.markdown('<div class="case-row">', unsafe_allow_html=True)
    left, right = st.columns([4, 1], vertical_alignment="center")
    with left:
        st.markdown(
            f"<span class='case-id'>{case['id']}</span> &nbsp; "
            f"<span class='muted'>{case['submitted']}</span><br>"
            f"<span class='status'>{case['status']}</span>",
            unsafe_allow_html=True,
        )
    with right:
        clicked = False
        if show_open_for_user:
            if case["status"] == STATUS_REVIEWED:
                clicked = st.button("Open", key=f"{key_prefix}-{row_index}-{case['id']}")
        else:
            clicked = st.button("Open", key=f"{key_prefix}-{row_index}-{case['id']}")
    st.markdown("</div>", unsafe_allow_html=True)
    return clicked


def render_user():
    user_cases = get_user_cases()

    st.markdown('<div class="eyebrow">Bonewise / patient portal</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top:4px'>My studies</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="lede">Upload an X-ray and follow its clinical review status.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="disclaimer">✦ &nbsp;' + DISCLAIMER + "</div>", unsafe_allow_html=True)
    st.markdown(f"### Good to see you, {st.session_state.user_name.split()[0]}")

    current = next((case for case in reversed(user_cases) if case["status"] != STATUS_REVIEWED), None)
    if current:
        st.markdown('<div class="paper"><div class="eyebrow">Current study</div>', unsafe_allow_html=True)
        st.markdown(f"### {current['id']}")
        st.markdown(f"<span class='muted'>Submitted {current['submitted']}</span>", unsafe_allow_html=True)
        case_timeline(current)
        if current["status"] == STATUS_NEW:
            st.info("Your study has been received and is waiting for clinical review.")
        else:
            st.info("Your study is currently being reviewed by a clinician.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("## Submit an X-ray")
    st.caption("PNG, JPG, and JPEG files are supported.")
    uploader_key = f"xray-upload-{st.session_state.uploader_version}"
    upload = st.file_uploader(
        "Choose an X-ray image",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        key=uploader_key,
    )

    if upload:
        image_bytes = upload.getvalue()
        upload_signature = __import__("hashlib").sha256(image_bytes).hexdigest()
        already_submitted = any(
            case.get("upload_signature") == upload_signature and case.get("client_email") == st.session_state.user_email
            for case in st.session_state.cases
        )
        st.image(upload, caption="Selected X-ray", width=420)

        doctor_options = [info["name"] for info in DOCTOR_ACCOUNTS.values()]
        selected_doctor = st.selectbox(
            "Choose a doctor for clinical review",
            doctor_options,
            key="submit-doctor",
        )

        if already_submitted:
            st.info("This X-ray has already been submitted as one of your studies.")
        elif st.button("Submit X-ray", type="primary", key="submit-xray"):
            case_id = next_case_id()
            prediction = predict_xray(image_bytes)
            now = datetime.now().strftime("%d %b %Y")
            st.session_state.cases.append({
                "id": case_id,
                "client_name": st.session_state.user_name,
                "client_email": st.session_state.user_email,
                "assigned_doctor": selected_doctor,
                "client_id": case_id,
                "patient_id": None,
                "patient_added": False,
                "age": "Not recorded",
                "sex": "Not recorded",
                "body_location": "Not specified",
                "study_type": "Uploaded X-ray",
                "date_added": now,
                "imaging_status": "Study available",
                "submitted": now,
                "status": STATUS_NEW,
                "image": None,
                "image_bytes": image_bytes,
                "upload_signature": upload_signature,
                "assessment": None,
                "comments": None,
                "final_impression": None,
                "recommendation": None,
                "review_date": None,
                "reviewed_by": None,
                "ai_prediction": prediction.get("label") if prediction else None,
                "ai_confidence": prediction.get("confidence") if prediction else None,
            })
            st.session_state.uploader_version += 1
            st.success(f"Your X-ray has been sent to {selected_doctor} for clinical review. Case ID: {case_id}.")
            st.rerun()

    st.markdown("## Previous studies")
    previous = [case for case in reversed(user_cases) if not (current and case["id"] == current["id"])]
    if not previous:
        st.caption("No previous studies yet.")
    for index, case in enumerate(previous):
        if render_case_row(case, show_open_for_user=True, key_prefix="open-user", row_index=index):
            st.session_state.open_user_case = case["id"]
            st.rerun()

    if st.session_state.open_user_case:
        opened = next((case for case in user_cases if case["id"] == st.session_state.open_user_case), None)
        if opened:
            st.markdown(f"### Case {opened['id']}")
            if opened.get("status") == STATUS_REVIEWED:
                render_clinical_report(opened)
            else:
                st.info("The clinician's report will appear here after the review is completed.")
            if st.button("Close", key="close-user-case"):
                st.session_state.open_user_case = None
                st.rerun()

    render_health()

def render_doctor():
    if st.session_state.get("review_success_case"):
        st.success("Review submitted successfully.")
        st.session_state.review_success_case = None

    if st.session_state.selected_case:
        render_case_review(st.session_state.selected_case)
        return

    with st.sidebar:
        st.markdown('<div class="clinician-brand"><strong>Bonewise</strong><span>Clinical Review</span></div>', unsafe_allow_html=True)
        if st.button("Overview", key="doctor-nav-overview", width="stretch"):
            st.session_state.doctor_section = "Overview"
            st.session_state.selected_client = None
            st.session_state.selected_report = None
            st.rerun()

        st.markdown('<div class="section-kicker">Patients</div>', unsafe_allow_html=True)
        if st.button("All patients", key="doctor-nav-all-patients", width="stretch"):
            st.session_state.doctor_section = "All Patients"
            st.session_state.selected_client = None
            st.rerun()
        if st.button("New submissions", key="doctor-nav-new-submissions", width="stretch"):
            st.session_state.doctor_section = "New Submissions"
            st.session_state.selected_client = None
            st.rerun()

        st.markdown('<div class="section-kicker">Reviews</div>', unsafe_allow_html=True)
        if st.button("In review", key="doctor-nav-in-review", width="stretch"):
            st.session_state.doctor_section = "In Review"
            st.session_state.selected_report = None
            st.rerun()
        if st.button("Reviewed", key="doctor-nav-reviewed", width="stretch"):
            st.session_state.doctor_section = "Reviewed Cases"
            st.session_state.selected_report = None
            st.rerun()

        st.markdown('<div class="section-kicker">Account</div>', unsafe_allow_html=True)
        st.markdown(f"**{st.session_state.user_name}**<br><span class='muted'>Clinician</span>", unsafe_allow_html=True)
        render_reset_cases("doctor")

        if st.button("Sign out", key="doctor-sign-out", width="stretch"):
            st.session_state.authenticated = False
            st.session_state.role = None
            st.session_state.user_name = None
            st.session_state.user_email = None
            st.session_state.selected_case = None
            st.session_state.selected_client = None
            st.session_state.selected_report = None
            st.rerun()

    if st.session_state.selected_client:
        render_client_profile(st.session_state.selected_client)
        return
    if st.session_state.doctor_section == "Overview":
        render_doctor_overview()
    elif st.session_state.doctor_section == "All Patients":
        render_client_registry()
    elif st.session_state.doctor_section == "New Submissions":
        render_new_clients()
    elif st.session_state.doctor_section == "Reviewed Cases":
        render_reviewed_cases()
    else:
        render_review_queue()

def render_doctor_overview():
    cases = get_doctor_cases()
    patient_cases = [case for case in cases if case.get("patient_added")]
    new_submissions = [case for case in cases if not case.get("patient_added")]
    in_review = [case for case in patient_cases if case["status"] == STATUS_IN_REVIEW]
    reviewed_today = [case for case in patient_cases if case.get("review_date") == datetime.now().strftime("%d %b %Y")]
    patients = {case.get("patient_id") for case in patient_cases if case.get("patient_id")}

    st.markdown('<div class="eyebrow">Bonewise / clinician overview</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top:4px'>Clinical overview</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lede'>New submissions, patient records, and active clinical reviews.</p>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="stat-row"><div class="stat-cell"><div class="stat-label">Patients</div><div class="metric">{len(patients)}</div></div>'
        f'<div class="stat-cell"><div class="stat-label">New submissions</div><div class="metric">{len(new_submissions)}</div></div>'
        f'<div class="stat-cell"><div class="stat-label">In review</div><div class="metric">{len(in_review)}</div></div>'
        f'<div class="stat-cell"><div class="stat-label">Reviewed today</div><div class="metric">{len(reviewed_today)}</div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Recent submissions")
    recent = sorted(cases, key=lambda case: case["submitted"], reverse=True)[:5]
    if not recent:
        st.info("No submissions yet.")
        return
    render_worklist_header(["User", "Study ID", "Region", "Study date", "Status", "Action"])
    for index, case in enumerate(recent):
        display_status = "NEW SUBMISSION" if not case.get("patient_added") else case["status"]
        st.markdown(
            f'<div class="worklist-row"><span><strong>{case["client_name"]}</strong></span><span class="case-id">{case["id"]}</span>'
            f'<span>{case["body_location"]}</span><span>{case["submitted"]}</span><span class="status-dot {"reviewed" if case["status"] == STATUS_REVIEWED else ""}">{display_status}</span><span></span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open", key=f"overview-open-{index}-{case['id']}"):
            st.session_state.selected_case = case["id"]
            st.rerun()
    render_worklist_footer()

def render_worklist_header(columns):
    st.markdown(
        '<div class="worklist"><div class="worklist-head">'
        + "".join(f"<span>{column}</span>" for column in columns)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_worklist_footer():
    st.markdown("</div>", unsafe_allow_html=True)


def render_review_queue():
    cases = [case for case in get_doctor_cases() if case.get("patient_added") and case["status"] == STATUS_IN_REVIEW]
    st.markdown('<div class="eyebrow">Reviews / active cases</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top:4px'>In review</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lede'>Patient studies currently being reviewed by the clinical team.</p>", unsafe_allow_html=True)
    if not cases:
        st.info("No cases are currently being reviewed.")
        return
    render_worklist_header(["Patient", "Patient ID", "Study", "Study date", "Status", "Action"])
    for index, case in enumerate(cases):
        st.markdown(
            f'<div class="worklist-row"><span><strong>{case["client_name"]}</strong></span><span class="case-id">{case.get("patient_id") or "-"}</span>'
            f'<span>{case["body_location"]}<br><span class="muted">{case["id"]}</span></span><span>{case["submitted"]}</span>'
            f'<span class="status-dot">{case["status"]}</span><span></span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open review", key=f"review-queue-open-{index}-{case['id']}", type="primary"):
            st.session_state.selected_case = case["id"]
            st.rerun()
    render_worklist_footer()


def render_client_registry():
    query = st.text_input("Search patients", placeholder="Search by name or patient ID", key="client-search")
    patients = {}
    for case in get_doctor_cases():
        if case.get("patient_added") and case.get("patient_id"):
            patients.setdefault(case["patient_id"], case)
    filtered = [case for case in patients.values() if query.lower() in f"{case['client_name']} {case['patient_id']}".lower()]

    st.markdown('<div class="eyebrow">Patients / registry</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top:4px'>All patients</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lede'>Patients added to the clinical workspace by a doctor.</p>", unsafe_allow_html=True)
    if not filtered:
        st.info("No patients yet.")
        return
    render_worklist_header(["Patient", "Patient ID", "Latest study", "Last study", "Status", "Action"])
    for index, case in enumerate(filtered):
        patient_cases = [c for c in get_doctor_cases() if c.get("patient_id") == case.get("patient_id")]
        latest = max(patient_cases, key=lambda c: c["submitted"])
        st.markdown(
            f'<div class="worklist-row"><span><strong>{case["client_name"]}</strong></span><span class="case-id">{case["patient_id"]}</span>'
            f'<span>{latest["body_location"]}<br><span class="muted">{latest["id"]}</span></span><span>{latest["submitted"]}</span>'
            f'<span class="status-dot {"reviewed" if latest["status"] == STATUS_REVIEWED else ""}">{latest["status"]}</span><span></span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open patient", key=f"patient-registry-open-{index}-{case['patient_id']}"):
            st.session_state.selected_client = case["patient_id"]
            st.rerun()
    render_worklist_footer()


def render_new_clients():
    cases = [case for case in get_doctor_cases() if not case.get("patient_added")]
    st.markdown('<div class="eyebrow">New submissions</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top:4px'>New submissions</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lede'>Studies submitted by users that have not yet been added to a patient record.</p>", unsafe_allow_html=True)
    if not cases:
        st.info("No new submissions.")
        return
    render_worklist_header(["User", "Study ID", "Region", "Date submitted", "Status", "Action"])
    for index, case in enumerate(cases):
        st.markdown(
            f'<div class="worklist-row"><span><strong>{case["client_name"]}</strong></span><span class="case-id">{case["id"]}</span>'
            f'<span>{case["body_location"]}</span><span>{case["submitted"]}</span><span class="status-dot">NEW SUBMISSION</span><span></span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open submission", key=f"new-submission-open-{index}-{case['id']}", type="primary"):
            st.session_state.selected_case = case["id"]
            st.rerun()
    render_worklist_footer()


def render_client_profile(patient_id):
    cases = [case for case in get_doctor_cases() if case.get("patient_id") == patient_id and case.get("patient_added")]
    if not cases:
        st.session_state.selected_client = None
        return
    client = cases[0]
    if st.button("← Back to patients", key=f"back-to-client-registry-{patient_id}"):
        st.session_state.selected_client = None
        st.rerun()
    st.markdown('<div class="eyebrow">Patient profile</div>', unsafe_allow_html=True)
    st.markdown(f"<h1 style='margin-top:4px'>{client['client_name']}</h1><span class='case-id'>{patient_id}</span>", unsafe_allow_html=True)
    st.markdown(f"<p class='case-meta'>{client['age']} · {client['sex']}</p>", unsafe_allow_html=True)
    st.markdown("### Studies")
    for index, case in enumerate(sorted(cases, key=lambda item: item["submitted"], reverse=True)):
        left, right = st.columns([5, 1])
        with left:
            st.markdown(f'<div class="profile-row"><span><strong>{case["id"]}</strong><br>{case["submitted"]} · {case["body_location"]} · {case["study_type"]}</span><span class="status">{case["status"]}</span></div>', unsafe_allow_html=True)
        with right:
            if st.button("Open", key=f"profile-study-open-{index}-{case['id']}"):
                st.session_state.selected_case = case["id"]
                st.rerun()

def render_reviewed_cases():
    cases = [case for case in get_doctor_cases() if case.get("patient_added") and case["status"] == STATUS_REVIEWED]
    st.markdown('<div class="eyebrow">Case review / completed records</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top:4px'>Reviewed cases</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lede'>Completed reviews and their clinical reports.</p>", unsafe_allow_html=True)
    if not cases:
        st.info("No reviewed cases yet.")
        return
    render_worklist_header(["Client", "Case ID", "Study", "Review date", "Reviewed by", "Status"])
    for index, case in enumerate(cases):
        st.markdown(
            f'<div class="worklist-row"><span><strong>{case["client_name"]}</strong></span><span class="case-id">{case["client_id"]}</span>'
            f'<span>{case["body_location"]}</span><span>{case["review_date"]}</span><span>{case["reviewed_by"]}</span><span class="status-dot reviewed">Reviewed</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open report", key=f"reviewed-report-open-{index}-{case['id']}"):
            st.session_state.selected_report = case["id"]
            st.rerun()
    render_worklist_footer()
    if st.session_state.selected_report:
        report_case = next((case for case in cases if case["id"] == st.session_state.selected_report), None)
        if report_case:
            render_clinical_report(report_case)


def render_clinical_report(case):
    st.markdown('<div class="report-sheet">', unsafe_allow_html=True)

    st.markdown(
        "<div class='eyebrow'>Bonewise</div><h1>Clinical review report</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"**Client**  \n{case['client_name']}  \n\n"
        f"**Case ID**  \n{case['client_id']}  \n\n"
        f"**Study**  \n{case['body_location']} · {case['study_type']}  \n\n"
        f"**Study date**  \n{case['submitted']}"
    )

    st.markdown("## Imaging")
    st.markdown("The reviewed X-ray is retained in the case workspace.")

    st.markdown("## AI-assisted analysis")

    prediction = case.get("ai_prediction")
    confidence = case.get("ai_confidence")

    if prediction and confidence is not None:
        confidence_percent = confidence * 100

        st.markdown(
            f"""
            <div class="analysis-block">
                <div class="eyebrow">Model output</div>

                <div class="analysis-label">Model status</div>
                <div class="analysis-value">Connected</div>

                <div class="analysis-label">Model</div>
                <div class="analysis-value">
                    ResNet50 + RadImageNet
                </div>

                <div class="analysis-label">Prediction</div>
                <div class="analysis-value">
                    {prediction}
                </div>

                <div class="analysis-label">Confidence</div>
                <div class="analysis-value">
                    {confidence_percent:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "AI output is intended to support professional review "
            "and is not a medical diagnosis."
        )

    else:
        st.info("AI prediction unavailable. Model checkpoint not found.")

    st.markdown(
        f"## Doctor assessment\n"
        f"{case['assessment'] or 'Not recorded'}\n\n"
        f"## Clinical notes\n"
        f"{case['comments'] or 'Not recorded'}"
    )

    st.markdown(
        f"## Final impression\n"
        f"<div class='final-impression'>"
        f"{case.get('final_impression') or case['assessment'] or 'Not recorded'}"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"## Recommendation\n"
        f"{case.get('recommendation') or 'Not recorded'}"
    )

    st.markdown(
        f"## Review information\n"
        f"**Reviewed by:** {case['reviewed_by']}  \n"
        f"**Review date:** {case['review_date']}"
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_ai_panel(case):
    prediction = case.get("ai_prediction")
    confidence = case.get("ai_confidence")

    st.markdown("### AI screening")

    if prediction and confidence is not None:
        st.markdown(
            f"""
            <div class="analysis-block">
                <div class="analysis-label">Prediction</div>
                <div class="analysis-value">{prediction}</div>
                <div class="analysis-label">Confidence</div>
                <div class="analysis-value">{confidence * 100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("AI output supports professional review and is not a diagnosis.")
    else:
        st.info("AI prediction unavailable. Model checkpoint not found.")


def render_further_analysis(case):
    prediction = case.get("ai_prediction")
    confidence = case.get("ai_confidence")

    st.markdown("## Further analysis")

    if prediction and confidence is not None:
        confidence_percent = confidence * 100

        st.markdown(
            f"""
            <div class="analysis-block">
                <div class="eyebrow">AI-assisted review context</div>

                <div class="analysis-label">Current model output</div>
                <div class="analysis-value">{prediction}</div>

                <div class="analysis-label">Confidence</div>
                <div class="analysis-value">{confidence_percent:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "This model output is provided as research and decision-support "
            "information for professional review. It is not a medical diagnosis."
        )

    else:
        st.info(
            "Further AI-assisted analysis will be available when an X-ray "
            "has been processed by the model."
        )

    st.markdown("### Clinical review")

    st.write(
        "Use the model output together with the X-ray and other available "
        "clinical information. Final interpretation remains with the reviewing clinician."
    )


def render_doctor_report_placeholder():
    pass


def render_case_review(case_id):
    case = next((case for case in st.session_state.cases if case["id"] == case_id), None)
    if case is None:
        st.session_state.selected_case = None
        return

    if st.button("← Back", key=f"back-to-case-{case_id}"):
        st.session_state.selected_case = None
        st.rerun()

    if not case.get("patient_added"):
        st.markdown('<div class="eyebrow">New submission</div>', unsafe_allow_html=True)
        st.markdown(f"<h1 style='margin-top:4px'>{case['client_name']}</h1>", unsafe_allow_html=True)
        st.markdown(f"<span class='case-id'>{case['id']}</span>", unsafe_allow_html=True)
        st.markdown("### Submitted study")
        st.write(f"Study date: {case['submitted']}")
        st.write(f"Region: {case['body_location']}")
        if case.get("image_bytes"):
            st.image(case["image_bytes"], width="stretch")
        st.markdown("### Patient record")
        st.info("This submission has not been added to a patient record yet.")
        if st.button("Add as patient", type="primary", key=f"add-patient-{case_id}"):
            patient_id = get_patient_id_for_email(case.get("client_email")) or next_patient_id()
            case["patient_id"] = patient_id
            case["patient_added"] = True
            case["status"] = STATUS_IN_REVIEW
            st.session_state.selected_case = case_id
            st.rerun()
        return

    st.markdown(
        f'<div class="case-header"><div class="eyebrow">Clinical review workspace</div>'
        f'<h1 style="margin-top:4px">{case["client_name"]}</h1>'
        f'<span class="case-id">{case["patient_id"]} · {case["id"]}</span>'
        f'<div class="case-meta">{case["age"]} · {case["sex"]} · {case["body_location"]} · {case["study_type"]} · Study date: {case["submitted"]}</div>'
        f'<div class="status">Status: {case["status"]}</div></div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, .9], gap="large")
    with left:
        st.markdown("### Imaging")
        if case.get("image_bytes"):
            st.markdown('<div class="image-stage">', unsafe_allow_html=True)
            st.image(case["image_bytes"], width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No image file is attached to this case.")
        st.caption("X-ray submitted by the user.")
    with right:
        render_ai_panel(case)

    if case["status"] == STATUS_REVIEWED:
        st.markdown("### Clinical review")
        st.markdown(f"**Assessment**\n\n{case.get('assessment') or 'Not recorded'}")
        st.markdown(f"**Clinical notes**\n\n{case.get('comments') or 'Not recorded'}")
        st.markdown(f"**Final impression**\n\n{case.get('final_impression') or 'Not recorded'}")
        st.markdown(f"**Recommendation**\n\n{case.get('recommendation') or 'Not recorded'}")
        st.markdown(f"**Reviewed by:** {case.get('reviewed_by') or 'Not recorded'}")
        st.markdown(f"**Review date:** {case.get('review_date') or 'Not recorded'}")
        return

    st.markdown("### Doctor review")
    assessment = st.text_area("Assessment", value=case.get("assessment") or "", height=90, key=f"review-assessment-{case_id}")
    comments = st.text_area("Clinical notes", value=case.get("comments") or "", height=130, key=f"review-notes-{case_id}")
    final_impression = st.text_area("Final impression", value=case.get("final_impression") or "", height=100, key=f"review-impression-{case_id}")
    recommendation = st.text_area("Recommendation / next step", value=case.get("recommendation") or "", height=90, key=f"review-recommendation-{case_id}")

    if comments.strip():
        extracted = extract_notes_locally(comments)
        with st.expander("Documentation extraction preview", expanded=True):
            st.caption("Local placeholder extraction for documentation support only.")
            st.markdown(f"**Location:** {extracted.get('location') or 'Not identified'}")
            st.markdown(f"**Finding:** {extracted.get('finding') or 'Not identified'}")
            st.markdown(f"**Recommendation:** {extracted.get('recommendation') or 'Not identified'}")

    if st.button("Submit review", type="primary", key=f"submit-review-{case_id}"):
        if not assessment.strip() or not comments.strip() or not final_impression.strip() or not recommendation.strip():
            st.error("Please complete the assessment, clinical notes, final impression, and recommendation before submitting.")
        else:
            case["assessment"] = assessment.strip()
            case["comments"] = comments.strip()
            case["final_impression"] = final_impression.strip()
            case["recommendation"] = recommendation.strip()
            case["review_date"] = datetime.now().strftime("%d %b %Y")
            case["status"] = STATUS_REVIEWED
            case["reviewed_by"] = st.session_state.user_name
            st.session_state.review_success_case = case["id"]
            st.session_state.selected_case = None
            st.session_state.doctor_section = "Reviewed Cases"
            st.rerun()

def main():
    configure_page()
    initialize_state()

    if not st.session_state.authenticated:
        render_login()
        return

    if st.session_state.role == "User":
        with st.sidebar:
            st.markdown("## Bonewise")
            st.markdown(f"**{st.session_state.user_name}**")
            st.markdown(f'<span class="role-chip">{st.session_state.role}</span>', unsafe_allow_html=True)
            st.markdown("---")
            st.caption("No diagnosis is made by this prototype.")
            st.markdown("---")
            render_reset_cases("user")

            if st.button("Sign out", key="user-sign-out"):
                st.session_state.authenticated = False
                st.session_state.role = None
                st.session_state.user_name = None
                st.session_state.user_email = None
                st.session_state.selected_case = None
                st.session_state.open_user_case = None
                st.rerun()
        render_user()
    else:
        render_doctor()


if __name__ == "__main__":
    main()
