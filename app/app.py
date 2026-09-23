from datetime import datetime
from pathlib import Path
import re

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    return [
        {
            "id": "BW-2401",
            "client_name": "Priya Shah",
            "client_id": "BW-1042",
            "age": 34,
            "sex": "Female",
            "body_location": "Left tibia",
            "study_type": "AP view",
            "date_added": "18 Sep 2026",
            "imaging_status": "Study available",
            "submitted": "21 Sep 2026",
            "status": "Pending doctor review",
            "image": demo_image_path(),
            "image_bytes": None,
            "assessment": None,
            "comments": None,
            "final_impression": None,
            "recommendation": None,
            "review_date": None,
            "reviewed_by": None,
        },
        {
            "id": "BW-2398",
            "client_name": "Kavya Iyer",
            "client_id": "BW-1035",
            "age": 29,
            "sex": "Female",
            "body_location": "Left wrist",
            "study_type": "AP view",
            "date_added": "12 Sep 2026",
            "imaging_status": "Study available",
            "submitted": "19 Sep 2026",
            "status": "Reviewed",
            "image": None,
            "image_bytes": None,
            "assessment": "Further clinical correlation recommended",
            "comments": "This is a demonstration response. Please discuss any concerns with a qualified professional.",
            "final_impression": "Further clinical correlation recommended",
            "recommendation": "Discuss with the treating clinician.",
            "review_date": "20 Sep 2026",
            "reviewed_by": "Dr. Meera Rao",
        },
        {
            "id": "BW-2394",
            "client_name": "Arjun Mehta",
            "client_id": "BW-1039",
            "age": 47,
            "sex": "Male",
            "body_location": "Right femur",
            "study_type": "AP + lateral",
            "date_added": "16 Sep 2026",
            "imaging_status": "Study available",
            "submitted": "16 Sep 2026",
            "status": "Awaiting review",
            "image": demo_image_path(),
            "image_bytes": None,
            "assessment": None,
            "comments": None,
            "final_impression": None,
            "recommendation": None,
            "review_date": None,
            "reviewed_by": None,
        },
    ]


def initialize_state():
    if "cases" not in st.session_state:
        st.session_state.cases = make_demo_cases()
    if "next_case_number" not in st.session_state:
        st.session_state.next_case_number = 2402
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "role" not in st.session_state:
        st.session_state.role = None
    if "user_name" not in st.session_state:
        st.session_state.user_name = None
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
    reviewed = case["status"] == "Reviewed"
    rows = [
        ("X-ray submitted", True),
        ("AI-assisted analysis", True),
        ("Doctor review", reviewed),
        ("Doctor response", reviewed),
    ]
    st.markdown(
        '<div class="timeline">'
        + "".join(
            f'<div class="timeline-row {"done" if done else ""}">'
            f'{"✓" if done else "○"} &nbsp;{label}'
            f'{"<br><span class=\"muted\">Pending</span>" if not done else ""}</div>'
            for label, done in rows
        )
        + "</div>",
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
            if case["status"] == "Reviewed":
                clicked = st.button("Open", key=f"{key_prefix}-{row_index}-{case['id']}")
        else:
            clicked = st.button("Open", key=f"{key_prefix}-{row_index}-{case['id']}")
    st.markdown("</div>", unsafe_allow_html=True)
    return clicked


def render_user():
    st.markdown('<div class="eyebrow">Bonewise / patient view</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top:4px'>Study status</h1>", unsafe_allow_html=True)
    st.markdown(
        '<p class="lede">View submitted studies and clinician review status.</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="disclaimer">✦ &nbsp;' + DISCLAIMER + "</div>", unsafe_allow_html=True)
    st.markdown(f"### Good to see you, {st.session_state.user_name.split()[0]}")

    pending = next((case for case in reversed(st.session_state.cases) if case["status"] != "Reviewed"), None)
    if pending:
        st.markdown('<div class="paper"><div class="eyebrow">Current case</div>', unsafe_allow_html=True)
        st.markdown(f"### {pending['id']}")
        st.markdown(f"<span class='muted'>Submitted {pending['submitted']}</span>", unsafe_allow_html=True)
        case_timeline(pending)
        st.info("Your case is waiting for professional review. You will see the doctor's response here once it's ready.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("You do not have a case waiting for review.")

    st.markdown("## Submit an X-ray")
    st.caption("PNG, JPG, and JPEG files are supported.")
    uploader_key = f"xray-upload-{st.session_state.uploader_version}"
    upload = st.file_uploader("Choose an X-ray image", type=["png", "jpg", "jpeg"], label_visibility="collapsed", key=uploader_key)
    if upload:
        st.image(upload, caption="Selected X-ray", width=420)
        if st.button("Submit X-ray", type="primary", key="submit-xray"):
            case_id = next_case_id()
            st.session_state.cases.append(
                {
                    "id": case_id,
                    "client_name": st.session_state.user_name,
                    "client_id": case_id,
                    "age": "Not recorded",
                    "sex": "Not recorded",
                    "body_location": "Not specified",
                    "study_type": "Uploaded X-ray",
                    "date_added": datetime.now().strftime("%d %b %Y"),
                    "imaging_status": "Study available",
                    "submitted": datetime.now().strftime("%d %b %Y"),
                    "status": "Pending doctor review",
                    "image": None,
                    "image_bytes": upload.getvalue(),
                    "assessment": None,
                    "comments": None,
                    "final_impression": None,
                    "recommendation": None,
                    "review_date": None,
                    "reviewed_by": None,
                }
            )
            st.session_state.uploader_version += 1
            st.success(f"Your X-ray has been received. Case ID: {case_id}. You can track its status below.")
            st.rerun()

    st.markdown("## Previous cases")
    other_cases = [case for case in reversed(st.session_state.cases) if not (pending and case["id"] == pending["id"])]
    if not other_cases:
        st.caption("No previous cases yet.")
    for index, case in enumerate(other_cases):
        if render_case_row(case, show_open_for_user=True, key_prefix="open-user", row_index=index):
            st.session_state.open_user_case = case["id"]
            st.rerun()

    if st.session_state.open_user_case:
        opened = next((case for case in st.session_state.cases if case["id"] == st.session_state.open_user_case), None)
        if opened:
            st.markdown(f"### Case {opened['id']}")
            render_doctor_report(opened)
            if st.button("Close", key="close-user-case"):
                st.session_state.open_user_case = None
                st.rerun()

    render_health()


def render_doctor():
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
        st.markdown('<div class="section-kicker">My clients</div>', unsafe_allow_html=True)
        if st.button("All clients", key="doctor-nav-all-clients", width="stretch"):
            st.session_state.doctor_section = "My Clients"
            st.session_state.selected_client = None
            st.rerun()
        if st.button("New clients", key="doctor-nav-new-clients", width="stretch"):
            st.session_state.doctor_section = "New Clients"
            st.session_state.selected_client = None
            st.rerun()
        st.markdown('<div class="section-kicker">Case review</div>', unsafe_allow_html=True)
        if st.button("Review cases", key="doctor-nav-review-cases", width="stretch"):
            st.session_state.doctor_section = "Review Cases"
            st.session_state.selected_report = None
            st.rerun()
        if st.button("Reviewed cases", key="doctor-nav-reviewed-cases", width="stretch"):
            st.session_state.doctor_section = "Reviewed Cases"
            st.rerun()
        st.markdown('<div class="section-kicker">Account</div>', unsafe_allow_html=True)
        st.markdown(f"**{st.session_state.user_name}**<br><span class='muted'>Clinician</span>", unsafe_allow_html=True)
        if st.button("Sign out", key="doctor-sign-out", width="stretch"):
            st.session_state.authenticated = False
            st.session_state.role = None
            st.session_state.user_name = None
            st.session_state.selected_case = None
            st.session_state.selected_client = None
            st.session_state.selected_report = None
            st.rerun()

    if st.session_state.selected_client:
        render_client_profile(st.session_state.selected_client)
        return
    if st.session_state.doctor_section == "Overview":
        render_doctor_overview()
    elif st.session_state.doctor_section == "My Clients":
        render_client_registry()
    elif st.session_state.doctor_section == "New Clients":
        render_new_clients()
    elif st.session_state.doctor_section == "Reviewed Cases":
        render_reviewed_cases()
    else:
        render_review_queue()


def render_doctor_overview():
    cases = st.session_state.cases
    open_cases = [case for case in cases if case["status"] != "Reviewed"]
    reviewed_today = [case for case in cases if case.get("review_date") == datetime.now().strftime("%d %b %Y")]
    clients = {case["client_id"] for case in cases}
    st.markdown('<div class="eyebrow">Bonewise / clinician overview</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top:4px'>Clinical overview</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lede'>Current patient and study activity for this review workspace.</p>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="stat-row"><div class="stat-cell"><div class="stat-label">Active patients</div><div class="metric">{len(clients)}</div></div>'
        f'<div class="stat-cell"><div class="stat-label">New cases</div><div class="metric">{sum(case["status"] == "Pending doctor review" for case in cases)}</div></div>'
        f'<div class="stat-cell"><div class="stat-label">Awaiting review</div><div class="metric">{len(open_cases)}</div></div>'
        f'<div class="stat-cell"><div class="stat-label">Reviewed today</div><div class="metric">{len(reviewed_today)}</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### Recent studies")
    recent = sorted(cases, key=lambda case: case["submitted"], reverse=True)[:5]
    render_worklist_header(["Patient", "Study ID", "Region", "Study date", "Status", "Action"])
    for index, case in enumerate(recent):
        st.markdown(
            f'<div class="worklist-row"><span><strong>{case["client_name"]}</strong></span><span class="case-id">{case["id"]}</span>'
            f'<span>{case["body_location"]}</span><span>{case["submitted"]}</span><span class="status-dot {"reviewed" if case["status"] == "Reviewed" else ""}">{case["status"]}</span><span></span></div>',
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
    cases = [case for case in st.session_state.cases if case["status"] != "Reviewed"]
    st.markdown('<div class="eyebrow">Case review / clinical queue</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top:4px'>Review cases</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lede'>Cases requiring clinician attention. Open one case at a time for review.</p>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="stat-row"><div class="stat-cell"><div class="stat-label">Open cases</div><div class="metric">{len(cases)}</div></div>'
        f'<div class="stat-cell"><div class="stat-label">Reviewed cases</div><div class="metric">{len(st.session_state.cases) - len(cases)}</div></div></div>',
        unsafe_allow_html=True,
    )
    if not cases:
        st.info("There are no cases waiting for review.")
        return
    render_worklist_header(["Client", "Case ID", "Study", "Study date", "Status", "Action"])
    for index, case in enumerate(cases):
        st.markdown(
            f'<div class="worklist-row"><span><strong>{case["client_name"]}</strong><br><span class="muted">{case["age"]} · {case["sex"]}</span></span>'
            f'<span class="case-id">{case["client_id"]}</span><span>{case["body_location"]}<br><span class="muted">{case["study_type"]}</span></span>'
            f'<span>{case["submitted"]}</span><span class="status-dot">{case["status"]}</span><span></span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open review", key=f"review-queue-open-{index}-{case['id']}", type="primary"):
            st.session_state.selected_case = case["id"]
            st.rerun()
    render_worklist_footer()


def render_client_registry():
    query = st.text_input("Search clients", placeholder="Search by name or client ID", key="client-search")
    clients = {}
    for case in st.session_state.cases:
        clients.setdefault(case["client_id"], case)
    filtered = [case for case in clients.values() if query.lower() in f"{case['client_name']} {case['client_id']}".lower()]
    st.markdown('<div class="eyebrow">My clients / registry</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top:4px'>My clients</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lede'>Clients assigned to this clinical workspace and their latest available study.</p>", unsafe_allow_html=True)
    if not filtered:
        st.info("No clients match that search.")
        return
    render_worklist_header(["Client", "Client ID", "Study", "Last study", "Status", "Action"])
    for index, case in enumerate(filtered):
        st.markdown(
            f'<div class="worklist-row"><span><strong>{case["client_name"]}</strong></span><span class="case-id">{case["client_id"]}</span>'
            f'<span>{case["body_location"]}<br><span class="muted">{case["study_type"]}</span></span><span>{case["submitted"]}</span>'
            f'<span class="status-dot {"reviewed" if case["status"] == "Reviewed" else ""}">{case["status"]}</span><span></span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open client", key=f"client-registry-open-{index}-{case['client_id']}"):
            st.session_state.selected_client = case["client_id"]
            st.rerun()
    render_worklist_footer()


def render_new_clients():
    cases = [case for case in st.session_state.cases if case["status"] == "Pending doctor review"]
    st.markdown('<div class="eyebrow">My clients / recent arrivals</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top:4px'>New clients</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lede'>Recently added studies that have not yet received a clinical review.</p>", unsafe_allow_html=True)
    if not cases:
        st.info("There are no new clients right now.")
        return
    render_worklist_header(["Client", "Case ID", "Date added", "Imaging", "Review status", "Action"])
    for index, case in enumerate(cases):
        st.markdown(
            f'<div class="worklist-row"><span><strong>{case["client_name"]}</strong></span><span class="case-id">{case["client_id"]}</span>'
            f'<span>{case["date_added"]}</span><span>{case["imaging_status"]}</span><span class="status-dot">New</span><span></span></div>',
            unsafe_allow_html=True,
        )
        if st.button("Open case", key=f"new-client-open-{index}-{case['id']}", type="primary"):
            st.session_state.selected_case = case["id"]
            st.rerun()
    render_worklist_footer()


def render_client_profile(client_id):
    cases = [case for case in st.session_state.cases if case["client_id"] == client_id]
    if not cases:
        st.session_state.selected_client = None
        return
    client = cases[0]
    if st.button("← Back to clients", key="back-to-client-registry"):
        st.session_state.selected_client = None
        st.rerun()
    st.markdown('<div class="eyebrow">My clients / profile</div>', unsafe_allow_html=True)
    st.markdown(f"<h1 style='margin-top:4px'>{client['client_name']}</h1><span class='case-id'>{client['client_id']}</span>", unsafe_allow_html=True)
    st.markdown(f"<p class='case-meta'>{client['age']} years · {client['sex']}</p>", unsafe_allow_html=True)
    st.markdown("### Imaging and review history")
    for index, case in enumerate(sorted(cases, key=lambda item: item["submitted"], reverse=True)):
        left, right = st.columns([5, 1])
        with left:
            st.markdown(f'<div class="profile-row"><span><strong>{case["submitted"]}</strong><br>{case["body_location"]} · {case["study_type"]}</span><span class="status">{case["status"]}</span></div>', unsafe_allow_html=True)
        with right:
            if st.button("Open", key=f"profile-study-open-{index}-{case['id']}"):
                st.session_state.selected_case = case["id"]
                st.rerun()


def render_reviewed_cases():
    cases = [case for case in st.session_state.cases if case["status"] == "Reviewed"]
    st.markdown('<div class="eyebrow">Case review / completed records</div>', unsafe_allow_html=True)
    st.markdown("<h1 style='margin-top:4px'>Reviewed cases</h1>", unsafe_allow_html=True)
    st.markdown("<p class='lede'>Completed reviews and their clinical reports.</p>", unsafe_allow_html=True)
    if not cases:
        st.info("No completed reviews yet.")
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
    st.markdown("<div class='eyebrow'>Bonewise</div><h1>Clinical review report</h1>", unsafe_allow_html=True)
    st.markdown(f"**Client**  \n{case['client_name']}  \n\n**Case ID**  \n{case['client_id']}  \n\n**Study**  \n{case['body_location']} · {case['study_type']}  \n\n**Study date**  \n{case['submitted']}")
    st.markdown("## Imaging")
    st.markdown("The reviewed X-ray is retained in the case workspace.")
    st.markdown("## AI-assisted analysis")
    st.info("Model status: Not connected. No model output or prediction is available in this frontend prototype.")
    st.markdown("EfficientNet-B0: **Pending model integration**  \nResNet50: **Pending model integration**  \nThird model: **Planned**")
    st.markdown(f"## Doctor assessment\n{case['assessment'] or 'Not recorded'}\n\n## Clinical notes\n{case['comments'] or 'Not recorded'}")
    st.markdown(f"## Final impression\n<div class='final-impression'>{case.get('final_impression') or case['assessment'] or 'Not recorded'}</div>", unsafe_allow_html=True)
    st.markdown(f"## Recommendation\n{case.get('recommendation') or 'Not recorded'}")
    st.markdown(f"## Review information\n**Reviewed by:** {case['reviewed_by']}  \n**Review date:** {case['review_date']}")
    st.markdown("</div>", unsafe_allow_html=True)


def render_ai_panel():
    st.markdown(
        f"""
        <div class="analysis-block">
            <div class="eyebrow">AI-assisted analysis</div>
            <div class="analysis-label">Model status</div>
            <div class="analysis-value">Not connected</div>
            <div class="analysis-label">EfficientNet-B0</div>
            <div class="analysis-value">Pending model integration</div>
            <div class="analysis-label">ResNet50</div>
            <div class="analysis-value">Pending model integration</div>
            <div class="analysis-label">Third model</div>
            <div class="analysis-value">Planned</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Model outputs will appear here once inference is connected.")


def render_further_analysis(case):
    st.markdown("### Further analysis")
    st.caption("Similar-case retrieval will be available after a model connection is added.")


def render_doctor_report_placeholder():
    pass


def render_case_review(case_id):
    case = next((case for case in st.session_state.cases if case["id"] == case_id), None)
    if case is None:
        st.session_state.selected_case = None
        return

    if st.button("← Back to review cases", key=f"back-to-review-{case_id}"):
        st.session_state.selected_case = None
        st.rerun()

    st.markdown(
        f'<div class="case-header"><div class="eyebrow">Clinical review workspace</div>'
        f'<h1 style="margin-top:4px">{case["client_name"]}</h1>'
        f'<span class="case-id">{case["client_id"]} · Case {case["id"]}</span>'
        f'<div class="case-meta">{case["age"]}{case["sex"][0]} · {case["body_location"]} · {case["study_type"]} · Study date: {case["submitted"]}</div>'
        f'<div class="status">Status: {case["status"]}</div></div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([1.25, .9], gap="large")
    with left:
        st.markdown("### Imaging")
        if case["image_bytes"]:
            st.markdown('<div class="image-stage">', unsafe_allow_html=True)
            st.image(case["image_bytes"], width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        elif case["image"] and Path(case["image"]).exists():
            st.markdown('<div class="image-stage">', unsafe_allow_html=True)
            st.image(case["image"], width="stretch")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No image file is attached to this demonstration case.")
        zoom, fit, reset = st.columns(3)
        with zoom:
            st.button("Zoom", key=f"review-zoom-{case_id}")
        with fit:
            st.button("Fit to screen", key=f"review-fit-{case_id}")
        with reset:
            st.button("Reset", key=f"review-reset-{case_id}")
        st.toggle("Grad-CAM / attention", key=f"review-gradcam-{case_id}")
        st.caption("Attention map unavailable until model integration.")
        st.caption("Viewer controls are frontend placeholders until an imaging viewer is connected.")
    with right:
        render_ai_panel()

    st.markdown("### Doctor review")
    assessment = st.text_area("Assessment", value=case["assessment"] or "", height=90, key=f"review-assessment-{case_id}")
    comments = st.text_area("Clinical notes", value=case["comments"] or "", height=130, key=f"review-notes-{case_id}")
    final_impression = st.text_area("Final impression", value=case.get("final_impression") or "", height=100, key=f"review-impression-{case_id}")
    recommendation = st.text_area("Recommendation / next step", value=case.get("recommendation") or "", height=90, key=f"review-recommendation-{case_id}")
    if comments.strip():
        extracted = extract_notes_locally(comments)
        with st.expander("Documentation extraction preview", expanded=True):
            st.caption("Local placeholder extraction for documentation support only.")
            st.markdown(f"**Location:** {extracted.get('location') or 'Not identified'}")
            st.markdown(f"**Finding:** {extracted.get('finding') or 'Not identified'}")
            st.markdown(f"**Recommendation:** {extracted.get('recommendation') or 'Not identified'}")
    st.markdown('<div class="final-impression"><strong>Final impression</strong><br>The clinician-authored conclusion will be shown most prominently in the completed report.</div>', unsafe_allow_html=True)
    if st.button("Submit review", type="primary", key=f"submit-review-{case_id}"):
        if not assessment.strip() or not comments.strip() or not final_impression.strip() or not recommendation.strip():
            st.error("Please complete the assessment, clinical notes, final impression, and recommendation before submitting.")
        else:
            case["assessment"] = assessment.strip()
            case["comments"] = comments.strip()
            case["final_impression"] = final_impression.strip()
            case["recommendation"] = recommendation.strip()
            case["review_date"] = datetime.now().strftime("%d %b %Y")
            case["status"] = "Reviewed"
            case["reviewed_by"] = st.session_state.user_name
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
            st.caption("Session-only demo data")
            st.caption("No diagnosis is made by this prototype.")
            st.markdown("---")
            if st.button("Sign out", key="user-sign-out"):
                st.session_state.authenticated = False
                st.session_state.role = None
                st.session_state.user_name = None
                st.session_state.selected_case = None
                st.session_state.open_user_case = None
                st.rerun()
        render_user()
    else:
        render_doctor()


if __name__ == "__main__":
    main()
