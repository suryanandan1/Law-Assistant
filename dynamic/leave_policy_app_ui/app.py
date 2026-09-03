import streamlit as st

from loaders import load_pdfs
from splitter import split_documents
from embeddings_store import create_vectorstore
from qa_chain import build_qa_chain

from employee_data import get_employee_by_id, signup_employee_excel
from auth_db import create_user, verify_user


# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Leave Policy Assistant",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@600;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, .stApp {
    background: #0f1117 !important;
    font-family: 'Inter', sans-serif;
    color: #e2e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1100px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2535 !important;
}
[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.25rem !important;
}

/* ── Logo / App Header ── */
.app-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #1e2535;
}
.app-logo .icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.app-logo .name {
    font-family: 'Sora', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
}
.app-logo .tagline {
    font-size: 11px;
    color: #64748b;
    margin-top: 2px;
}

/* ── Sidebar section labels ── */
.sidebar-label {
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #475569;
    text-transform: uppercase;
    margin: 1.25rem 0 0.5rem;
}

/* ── Profile card in sidebar ── */
.profile-card {
    background: #1e2535;
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.profile-name {
    font-family: 'Sora', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 2px;
}
.profile-id {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 10px;
}
.profile-badge {
    display: inline-block;
    background: #1a2744;
    border: 1px solid #2563eb44;
    color: #60a5fa;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
}

/* ── Leave balance pills ── */
.leave-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin-top: 10px;
}
.leave-pill {
    background: #0f1117;
    border: 1px solid #1e2535;
    border-radius: 8px;
    padding: 8px 6px;
    text-align: center;
}
.leave-pill .ltype {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: #64748b;
    text-transform: uppercase;
}
.leave-pill .lval {
    font-size: 20px;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.2;
}
.leave-pill .lsub {
    font-size: 9px;
    color: #475569;
}

/* ── Main Page Header ── */
.main-header {
    margin-bottom: 2rem;
}
.main-header .greeting {
    font-size: 13px;
    color: #64748b;
    font-weight: 500;
    margin-bottom: 4px;
}
.main-header h1 {
    font-family: 'Sora', sans-serif;
    font-size: 28px;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 6px;
    line-height: 1.2;
}
.main-header .subtitle {
    font-size: 14px;
    color: #64748b;
}

/* ── Chat area ── */
.chat-container {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 16px;
    padding: 1.5rem;
    min-height: 420px;
    margin-bottom: 1rem;
}

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 3rem 1rem;
    text-align: center;
}
.empty-icon {
    width: 56px; height: 56px;
    background: linear-gradient(135deg, #6366f144, #8b5cf644);
    border: 1px solid #6366f133;
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    margin-bottom: 1rem;
}
.empty-state h3 {
    font-family: 'Sora', sans-serif;
    font-size: 17px;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 8px;
}
.empty-state p {
    font-size: 13.5px;
    color: #64748b;
    max-width: 380px;
    line-height: 1.6;
    margin: 0;
}

/* ── Suggestion chips ── */
.chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
    margin-top: 1.25rem;
}
.chip {
    background: #1e2535;
    border: 1px solid #2d3748;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 12px;
    color: #94a3b8;
    cursor: pointer;
    transition: all 0.15s;
}
.chip:hover { border-color: #6366f1; color: #a5b4fc; }

/* ── Message bubbles ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 1rem !important;
}
[data-testid="stChatMessage"][data-testid*="user"] .stMarkdown {
    background: linear-gradient(135deg, #312e81, #1e1b4b) !important;
    border: 1px solid #4338ca44 !important;
}

/* ── Input bar ── */
[data-testid="stChatInput"] {
    border: 1px solid #2d3748 !important;
    background: #161b27 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px #6366f122 !important;
}

/* ── Streamlit inputs / selects ── */
.stTextInput input,
.stSelectbox select,
.stNumberInput input,
[data-baseweb="input"] input,
[data-baseweb="select"] {
    background: #1e2535 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus,
[data-baseweb="input"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px #6366f133 !important;
}
label, .stTextInput label, .stNumberInput label {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.25rem !important;
    transition: opacity 0.15s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── Radio ── */
.stRadio label { color: #94a3b8 !important; font-size: 13.5px !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #94a3b8 !important; }

/* ── Alerts ── */
.stSuccess { background: #14291f !important; border-color: #16a34a44 !important; color: #4ade80 !important; border-radius: 8px !important; }
.stError   { background: #2a1215 !important; border-color: #dc262644 !important; color: #f87171 !important; border-radius: 8px !important; }
.stInfo    { background: #172035 !important; border-color: #3b82f644 !important; color: #93c5fd !important; border-radius: 8px !important; }
.stWarning { background: #29200e !important; border-color: #d9770644 !important; color: #fbbf24 !important; border-radius: 8px !important; }

/* ── Divider ── */
hr { border-color: #1e2535 !important; margin: 1rem 0 !important; }

/* ── Tab styling for auth pages ── */
.auth-card {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 16px;
    padding: 2rem;
    max-width: 420px;
    margin: 3rem auto 0;
}
.auth-card h2 {
    font-family: 'Sora', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 0.25rem;
}
.auth-card .auth-sub {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ─── QA Chain Loader ────────────────────────────────────────────────────────
@st.cache_resource
def load_chain():
    docs = load_pdfs(["data/leave.pdf"])
    split_docs = split_documents(docs)
    vectorstore = create_vectorstore(split_docs)
    return build_qa_chain(vectorstore)


# ─── Session State Defaults ─────────────────────────────────────────────────
for key, val in {
    "logged_in": False,
    "employee_data": None,
    "messages": [],
    "signup_id": "",
    "signup_name": "",
    "signup_grade": "",
    "signup_pl": 0,
    "signup_cl": 0,
    "signup_sl": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ═══════════════════════════════════════════════════════════════════════════
#  AUTH SCREENS
# ═══════════════════════════════════════════════════════════════════════════
if not st.session_state.logged_in:

    # Sidebar logo only
    with st.sidebar:
        st.markdown("""
        <div class="app-logo">
            <div class="icon">🗓️</div>
            <div>
                <div class="name">LeaveIQ</div>
                <div class="tagline">Powered by AI</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        menu = st.radio("", ["Login", "Sign Up"], label_visibility="collapsed")

    # ── LOGIN ──
    if menu == "Login":
        st.markdown("""
        <div style="text-align:center; margin: 2rem 0 0.5rem;">
            <div style="font-size:40px; margin-bottom:0.75rem;">🗓️</div>
            <div style="font-family:'Sora',sans-serif; font-size:26px; font-weight:700; color:#f1f5f9; margin-bottom:6px;">Welcome back</div>
            <div style="font-size:14px; color:#64748b;">Sign in with your Employee ID to continue</div>
        </div>
        """, unsafe_allow_html=True)

        col_l, col_c, col_r = st.columns([1, 1.6, 1])
        with col_c:
            st.markdown('<div style="background:#161b27; border:1px solid #1e2535; border-radius:16px; padding:1.75rem 1.5rem; margin-top:1rem;">', unsafe_allow_html=True)

            employee_id = st.text_input("Employee ID", placeholder="e.g. EMP001")
            password    = st.text_input("Password", type="password", placeholder="Your password")

            if st.button("Sign In →"):
                if not employee_id or not password:
                    st.error("Please enter both Employee ID and password.")
                    st.stop()

                if not verify_user(employee_id, password):
                    st.error("Invalid Employee ID or password.")
                    st.stop()

                employee_data = get_employee_by_id(employee_id)
                if employee_data is None:
                    st.error("Employee record not found. Contact HR.")
                    st.stop()

                st.session_state.logged_in      = True
                st.session_state.employee_data  = employee_data
                st.session_state.messages       = []
                st.rerun()

            st.markdown('<div style="text-align:center; margin-top:0.75rem; font-size:12.5px; color:#475569;">Don\'t have an account? Switch to <b style="color:#818cf8;">Sign Up</b> in the sidebar.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.stop()

    # ── SIGN UP ──
    if menu == "Sign Up":
        st.markdown("""
        <div style="text-align:center; margin: 2rem 0 0.5rem;">
            <div style="font-size:40px; margin-bottom:0.75rem;">✨</div>
            <div style="font-family:'Sora',sans-serif; font-size:26px; font-weight:700; color:#f1f5f9; margin-bottom:6px;">Create your account</div>
            <div style="font-size:14px; color:#64748b;">Use your Employee ID to get started</div>
        </div>
        """, unsafe_allow_html=True)

        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.markdown('<div style="background:#161b27; border:1px solid #1e2535; border-radius:16px; padding:1.75rem 1.5rem; margin-top:1rem;">', unsafe_allow_html=True)

            # Prefill from Excel
            st.markdown('<div style="font-size:12px; font-weight:600; color:#6366f1; margin-bottom:8px; letter-spacing:0.04em;">STEP 1 — LOOK UP YOUR RECORD</div>', unsafe_allow_html=True)
            c1, c2 = st.columns([2, 1])
            with c1:
                fetch_id = st.text_input("Employee ID", placeholder="e.g. EMP001", key="fetch_id_input")
            with c2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Fetch →"):
                    existing = get_employee_by_id(fetch_id)
                    if existing is None:
                        st.warning("Not found — fill in manually below.")
                    else:
                        st.session_state.signup_id    = existing["employee_id"]
                        st.session_state.signup_name  = existing["name"]
                        st.session_state.signup_grade = existing["grade"]
                        st.session_state.signup_pl    = int(existing["PL_taken"])
                        st.session_state.signup_cl    = int(existing["CL_taken"])
                        st.session_state.signup_sl    = int(existing["SL_taken"])
                        st.success("Record found — fields pre-filled.")

            st.markdown('<hr style="margin:1rem 0;"><div style="font-size:12px; font-weight:600; color:#6366f1; margin-bottom:8px; letter-spacing:0.04em;">STEP 2 — COMPLETE YOUR PROFILE</div>', unsafe_allow_html=True)

            with st.form("signup_form", clear_on_submit=False):
                new_id       = st.text_input("Employee ID *", value=st.session_state.signup_id)
                new_password = st.text_input("Create Password *", type="password")
                name         = st.text_input("Full Name *", value=st.session_state.signup_name)
                grade        = st.text_input("Grade / Band *", value=st.session_state.signup_grade)
                joining_date = st.date_input("Joining Date")

                st.markdown('<div style="font-size:12px; color:#64748b; margin:0.5rem 0 0.25rem;">Leaves already taken this year</div>', unsafe_allow_html=True)
                lc1, lc2, lc3 = st.columns(3)
                with lc1: pl = st.number_input("PL", min_value=0, step=1, value=st.session_state.signup_pl)
                with lc2: cl = st.number_input("CL", min_value=0, step=1, value=st.session_state.signup_cl)
                with lc3: sl = st.number_input("SL", min_value=0, step=1, value=st.session_state.signup_sl)

                submitted = st.form_submit_button("Create Account →")

            if submitted:
                new_id = str(new_id).strip()
                if not new_id:       st.error("Employee ID is required."); st.stop()
                if not new_password: st.error("Password is required."); st.stop()
                if not name:         st.error("Full Name is required."); st.stop()
                if not grade:        st.error("Grade / Band is required."); st.stop()

                ok, msg = create_user(new_id, new_password)
                if not ok: st.error(msg); st.stop()

                ok2, msg2 = signup_employee_excel(new_id, name, grade, str(joining_date), pl, cl, sl)
                if not ok2: st.error(msg2); st.stop()

                st.success("Account created! Switch to Login to sign in.")

            st.markdown('</div>', unsafe_allow_html=True)

        st.stop()


# ═══════════════════════════════════════════════════════════════════════════
#  AUTHENTICATED APP
# ═══════════════════════════════════════════════════════════════════════════
emp = st.session_state.employee_data

# ── Sidebar (logged-in) ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="app-logo">
        <div class="icon">🗓️</div>
        <div>
            <div class="name">LeaveIQ</div>
            <div class="tagline">Powered by AI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Profile card
    st.markdown(f"""
    <div class="profile-card">
        <div class="profile-name">{emp['name']}</div>
        <div class="profile-id">ID: {emp['employee_id']}</div>
        <span class="profile-badge">{emp['grade']}</span>
    </div>
    """, unsafe_allow_html=True)

    # Leave balance grid
    st.markdown('<div class="sidebar-label">Leave Balance (Taken)</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="leave-grid">
        <div class="leave-pill">
            <div class="ltype">PL</div>
            <div class="lval">{int(emp['PL_taken'])}</div>
            <div class="lsub">Privilege</div>
        </div>
        <div class="leave-pill">
            <div class="ltype">CL</div>
            <div class="lval">{int(emp['CL_taken'])}</div>
            <div class="lsub">Casual</div>
        </div>
        <div class="leave-pill">
            <div class="ltype">SL</div>
            <div class="lval">{int(emp['SL_taken'])}</div>
            <div class="lsub">Sick</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-label">Session</div>', unsafe_allow_html=True)
    if st.button("Sign Out"):
        for k in ["logged_in", "employee_data", "messages"]:
            st.session_state[k] = False if k == "logged_in" else None if k == "employee_data" else []
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px; color:#334155; text-align:center; padding-top:1rem; border-top:1px solid #1e2535;">Answers are based on your<br>company leave policy PDF</div>', unsafe_allow_html=True)


# ── Main Chat Area ───────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <div class="greeting">Good day,</div>
    <h1>{emp['name'].split()[0]}'s Leave Assistant</h1>
    <div class="subtitle">Ask anything about your leave entitlements, balances, or policy rules.</div>
</div>
""", unsafe_allow_html=True)


# Render existing messages
if st.session_state.messages:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
else:
    # Empty state with suggestion chips
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">💬</div>
        <h3>How can I help you today?</h3>
        <p>Ask me anything about your leave policy — balances, entitlements, carry-forward rules, and more.</p>
        <div class="chips">
            <div class="chip">How many PL days do I have left?</div>
            <div class="chip">Can I carry forward unused leaves?</div>
            <div class="chip">What's the CL policy for my grade?</div>
            <div class="chip">How do I apply for sick leave?</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Load QA Chain ────────────────────────────────────────────────────────
qa_chain = load_chain()


# ── Chat Input ───────────────────────────────────────────────────────────
query = st.chat_input("Ask your leave question…")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    full_query = f"""
Employee Information:
Employee ID: {emp['employee_id']}
Name: {emp['name']}
Grade/Band: {emp['grade']}
PL Taken: {emp['PL_taken']}
CL Taken: {emp['CL_taken']}
SL Taken: {emp['SL_taken']}

User Question:
{query}
"""

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = qa_chain.invoke({"query": full_query})
            answer   = response["result"].replace("**", "")
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
