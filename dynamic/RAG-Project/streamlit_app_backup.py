import streamlit as st

from loaders import load_pdfs
from splitter import split_documents
from embeddings_store import create_vectorstore
from qa_chain import build_qa_chain

from employee_data import get_employee_by_id, signup_employee_excel
from auth_db import create_user, verify_user
from employee_data import get_employee_by_id, signup_employee_excel


st.set_page_config(
    page_title="Personalized Leave Policy Assistant",
    layout="wide"
)


@st.cache_resource
def load_chain():
    docs = load_pdfs([
        "data/leave.pdf"  # change to your actual policy PDF name
    ])

    split_docs = split_documents(docs)
    vectorstore = create_vectorstore(split_docs)
    qa_chain = build_qa_chain(vectorstore)

    return qa_chain


st.title("Personalized Leave Policy Assistant")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "employee_data" not in st.session_state:
    st.session_state.employee_data = None

if "messages" not in st.session_state:
    st.session_state.messages = []


menu = st.sidebar.radio("Choose Option", ["Login", "Sign Up"])


if not st.session_state.logged_in:

    if menu == "Login":
        st.sidebar.header("Employee Login")

        employee_id = st.sidebar.text_input("Employee ID")
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            is_valid = verify_user(employee_id, password)

            if not is_valid:
                st.sidebar.error("Invalid Employee ID or Password")
                st.stop()

            employee_data = get_employee_by_id(employee_id)

            if employee_data is None:
                st.sidebar.error("Employee data not found in Excel.")
                st.stop()

            st.session_state.logged_in = True
            st.session_state.employee_data = employee_data
            st.session_state.messages = []
            st.rerun()

        st.info("Please login using Employee ID and Password.")
        st.stop()

    if menu == "Sign Up":
        st.sidebar.header("Employee Sign Up")

        fetch_id = st.sidebar.text_input("Enter Employee ID to Fetch", key="fetch_id")

        if st.sidebar.button("Fetch Employee Data"):
            existing_employee = get_employee_by_id(fetch_id)

            if existing_employee is None:
                st.sidebar.warning("Employee not found in Excel. You can create new signup.")
            else:
                st.session_state.signup_id = existing_employee["employee_id"]
                st.session_state.signup_name = existing_employee["name"]
                st.session_state.signup_grade = existing_employee["grade"]
                st.session_state.signup_pl = int(existing_employee["PL_taken"])
                st.session_state.signup_cl = int(existing_employee["CL_taken"])
                st.session_state.signup_sl = int(existing_employee["SL_taken"])
                st.sidebar.success("Employee data fetched from Excel.")

        with st.sidebar.form("signup_form"):
            new_employee_id = st.text_input(
                "New Employee ID",
                value=st.session_state.get("signup_id", "")
            )

            new_password = st.text_input(
                "Create Password",
                type="password"
            )

            name = st.text_input(
                "Full Name",
                value=st.session_state.get("signup_name", "")
            )

            grade = st.text_input(
                "Grade/Band",
                value=st.session_state.get("signup_grade", "")
            )

            joining_date = st.date_input("Joining Date")

            pl_taken = st.number_input(
                "PL Taken",
                min_value=0,
                step=1,
                value=st.session_state.get("signup_pl", 0)
            )

            cl_taken = st.number_input(
                "CL Taken",
                min_value=0,
                step=1,
                value=st.session_state.get("signup_cl", 0)
            )

            sl_taken = st.number_input(
                "SL Taken",
                min_value=0,
                step=1,
                value=st.session_state.get("signup_sl", 0)
            )

            submitted = st.form_submit_button("Sign Up")

        if submitted:
            new_employee_id = str(new_employee_id).strip()
            new_password = str(new_password).strip()
            name = str(name).strip()
            grade = str(grade).strip()
            joining_date = str(joining_date)

            if new_employee_id == "":
                st.sidebar.error("Employee ID is required.")
                st.stop()

            if new_password == "":
                st.sidebar.error("Password is required.")
                st.stop()

            if name == "":
                st.sidebar.error("Full Name is required.")
                st.stop()

            if grade == "":
                st.sidebar.error("Grade/Band is required.")
                st.stop()

            auth_success, auth_message = create_user(
                new_employee_id,
                new_password
            )

            if not auth_success:
                st.sidebar.error(auth_message)
                st.stop()

            excel_success, excel_message = signup_employee_excel(
                new_employee_id,
                name,
                grade,
                joining_date,
                pl_taken,
                cl_taken,
                sl_taken
            )

            if not excel_success:
                st.sidebar.error(excel_message)
                st.stop()

            st.sidebar.success("Signup successful. Please login.")

        st.info("Create your account from the sidebar.")
        st.stop()
employee_data = st.session_state.employee_data

st.sidebar.success("Logged in")
st.sidebar.write("Name:", employee_data["name"])
st.sidebar.write("Employee ID:", employee_data["employee_id"])
st.sidebar.write("Grade/Band:", employee_data["grade"])
st.sidebar.write("PL Taken:", employee_data["PL_taken"])
st.sidebar.write("CL Taken:", employee_data["CL_taken"])
st.sidebar.write("SL Taken:", employee_data["SL_taken"])

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.employee_data = None
    st.session_state.messages = []
    st.rerun()


qa_chain = load_chain()

st.subheader(f"Welcome, {employee_data['name']}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


query = st.chat_input("Ask your leave question...")

if query:
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.write(query)

    full_query = f"""
Employee Information:
Employee ID: {employee_data["employee_id"]}
Name: {employee_data["name"]}
Grade/Band: {employee_data["grade"]}
PL Taken: {employee_data["PL_taken"]}
CL Taken: {employee_data["CL_taken"]}
SL Taken: {employee_data["SL_taken"]}

User Question:
{query}
"""

    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            response = qa_chain.invoke({"query": full_query})
            answer = response["result"].replace("**", "")
            st.write(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })