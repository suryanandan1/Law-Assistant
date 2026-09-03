from django.contrib import messages
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from employee_data import create_or_update_employee, get_employee_by_id
from auth_db import create_user, verify_user
from user_documents import (
    get_assistant_title,
    get_user_chain,
    get_user_documents,
    get_pdf_question_suggestions,
    get_user_pdf_summary,
    save_uploaded_pdf,
    delete_user_pdf,
)


def _get_employee_from_session(request):
    employee_id = request.session.get("employee_id")
    if not employee_id:
        return None
    return get_employee_by_id(employee_id)


def _build_full_query(employee_data, query):
    return f"""
Employee Information:
Employee ID: {employee_data['employee_id']}
Name: {employee_data['name']}
Grade/Band: {employee_data['grade']}
PL Taken: {employee_data['PL_taken']}
CL Taken: {employee_data['CL_taken']}
SL Taken: {employee_data['SL_taken']}
Uploaded PDF Heading(s): {employee_data.get('pdf_heading', '')}

User Question:
{query}
"""


def _get_chat_title(chat_messages):
    for msg in chat_messages:
        if msg.get("role") == "user":
            title = msg.get("content", "").strip()
            return title[:32] + "..." if len(title) > 32 else title
    return "New Chat"


@require_http_methods(["GET", "POST"])
def login_page(request):
    if request.method == "POST":
        employee_id = request.POST.get("employee_id", "").strip()
        password = request.POST.get("password", "").strip()

        if not employee_id or not password:
            messages.error(request, "Employee ID and password are required.")
            return redirect("login")

        if not verify_user(employee_id, password):
            messages.error(request, "Invalid Employee ID or Password.")
            return redirect("login")

        employee_data = get_employee_by_id(employee_id)
        if employee_data is None:
            messages.error(request, "Employee data not found in Excel.")
            return redirect("login")

        request.session["employee_id"] = employee_id
        request.session["chat_messages"] = []
        request.session.setdefault("chat_history", [])
        request.session.pop("opened_history_index", None)
        request.session.modified = True

        return redirect("chat")

    return render(request, "leave_app/login.html")


@require_http_methods(["GET", "POST"])
def signup_page(request):
    if request.method == "POST":
        employee_id = request.POST.get("employee_id", "").strip()
        password = request.POST.get("password", "").strip()
        name = request.POST.get("name", "").strip()

        if not employee_id or not password:
            messages.error(request, "Employee ID and password are required.")
            return redirect("signup")

        auth_success, auth_message = create_user(employee_id, password)
        if not auth_success:
            messages.error(request, auth_message)
            return redirect("signup")

        excel_success, excel_message = create_or_update_employee(employee_id, name)
        if not excel_success:
            messages.error(request, excel_message)
            return redirect("signup")

        messages.success(request, "Signup successful. Please login.")
        return redirect("login")

    return render(request, "leave_app/signup.html")


@require_http_methods(["GET", "POST"])
def chat_page(request):
    employee_data = _get_employee_from_session(request)

    if employee_data is None:
        return redirect("login")

    current_chat = request.session.get("chat_messages", [])
    chat_history = request.session.get("chat_history", [])

    action = request.GET.get("action")
    history_index = request.GET.get("history")

    # NEW CHAT:
    # If current chat is normal chat, save it in history.
    # If current chat is opened from history, do not duplicate it.
    if action == "new_chat":
        opened_index = request.session.get("opened_history_index")

        if current_chat and opened_index is None:
            chat_history.insert(0, {
                "title": _get_chat_title(current_chat),
                "messages": current_chat.copy()
            })

        request.session["chat_history"] = chat_history
        request.session["chat_messages"] = []
        request.session.pop("opened_history_index", None)
        request.session.modified = True

        return redirect("chat")

    # CLEAR CHAT:
    # If current chat is normal chat -> only clear current chat.
    # If current chat is opened history chat -> remove only that chat from history also.
    if action == "clear_chat":
        opened_index = request.session.get("opened_history_index")

        if opened_index is not None:
            try:
                opened_index = int(opened_index)

                if 0 <= opened_index < len(chat_history):
                    chat_history.pop(opened_index)

                request.session["chat_history"] = chat_history

            except Exception:
                pass

        request.session["chat_messages"] = []
        request.session.pop("opened_history_index", None)
        request.session.modified = True

        return redirect("chat")

    # CLEAR ALL HISTORY
    if action == "clear_history":
        request.session["chat_history"] = []
        request.session["chat_messages"] = []
        request.session.pop("opened_history_index", None)
        request.session.modified = True

        return redirect("chat")

    # OPEN HISTORY CHAT
    if history_index is not None:
        try:
            index = int(history_index)
            selected_chat = chat_history[index]

            if isinstance(selected_chat, dict):
                request.session["chat_messages"] = selected_chat.get("messages", [])
            else:
                request.session["chat_messages"] = selected_chat

            request.session["opened_history_index"] = index
            request.session.modified = True

            return redirect("chat")

        except Exception:
            messages.error(request, "Unable to open selected chat.")
            return redirect("chat")

    # UPLOAD PDF
    if request.method == "POST" and request.POST.get("action") == "upload_pdf":
        uploaded_file = request.FILES.get("pdf_file")
        if not uploaded_file:
            messages.error(request, "Please choose a PDF file to upload.")
            return redirect("chat")

        success, message = save_uploaded_pdf(employee_data["employee_id"], uploaded_file)
        if success:
            messages.success(request, message)
            request.session["chat_messages"] = []
            request.session.pop("opened_history_index", None)
        else:
            messages.error(request, message)
        request.session.modified = True
        return redirect("chat")


    # DELETE PDF
    if request.method == "POST" and request.POST.get("action") == "delete_pdf":
        filename = request.POST.get("filename", "").strip()
        success, message = delete_user_pdf(employee_data["employee_id"], filename)
        if success:
            messages.success(request, message)
            request.session["chat_messages"] = []
            request.session.pop("opened_history_index", None)
        else:
            messages.error(request, message)
        request.session.modified = True
        return redirect("chat")

    # ASK QUESTION
    if request.method == "POST":
        query = request.POST.get("query", "").strip()

        if query:
            current_chat.append({
                "role": "user",
                "content": query
            })

            full_query = _build_full_query(employee_data, query)

            try:
                chain = get_user_chain(employee_data["employee_id"])
                if chain is None:
                    answer = "Please upload a PDF first, then ask questions from that PDF."
                else:
                    response = chain.invoke({"query": full_query})
                    answer = response["result"].replace("**", "")

            except Exception as exc:
                answer = f"Error while generating answer: {exc}"

            current_chat.append({
                "role": "assistant",
                "content": answer
            })

            request.session["chat_messages"] = current_chat

            # If this chat came from history, update only that same history item
            opened_index = request.session.get("opened_history_index")
            if opened_index is not None:
                try:
                    opened_index = int(opened_index)

                    if 0 <= opened_index < len(chat_history):
                        chat_history[opened_index] = {
                            "title": _get_chat_title(current_chat),
                            "messages": current_chat.copy()
                        }

                        request.session["chat_history"] = chat_history

                except Exception:
                    pass

            request.session.modified = True

        return redirect("chat")

    total_questions = sum(1 for msg in current_chat if msg.get("role") == "user")
    for history in chat_history:
        messages_list = history.get("messages", []) if isinstance(history, dict) else history
        total_questions += sum(1 for msg in messages_list if msg.get("role") == "user")

    return render(
        request,
        "leave_app/chat.html",
        {
            "employee": employee_data,
            "chat_messages": current_chat,
            "chat_history": chat_history,
            "documents": get_user_documents(employee_data["employee_id"]),
            "assistant_title": get_assistant_title(employee_data["employee_id"]),
            "suggestions": get_pdf_question_suggestions(employee_data["employee_id"]),
            "pdf_summary": get_user_pdf_summary(employee_data["employee_id"]),
            "total_questions": total_questions,
        },
    )


def logout_page(request):
    request.session.flush()
    return redirect("login")