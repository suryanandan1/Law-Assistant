# Dynamic User PDF RAG Assistant

This updated project supports:

- Employee signup and login
- Fetching existing employee data from Excel during signup
- User-specific PDF upload after login
- RAG answers from the logged-in user's uploaded PDF files only
- Automatic assistant heading extracted from the uploaded PDF title / first-page heading
- Employee data included in the query, so questions can use previously stored user data when the uploaded PDF contains matching rules
- Chat history, new chat, clear chat, sidebar, and loader UI

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
DJANGO_SECRET_KEY=change-this-secret-key
DJANGO_DEBUG=True
```

Run the app:

```bash
python manage.py migrate
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## User flow

1. Create account from signup page.
2. If employee data already exists in Excel, enter employee ID and click **Fetch Employee Data**.
3. Login with employee ID and password.
4. Upload any PDF from the sidebar.
5. The page heading changes automatically based on the uploaded PDF title/heading.
6. Ask questions from the uploaded PDF.

## Important files changed

- `user_documents.py` — handles PDF upload, title extraction, per-user document list, and per-user RAG chain.
- `leave_app/views.py` — uses uploaded PDFs instead of fixed `data/leave.pdf`.
- `leave_app/templates/leave_app/chat.html` — added PDF upload section and dynamic heading.
- `qa_chain.py` — prompt changed from fixed leave policy assistant to strict uploaded PDF assistant.
- `leave_policy_project/settings.py` — added media upload configuration.

## Uploaded PDF storage

PDFs are saved here:

```text
media/user_pdfs/<employee_id>/
```

Do not push uploaded PDFs to GitHub.
