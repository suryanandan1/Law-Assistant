# Personalized Leave Policy Assistant - Django Version

This version keeps the existing RAG files unchanged and replaces the Streamlit interface with Django.

## Existing RAG files kept

- `loaders.py`
- `splitter.py`
- `embeddings_store.py`
- `qa_chain.py`
- `employee_data.py`
- `auth_db.py`
- `data/leave.pdf`
- `data/employees.xlsx`
- `data/users.db`

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
DJANGO_SECRET_KEY=change-this-secret-key
DJANGO_DEBUG=True
```

Run Django:

```bash
python manage.py migrate
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## Pages

- `/login/` - employee login
- `/signup/` - employee signup and Excel fetch
- `/` - chat assistant
- `/logout/` - logout
