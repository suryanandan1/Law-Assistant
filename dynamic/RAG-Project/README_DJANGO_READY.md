# Leave Policy RAG - Django Version

This Django version includes:

- Employee ID + Password login
- Signup with Fetch Employee Data from Excel
- Auto-filled signup fields when employee exists in employees.xlsx
- Personalized Leave Policy RAG chat
- Collapsible sidebar for fullscreen chat
- Generating answer loader
- Clean chat UI similar to the reference screenshot

## Run Project

```powershell
cd Leave_Policy_RAG-main
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000/
```

## Important

Keep these files in the `data` folder:

```text
data/leave.pdf
data/employees.xlsx
```

Create `.env` in project root:

```env
MISTRAL_API_KEY=your_actual_api_key_here
```
