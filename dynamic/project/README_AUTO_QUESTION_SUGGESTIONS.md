# Automatic PDF Question Suggestions

This version automatically creates 4 suggestion questions from the logged-in user's uploaded PDF.

## What changed

- Suggestions are no longer hard-coded leave questions.
- `user_documents.py` now extracts clean lines/topics from the uploaded PDF.
- The app converts those topics into 4 clickable question buttons.
- If no strong topic is found, it uses safe generic fallback questions based on the PDF title.

## Updated files

- `user_documents.py`
  - Added `get_pdf_question_suggestions(employee_id)`.
  - Added helper functions to extract topic lines from PDF text.

- `leave_app/views.py`
  - Sends `suggestions` to `chat.html`.

- `leave_app/templates/leave_app/chat.html`
  - Suggestion buttons now come from the uploaded PDF automatically.
