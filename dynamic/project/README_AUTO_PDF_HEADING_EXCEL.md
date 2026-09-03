# Auto PDF Heading Tracking in Excel

This version removes the old manual Excel data-entry signup flow.

## New flow

1. User signs up with:
   - Employee/User ID
   - Name
   - Password
2. The app automatically creates/updates that user row in `data/employees.xlsx`.
3. User logs in.
4. User uploads any PDF.
5. The app extracts the PDF heading/title.
6. The extracted heading is automatically saved in the `pdf_heading` column of that user's row in `employees.xlsx`.
7. Admin can open `data/employees.xlsx` and see which user uploaded/asked from which PDF.

## Excel columns

The app automatically creates these columns if missing:

- `employee_id`
- `name`
- `grade/band`
- `joining_date`
- `pl_taken`
- `cl_taken`
- `sl_taken`
- `pdf_heading`

You do not need to manually add `pdf_heading`; it is added automatically.

## Multiple PDFs

If the same user uploads multiple PDFs, headings are saved in the same `pdf_heading` cell separated by ` | `.
