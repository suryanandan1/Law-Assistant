import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXCEL_PATH = os.path.join(DATA_DIR, "employees.xlsx")

DEFAULT_COLUMNS = [
    "employee_id",
    "name",
    "grade/band",
    "joining_date",
    "pl_taken",
    "cl_taken",
    "sl_taken",
    "pdf_heading",
]


def clean_columns(df):
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )
    return df


def _ensure_columns(df):
    for column in DEFAULT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df


def load_employee_df():
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(EXCEL_PATH):
        df = pd.DataFrame(columns=DEFAULT_COLUMNS)
        df.to_excel(EXCEL_PATH, index=False)
        return clean_columns(df)

    df = pd.read_excel(EXCEL_PATH, header=0)
    df = clean_columns(df)

    if "employee_id" not in df.columns:
        df = pd.read_excel(EXCEL_PATH, header=1)
        df = clean_columns(df)

    df = _ensure_columns(df)
    return df


def save_employee_df(df):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        df = _ensure_columns(df)
        df.to_excel(EXCEL_PATH, index=False)
        return True, "Employee data saved."
    except PermissionError:
        return False, "Please close employees.xlsx and try again."


def _safe_int(value):
    try:
        if pd.isna(value) or value == "":
            return 0
        return int(float(value))
    except Exception:
        return 0


def _safe_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def get_employee_by_id(employee_id):
    df = load_employee_df()

    if "employee_id" not in df.columns:
        return None

    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    employee_id = str(employee_id).strip()

    employee = df[df["employee_id"] == employee_id]

    if employee.empty:
        return None

    row = employee.iloc[0]

    return {
        "employee_id": str(row.get("employee_id", "")),
        "name": _safe_text(row.get("name", "")),
        "grade": _safe_text(row.get("grade/band", row.get("grade", ""))),
        "joining_date": _safe_text(row.get("joining_date", "")),
        "PL_taken": _safe_int(row.get("pl_taken", 0)),
        "CL_taken": _safe_int(row.get("cl_taken", 0)),
        "SL_taken": _safe_int(row.get("sl_taken", 0)),
        "pdf_heading": _safe_text(row.get("pdf_heading", "")),
    }


def create_or_update_employee(employee_id, name=""):
    """Create the admin Excel row during signup. No manual leave/grade data entry is needed."""
    df = load_employee_df()
    df = _ensure_columns(df)

    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    employee_id = str(employee_id).strip()
    name = str(name or "").strip()

    if employee_id in df["employee_id"].values:
        index = df[df["employee_id"] == employee_id].index[0]
        if name:
            df.loc[index, "name"] = name
    else:
        new_employee = {column: "" for column in DEFAULT_COLUMNS}
        new_employee["employee_id"] = employee_id
        new_employee["name"] = name
        df = pd.concat([df, pd.DataFrame([new_employee])], ignore_index=True)

    return save_employee_df(df)


def update_employee_pdf_heading(employee_id, heading):
    """Add the uploaded PDF heading to the user's row in employees.xlsx."""
    df = load_employee_df()
    df = _ensure_columns(df)

    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    employee_id = str(employee_id).strip()
    heading = str(heading or "").strip()

    if not heading:
        return True, "No PDF heading found to update."

    if employee_id not in df["employee_id"].values:
        new_employee = {column: "" for column in DEFAULT_COLUMNS}
        new_employee["employee_id"] = employee_id
        new_employee["pdf_heading"] = heading
        df = pd.concat([df, pd.DataFrame([new_employee])], ignore_index=True)
    else:
        index = df[df["employee_id"] == employee_id].index[0]
        current = _safe_text(df.loc[index, "pdf_heading"]).strip()
        headings = [item.strip() for item in current.split(" | ") if item.strip()]
        if heading not in headings:
            headings.append(heading)
        df.loc[index, "pdf_heading"] = " | ".join(headings)

    return save_employee_df(df)


def remove_employee_pdf_heading(employee_id, heading):
    """Remove a deleted PDF heading from the user's row in employees.xlsx."""
    df = load_employee_df()
    df = _ensure_columns(df)

    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    employee_id = str(employee_id).strip()
    heading = str(heading or "").strip()

    if not heading or employee_id not in df["employee_id"].values:
        return True, "No PDF heading to remove."

    index = df[df["employee_id"] == employee_id].index[0]
    current = _safe_text(df.loc[index, "pdf_heading"]).strip()
    headings = [item.strip() for item in current.split(" | ") if item.strip()]
    headings = [item for item in headings if item != heading]
    df.loc[index, "pdf_heading"] = " | ".join(headings)

    return save_employee_df(df)


# Backward-compatible function name. Old code may still import it.
def signup_employee_excel(employee_id, name, grade="", joining_date="", pl_taken=0, cl_taken=0, sl_taken=0):
    return create_or_update_employee(employee_id, name)
