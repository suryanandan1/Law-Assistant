import pandas as pd

EXCEL_PATH = "data/employees.xlsx"


def clean_columns(df):
    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )
    return df


def load_employee_df():
    df = pd.read_excel(EXCEL_PATH, header=0)
    df = clean_columns(df)

    if "employee_id" not in df.columns:
        df = pd.read_excel(EXCEL_PATH, header=1)
        df = clean_columns(df)

    return df


def save_employee_df(df):
    try:
        df.to_excel(EXCEL_PATH, index=False)
        return True, "Employee data saved."
    except PermissionError:
        return False, "Please close employees.xlsx and try again."


def get_employee_by_id(employee_id):
    df = load_employee_df()

    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    employee_id = str(employee_id).strip()

    employee = df[df["employee_id"] == employee_id]

    if employee.empty:
        return None

    row = employee.iloc[0]

    return {
        "employee_id": row.get("employee_id", ""),
        "name": row.get("name", ""),
        "grade": row.get("grade/band", row.get("grade", "")),
        "joining_date": row.get("joining_date", ""),
        "PL_taken": row.get("pl_taken", 0),
        "CL_taken": row.get("cl_taken", 0),
        "SL_taken": row.get("sl_taken", 0),
    }


def signup_employee_excel(
    employee_id,
    name,
    grade,
    joining_date,
    pl_taken,
    cl_taken,
    sl_taken
):
    df = load_employee_df()

    df["employee_id"] = df["employee_id"].astype(str).str.strip()
    employee_id = str(employee_id).strip()

    new_employee = {
        "employee_id": employee_id,
        "name": name,
        "grade/band": grade,
        "joining_date": joining_date,
        "pl_taken": pl_taken,
        "cl_taken": cl_taken,
        "sl_taken": sl_taken,
    }

    if employee_id in df["employee_id"].values:
        index = df[df["employee_id"] == employee_id].index[0]

        for key, value in new_employee.items():
            df.loc[index, key] = value

        return save_employee_df(df)

    df = pd.concat([df, pd.DataFrame([new_employee])], ignore_index=True)

    return save_employee_df(df)