import sqlite3
import bcrypt

DB_PATH = "data/users.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            employee_id TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


def create_user(employee_id, password):
    init_db()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    try:
        cursor.execute(
            "INSERT INTO users (employee_id, password_hash) VALUES (?, ?)",
            (employee_id, password_hash)
        )
        conn.commit()
        return True, "Signup successful. Please login."
    except sqlite3.IntegrityError:
        return False, "Employee ID already exists."
    finally:
        conn.close()


def verify_user(employee_id, password):
    init_db()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT password_hash FROM users WHERE employee_id = ?",
        (employee_id,)
    )

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return False

    stored_hash = row[0]

    return bcrypt.checkpw(
        password.encode("utf-8"),
        stored_hash.encode("utf-8")
    )