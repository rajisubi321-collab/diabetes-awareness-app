from pathlib import Path
import re
import sqlite3
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError

BASE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = BASE_DIR / "database"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "users.sqlite3"
_hasher = PasswordHasher()

def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        con.commit()

def register_user(email, password):
    email = (email or "").strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return False, "Enter a valid email address."
    if len(password or "") < 10:
        return False, "Use a password with at least 10 characters."
    try:
        hashed = _hasher.hash(password)
        with sqlite3.connect(DB_PATH) as con:
            con.execute(
                "INSERT INTO users(email, password_hash) VALUES (?, ?)",
                (email, hashed)
            )
            con.commit()
        return True, "Account created."
    except sqlite3.IntegrityError:
        return False, "An account with that email already exists."

def authenticate_user(email, password):
    email = (email or "").strip().lower()
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute(
            "SELECT id, email, password_hash FROM users WHERE email = ?",
            (email,)
        ).fetchone()
    if not row:
        return None
    try:
        _hasher.verify(row[2], password or "")
        return {"id": row[0], "email": row[1]}
    except (VerifyMismatchError, VerificationError, ValueError):
        return None
