
"""PhishGuard AI - User Authentication and Database Management."""

import hashlib
import hmac
import re
import secrets
import sqlite3

from pathlib import Path
from datetime import datetime


# =========================================================
# 1. DATABASE CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATABASE_PATH = DATA_DIR / "phishguard.db"

HASH_ITERATIONS = 600_000


def get_connection():
    """Create and return a SQLite database connection."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


# =========================================================
# 2. DATABASE INITIALIZATION
# =========================================================

def init_db():
    """Create users and scan history tables if they do not exist."""

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                time TEXT NOT NULL,
                url TEXT NOT NULL,
                status TEXT NOT NULL,
                score TEXT NOT NULL,
                phishing_score REAL NOT NULL,
                is_phishing INTEGER NOT NULL,

                FOREIGN KEY (user_id)
                REFERENCES users(id)
                ON DELETE CASCADE
            )
        """)

        connection.commit()


# =========================================================
# 3. PASSWORD HASHING
# =========================================================

def hash_password(password, salt=None):
    """Generate a secure PBKDF2 password hash."""

    if salt is None:
        salt = secrets.token_bytes(16)

    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        HASH_ITERATIONS
    )

    return password_hash.hex(), salt.hex()


# =========================================================
# 4. USER REGISTRATION
# =========================================================

def register_user(name, email, password):
    """
    Register a new user.

    Returns:
        (True, success_message) on success
        (False, error_message) on failure
    """

    name = name.strip()
    email = email.strip().lower()

    if not name:
        return False, "Please enter your name."

    if len(name) > 100:
        return False, "Name must not exceed 100 characters."

    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        return False, "Please enter a valid email address."

    if len(email) > 254:
        return False, "Email address is too long."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    if len(password) > 128:
        return False, "Password must not exceed 128 characters."

    password_hash, salt = hash_password(password)

    created_at = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    try:
        with get_connection() as connection:

            connection.execute("""
                INSERT INTO users (
                    name,
                    email,
                    password_hash,
                    salt,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
            """, (
                name,
                email,
                password_hash,
                salt,
                created_at
            ))

            connection.commit()

        return True, "Account created successfully! Please log in."

    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."

    except sqlite3.Error:
        return False, "Registration failed. Please try again."


# =========================================================
# 5. USER LOGIN
# =========================================================

def authenticate_user(email, password):
    """
    Verify user credentials.

    Returns:
        User dictionary if credentials are correct.
        None if credentials are incorrect.
    """

    email = email.strip().lower()

    try:
        with get_connection() as connection:

            user = connection.execute("""
                SELECT id, name, email, password_hash, salt
                FROM users
                WHERE email = ?
            """, (email,)).fetchone()

        if user is None:
            return None

        salt = bytes.fromhex(user["salt"])

        entered_hash, _ = hash_password(password, salt)

        if not hmac.compare_digest(
            entered_hash,
            user["password_hash"]
        ):
            return None

        return {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"]
        }

    except (sqlite3.Error, ValueError):
        return None


# =========================================================
# 6. SAVE USER SCAN HISTORY
# =========================================================

def save_scan(user_id, scan_data):
    """Save a scan belonging to a specific user."""

    try:
        with get_connection() as connection:

            connection.execute("""
                INSERT INTO scans (
                    user_id,
                    time,
                    url,
                    status,
                    score,
                    phishing_score,
                    is_phishing
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                scan_data["time"],
                scan_data["url"],
                scan_data["status"],
                scan_data["score"],
                float(scan_data["phishing_score"]),
                int(scan_data["is_phishing"])
            ))

            connection.commit()

        return True

    except (sqlite3.Error, KeyError, TypeError, ValueError):
        return False


# =========================================================
# 7. FETCH USER SCAN HISTORY
# =========================================================

def get_user_scans(user_id):
    """Return scan history for the logged-in user only."""

    try:
        with get_connection() as connection:

            scans = connection.execute("""
                SELECT
                    time,
                    url,
                    status,
                    score,
                    phishing_score,
                    is_phishing
                FROM scans
                WHERE user_id = ?
                ORDER BY id DESC
            """, (user_id,)).fetchall()

        return [
            {
                "time": scan["time"],
                "url": scan["url"],
                "status": scan["status"],
                "score": scan["score"],
                "phishing_score": scan["phishing_score"],
                "is_phishing": bool(scan["is_phishing"])
            }
            for scan in scans
        ]

    except sqlite3.Error:
        return []


# =========================================================
# 8. DELETE USER SCAN HISTORY
# =========================================================

def delete_user_scans(user_id):
    """Delete scan history belonging only to the specified user."""

    try:
        with get_connection() as connection:

            connection.execute("""
                DELETE FROM scans
                WHERE user_id = ?
            """, (user_id,))

            connection.commit()

        return True

    except sqlite3.Error:
        return False
