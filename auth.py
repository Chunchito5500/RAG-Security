"""Protected shared workspace, not a multi-tenant identity system."""
import hashlib
import hmac
import secrets
import time


def hash_code(code, salt=None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", code.encode(), salt.encode(), 600000).hex()
    return f"pbkdf2_sha256${salt}${digest}"


class WorkspaceAuth:
    def __init__(self, store, code_hash, ttl=3600):
        parts = code_hash.split("$")
        if len(parts) != 3 or parts[0] != "pbkdf2_sha256" or len(parts[2]) != 64:
            raise ValueError("Set RAG_ACCESS_CODE_HASH using manage.py hash-password")
        self.store, self.code_hash, self.ttl = store, code_hash, ttl

    def login(self, code):
        supplied = hash_code(code, self.code_hash.split("$")[1])
        if not hmac.compare_digest(supplied, self.code_hash):
            return None
        token, expires = secrets.token_urlsafe(32), time.time() + self.ttl
        with self.store.connect() as db:
            db.execute("DELETE FROM sessions WHERE expires<=?", (time.time(),))
            db.execute("INSERT INTO sessions VALUES (?,?)", (hashlib.sha256(token.encode()).hexdigest(), expires))
        return {"token": token, "expires_at": expires}

    def valid(self, token):
        if not token or len(token) > 256:
            return False
        return bool(self.store.rows("SELECT 1 FROM sessions WHERE token_hash=? AND expires>?",
                                   (hashlib.sha256(token.encode()).hexdigest(), time.time())))

    def logout(self, token):
        with self.store.connect() as db:
            db.execute("DELETE FROM sessions WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),))
