import tempfile
import unittest

from fastapi.testclient import TestClient

from api import create_app
from auth import hash_code
from persistent_application import PersistentApplication
from test_persistent import TestEmbedder


class PersistentApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.password_hash = hash_code("test-workspace-secret")

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.service = PersistentApplication(temp.name, TestEmbedder(), start_worker=False)
        self.client = self.enterContext(TestClient(create_app(self.service, self.password_hash), raise_server_exceptions=False))

    def login(self):
        response = self.client.post("/api/session", json={"access_code": "test-workspace-secret"})
        self.assertEqual(response.status_code, 200, response.text)
        return {"Authorization": "Bearer " + response.json()["data"]["token"]}

    def test_authentication_cors_and_revocation(self):
        self.assertEqual(self.client.get("/api/documents").status_code, 401)
        self.assertEqual(self.client.post("/api/documents/upload?filename=x.txt", content=b"bank").status_code, 401)
        self.assertEqual(self.client.post("/api/session", json={"access_code": "wrong"}).status_code, 401)
        headers = self.login()
        self.assertEqual(self.client.get("/api/documents", headers=headers).status_code, 200)
        preflight = self.client.options("/api/query", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization,content-type"})
        self.assertEqual(preflight.status_code, 200)
        self.client.delete("/api/session", headers=headers)
        self.assertEqual(self.client.get("/api/documents", headers=headers).status_code, 401)

    def test_upload_query_and_response_validation(self):
        headers = {**self.login(), "Content-Type": "text/plain"}
        response = self.client.post("/api/documents/upload?filename=bank.txt", content=b"Bank records are retained.", headers=headers)
        self.assertEqual(response.status_code, 202, response.text)
        identity = response.json()["data"]["document_id"]
        self.service.process_next()
        doc = self.client.get(f"/api/documents/{identity}", headers=headers).json()["data"]
        self.assertEqual(doc["processing_status"], "processed")
        result = self.client.post("/api/query", json={"query": "bank", "top_k": 5}, headers=self.login())
        self.assertEqual(result.status_code, 200, result.text)
        self.assertEqual(result.json()["data"]["trusted_documents"][0]["document_id"], identity)
        self.assertEqual(self.client.post("/api/query", json={"query": "bank", "top_k": True}, headers=self.login()).status_code, 422)
        self.assertEqual(self.client.get("/api/documents/missing", headers=headers).status_code, 404)

    def test_expired_session_and_readiness(self):
        headers = self.login()
        with self.service.store.connect() as db:
            db.execute("UPDATE sessions SET expires=0")
        self.assertEqual(self.client.get("/api/flagged-events", headers=headers).status_code, 401)
        self.assertEqual(self.client.get("/api/ready").status_code, 200)
        with self.service.store.connect() as db:
            db.execute("PRAGMA user_version=999")
        self.assertEqual(self.client.get("/api/ready").status_code, 503)
        self.assertEqual(self.client.get("/api/health").json(), {"data": {"status": "ok"}})

    def test_missing_auth_configuration_fails_startup(self):
        with self.assertRaises(ValueError):
            with TestClient(create_app(self.service, "invalid")):
                pass
