import type { DemoScenario } from "./types";

/** Prepared examples only: no document scanning, retrieval, or generation occurs. */
export const demoScenarios: DemoScenario[] = [
  {
    "id": "incident",
    "label": "Normal retrieval",
    "description": "Only trusted incident-response documents.",
    "query": "What is the approved response path after suspicious retrieval behavior is detected?",
    "documents": [
      {
        "id": "incident-result-1",
        "documentId": "doc-incident-1",
        "title": "RAG Incident Runbook",
        "source": {
          "id": "src-incident-1",
          "name": "Security Ops",
          "chunk": "chunk 7"
        },
        "excerpt": "Quarantine flagged documents immediately and suspend them from retrieval expansion while review is pending.",
        "trustScore": 0.96,
        "status": "trusted",
        "flagReason": null,
        "decision": "allowed"
      },
      {
        "id": "incident-result-2",
        "documentId": "doc-incident-2",
        "title": "Admin Review SOP",
        "source": {
          "id": "src-incident-2",
          "name": "Ops Portal",
          "chunk": "chunk 3"
        },
        "excerpt": "Escalations require reviewer notification, provenance inspection, and a signed disposition record.",
        "trustScore": 0.92,
        "status": "trusted",
        "flagReason": null,
        "decision": "allowed"
      }
    ],
    "answer": {
      "status": "available",
      "text": "The approved path is to quarantine the suspicious document, freeze its embeddings from future retrieval, alert the security reviewer, and route the incident into the admin review workflow. User-facing answers should continue only with verified sources and include a reduced-confidence notice if coverage drops.",
      "citationIds": [
        "incident-result-1",
        "incident-result-2"
      ]
    }
  },
  {
    "id": "retention",
    "label": "Mixed retrieval",
    "description": "Trusted policy sources, a review candidate, and blocked conflicting uploads.",
    "query": "What changed in the document retention policy for customer complaint records?",
    "documents": [
      {
        "id": "retention-result-1",
        "documentId": "doc-retention-1",
        "title": "Retention Handbook v3",
        "source": {
          "id": "src-retention-1",
          "name": "Compliance KB",
          "chunk": "chunk 12"
        },
        "excerpt": "Complaint records are retained for a period of seven years from closure or final regulatory action.",
        "trustScore": 0.98,
        "status": "trusted",
        "flagReason": null,
        "decision": "allowed"
      },
      {
        "id": "retention-result-2",
        "documentId": "doc-retention-2",
        "title": "Audit Addendum 2026",
        "source": {
          "id": "src-retention-2",
          "name": "Governance Vault",
          "chunk": "chunk 4"
        },
        "excerpt": "Retention updates align with the revised complaints supervision policy effective January 2026.",
        "trustScore": 0.94,
        "status": "trusted",
        "flagReason": null,
        "decision": "allowed"
      },
      {
        "id": "retention-result-3",
        "documentId": "doc-retention-3",
        "title": "customer-retention-fast-answer.txt",
        "source": {
          "id": "src-external-retention-0",
          "name": "External submissions"
        },
        "excerpt": null,
        "trustScore": 0.12,
        "status": "blocked",
        "flagReason": "Prompt injection phrases and unverifiable uploader identity.",
        "decision": "excluded"
      },
      {
        "id": "retention-result-4",
        "documentId": "doc-retention-4",
        "title": "legacy-complaints-policy-copy.pdf",
        "source": {
          "id": "src-external-retention-1",
          "name": "External submissions"
        },
        "excerpt": null,
        "trustScore": 0.28,
        "status": "blocked",
        "flagReason": "Checksum mismatch with official policy archive.",
        "decision": "excluded"
      },
      {
        "id": "retention-result-5",
        "documentId": "doc-retention-summary",
        "title": "Regional retention summary",
        "source": {
          "id": "src-regional-wiki",
          "name": "Regional team wiki",
          "chunk": "2"
        },
        "excerpt": "A secondary summary of the seven-year policy; source attestation is incomplete.",
        "trustScore": 0.65,
        "status": "flagged",
        "flagReason": "Secondary source is missing signed provenance.",
        "decision": "excluded"
      }
    ],
    "answer": {
      "status": "available",
      "text": "Customer complaint records must now be retained for 7 years instead of 5.",
      "citationIds": [
        "retention-result-1",
        "retention-result-2"
      ]
    }
  },
  {
    "id": "vendors",
    "label": "Blocked vendor source",
    "description": "Trusted intake rules alongside a low-trust vendor submission.",
    "query": "Can external vendors upload policy PDFs directly into the retrieval index?",
    "documents": [
      {
        "id": "vendors-result-1",
        "documentId": "doc-vendors-1",
        "title": "Document Intake Policy",
        "source": {
          "id": "src-vendors-1",
          "name": "Ingestion Rules",
          "chunk": "chunk 5"
        },
        "excerpt": "Third-party uploads must enter a quarantine-first workflow and cannot be exposed to retrieval before review.",
        "trustScore": 0.99,
        "status": "trusted",
        "flagReason": null,
        "decision": "allowed"
      },
      {
        "id": "vendors-result-2",
        "documentId": "doc-vendors-2",
        "title": "Partner Access Control Matrix",
        "source": {
          "id": "src-vendors-2",
          "name": "Identity Vault",
          "chunk": "chunk 9"
        },
        "excerpt": "Vendor identities require validated signing keys before trusted publication is allowed.",
        "trustScore": 0.93,
        "status": "trusted",
        "flagReason": null,
        "decision": "allowed"
      },
      {
        "id": "vendors-result-3",
        "documentId": "doc-vendors-3",
        "title": "vendor-policy-bundle.zip",
        "source": {
          "id": "src-external-vendors-0",
          "name": "External submissions"
        },
        "excerpt": null,
        "trustScore": 0.12,
        "status": "blocked",
        "flagReason": "Archive upload attempted to bypass document-level validation.",
        "decision": "excluded"
      },
      {
        "id": "vendors-result-4",
        "documentId": "doc-vendors-4",
        "title": "partner-faq.pdf",
        "source": {
          "id": "src-external-vendors-1",
          "name": "External submissions"
        },
        "excerpt": null,
        "trustScore": 0.28,
        "status": "blocked",
        "flagReason": "Insufficient source attestation for direct indexing.",
        "decision": "excluded"
      }
    ],
    "answer": {
      "status": "available",
      "text": "Not directly. External vendors can submit documents into the ingestion queue, but those files remain isolated until metadata checks, signature validation, and manual approval complete. Only then can chunks enter the trusted retrieval index.",
      "citationIds": [
        "vendors-result-1",
        "vendors-result-2"
      ]
    }
  },
  {
    "id": "all-low",
    "label": "All documents low trust",
    "description": "Low-trust sources only.",
    "query": "Can we bypass approval for urgent vendor policy updates?",
    "documents": [
      {
        "id": "all-low-result-0",
        "documentId": "doc-vendors-3",
        "title": "vendor-policy-bundle.zip",
        "source": {
          "id": "src-external-vendors-0",
          "name": "External submissions"
        },
        "excerpt": null,
        "trustScore": 0.12,
        "status": "blocked",
        "flagReason": "Archive upload attempted to bypass document-level validation.",
        "decision": "excluded"
      },
      {
        "id": "all-low-result-1",
        "documentId": "doc-vendors-4",
        "title": "partner-faq.pdf",
        "source": {
          "id": "src-external-vendors-1",
          "name": "External submissions"
        },
        "excerpt": null,
        "trustScore": 0.28,
        "status": "blocked",
        "flagReason": "Insufficient source attestation for direct indexing.",
        "decision": "excluded"
      }
    ],
    "answer": {
      "status": "withheld",
      "text": "No answer is available because every retrieved document was excluded.",
      "citationIds": []
    }
  },
  {
    "id": "incomplete",
    "label": "Incomplete metadata",
    "description": "Source metadata and trust details unavailable.",
    "query": "What guidance is available in the imported policy notes?",
    "documents": [
      {
        "id": "incomplete-result-1",
        "documentId": null,
        "title": null,
        "source": null,
        "excerpt": null,
        "trustScore": null,
        "status": "flagged",
        "flagReason": null,
        "decision": "excluded"
      }
    ],
    "answer": {
      "status": "withheld",
      "text": "No answer is available because the retrieved item was excluded.",
      "citationIds": []
    }
  },
  {
    "id": "no-matches",
    "label": "No matching documents",
    "description": "A query event with no retrieved documents.",
    "query": "What is the policy for an undocumented exception?",
    "documents": [],
    "answer": {
      "status": "unavailable",
      "text": null,
      "citationIds": []
    }
  }
];

/** Static ingestion examples; no files have been uploaded or indexed by this UI. */
export const demoIngestionQueue = [
  {
    "id": 1,
    "name": "policy-retention-handbook-v3.pdf",
    "stage": "Indexed",
    "note": "Signature verified and chunked into 18 segments."
  },
  {
    "id": 2,
    "name": "third-party-guidance-notes.docx",
    "stage": "Review",
    "note": "Missing provenance metadata. Awaiting manual approval."
  },
  {
    "id": 3,
    "name": "it-security-runbook-q1.pdf",
    "stage": "Indexed",
    "note": "Trust score increased after checksum match with internal source."
  },
  {
    "id": 4,
    "name": "regional-policy-override.txt",
    "stage": "Quarantined",
    "note": "Prompt injection markers detected in uploaded content."
  }
];
