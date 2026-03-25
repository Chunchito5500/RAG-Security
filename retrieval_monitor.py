import json, time, uuid

documents = [
    {"id": "doc1", "content": "Safe financial advice", "trust": 0.9},
    {"id": "doc2", "content": "Unverified source", "trust": 0.2},
    {"id": "doc3", "content": "General info", "trust": 0.7},
]

LOG_FILE = "retrieval_logs.json"


def similarity(q, d):
    return len(set(q.split()) & set(d.split())) / 5


def run_query(query):
    # retrieve + score
    results = [
        {
            "id": doc["id"],
            "similarity": round(similarity(query, doc["content"]), 2),
            "trust": doc["trust"],
            "flagged": doc["trust"] < 0.3
        }
        for doc in documents
    ]

    results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:2]

    # log
    log = {
        "id": str(uuid.uuid4()),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "results": results
    }

    try:
        logs = json.load(open(LOG_FILE))
    except:
        logs = []

    logs.append(log)
    json.dump(logs, open(LOG_FILE, "w"), indent=2)

    return log


if __name__ == "__main__":
    print(json.dumps(run_query("financial advice"), indent=2))