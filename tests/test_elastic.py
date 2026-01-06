from code.index.elastic_client import ElasticClient
import json

def normalize(resp):
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if hasattr(resp, "body"):
        return resp.body
    return resp

def main():
    INDEX = "scientific_articles"
    client = ElasticClient()

    print("\n📌 Checking connection…")
    if not client.check_connection():
        print("❌ Could not connect.")
        return

    print(f"\n📌 Checking if index '{INDEX}' exists…")
    exists = normalize(client.es.indices.exists(index=INDEX))
    print("   → Exists?", exists)

    if not exists:
        print("❌ Index does not exist. Stop.")
        return

    print("\n📌 Counting documents…")
    count = normalize(client.es.count(index=INDEX))
    print(json.dumps(count, indent=2))

    print("\n📌 Fetching sample document…")
    response = normalize(client.search(
        index_name=INDEX,
        query={"match_all": {}},
        size=5
    ))

    print("\n📌 Raw search result:")
    print(json.dumps(response, indent=2))

    hits = response.get("hits", {}).get("hits", [])
    if not hits:
        print("⚠️ No documents found.")
        return

    print("\n📌 First hit:")
    print(json.dumps(hits[0], indent=2))

    print("\n📌 Trying specific search…")
    response2 = normalize(client.search(
        index_name=INDEX,
        query={"match": {"fullText": "rol de ingeniero de sistemas en proyectos formativos valorando su aporte para apoyar el desarrollo de comunidades caso departamento de la guajira"}},
        size=5
    ))

    print(json.dumps(response2, indent=2))


if __name__ == "__main__":
    main()
