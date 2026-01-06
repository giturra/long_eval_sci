import json

input_path = "../clustering/clusters_faiss_gpu.jsonl"
output_path = "cluster_38_ids.txt"

with open(input_path, "r", encoding="utf-8") as f, open(output_path, "w") as out:
    for line in f:
        obj = json.loads(line)
        if obj.get("cluster") == 38:
            out.write(obj["id"] + "\n")
