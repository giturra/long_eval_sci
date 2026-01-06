from typing import Any, Dict, Optional, Generator
import json
import logging
from pathlib import Path

import yaml
from elasticsearch import Elasticsearch, helpers

logger = logging.getLogger(__name__)


class ElasticClient:
    """
    High-level wrapper for interacting with Elasticsearch.
    """

    def __init__(self, host: str = "http://localhost:9200", **client_kwargs: Any) -> None:
        self.host = host
        self.es = Elasticsearch(host, **client_kwargs)

    # ------------------------------------------------------------------
    # ALT CONSTRUCTOR FROM YAML CONFIG
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ElasticClient":
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            config: Dict[str, Any] = yaml.safe_load(f) or {}

        es_cfg: Dict[str, Any] = config.get("elasticsearch", {})
        host: str = es_cfg.pop("host", "http://localhost:9200")

        logger.info("Loading ElasticClient from YAML config at %s", path)
        return cls(host=host, **es_cfg)

    # ---------------------------------------------------
    # 1. CHECK CONNECTION
    # ---------------------------------------------------
    def check_connection(self) -> bool:
        try:
            resp = self.es.info()
            logger.info("Connected to Elasticsearch cluster: %s", resp.get("cluster_name"))
            return True
        except Exception as exc:
            logger.error("Error while checking Elasticsearch connection: %s", exc, exc_info=True)
            return False

    # ---------------------------------------------------
    # 2. CREATE INDEX WITH MAPPINGS AND SETTINGS
    # ---------------------------------------------------
    def create_index(
        self,
        index_name: str,
        mappings: Optional[Dict[str, Any]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:

        if self.es.indices.exists(index=index_name):
            logger.info("Index '%s' already exists; skipping creation.", index_name)
            return

        self.es.indices.create(
            index=index_name,
            mappings=mappings,
            settings=settings
        )

        logger.info("Index '%s' created successfully.", index_name)


    # ---------------------------------------------------
    # 3. GENERATOR: READ JSONL FILES SAFELY
    # ---------------------------------------------------
    def load_jsonl(self, path: Path, index_name: str) -> Generator[dict, None, None]:
        """
        Streams documents from a JSONL file, yielding bulk index actions.

        Automatically uses document['id'] as _id if present.
        """

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                doc = json.loads(line)

                # Prefer using their own ID
                doc_id = doc.get("id")

                action = {
                    "_index": index_name,
                    "_source": doc,
                }
                if doc_id is not None:
                    action["_id"] = doc_id

                yield action

    # ---------------------------------------------------
    # 4. BULK INDEX MANY JSONL FILES
    # ---------------------------------------------------
    def bulk_insert_jsonl_dir(self, index_name: str, folder: str | Path) -> None:
        """
        Bulk index all .jsonl files in a directory efficiently.
        """

        folder = Path(folder)
        files = sorted(folder.glob("*.jsonl"))

        if not files:
            logger.warning("No JSONL files found in directory: %s", folder)
            return

        for file in files:
            logger.info("Indexing file: %s", file)

            try:
                helpers.bulk(self.es, self.load_jsonl(file, index_name))
                logger.info("Finished indexing: %s", file)
            except Exception as exc:
                logger.error("Error bulk indexing %s: %s", file, exc)
    
    def bulk_insert_docs(self, index_name: str, docs: list[dict]) -> None:
        actions = []

        for doc in docs:
            action = {
                "_index": index_name,
                "_source": doc,
            }
            if "id" in doc:
                action["_id"] = doc["id"]
            actions.append(action)

        try:
            helpers.bulk(self.es, actions)
            logger.info("Indexed %d documents into '%s'", len(docs), index_name)
        except Exception as exc:
            logger.error("Bulk insert error: %s", exc)

    # ---------------------------------------------------
    # 5. SEARCH
    # ---------------------------------------------------
    def search(self, index_name: str, query: Dict[str, Any], size: int = 10) -> Dict[str, Any]:
        response = self.es.search(
            index=index_name,
            query=query,
            size=size,
        )
        logger.debug(
            "Executed search on index '%s' with query=%s, size=%d",
            index_name, json.dumps(query), size
        )
        return response

    # ---------------------------------------------------
    # 6. DELETE INDEX
    # ---------------------------------------------------
    def delete_index(self, index_name: str) -> None:
        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
            logger.info("Index '%s' deleted.", index_name)
        else:
            logger.info("Index '%s' does not exist; nothing to delete.", index_name)
