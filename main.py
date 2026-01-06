import logging
import json
import sys
from pathlib import Path

from code.index.elastic_client import ElasticClient
from code.index.mappings import scientific_article_mapping


def setup_logging() -> None:
    """Configure application-wide logging behavior."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> None:
    """
    Full ingestion pipeline:

    1. Load config.yaml
    2. Connect to Elasticsearch
    3. Create index (if not exists)
    4. Load all JSONL files inside data/ directory
    5. Bulk insert documents
    """
    setup_logging()

    config_path = Path("config.yaml")

    if not config_path.exists():
        logging.error("Config file %s not found.", config_path)
        sys.exit(1)

    logging.info("Loading configuration from %s", config_path)

    # ---------------------------------------
    # 1. Load Elasticsearch client from YAML
    # ---------------------------------------
    client = ElasticClient.from_yaml(config_path)

    logging.info("Checking Elasticsearch connection at %s", client.host)

    # ---------------------------------------
    # 2. Check connection
    # ---------------------------------------
    if not client.check_connection():
        logging.error("Elasticsearch is NOT reachable.")
        sys.exit(1)

    logging.info("Elasticsearch is running and reachable!")

    # ---------------------------------------
    # 3. Create index
    # ---------------------------------------
    index_name = "scientific_articles"

    logging.info("Creating index '%s' (if not exists)...", index_name)
    client.create_index(
        index_name=index_name,
        mappings=scientific_article_mapping["mappings"],
        settings=scientific_article_mapping["settings"]
    )

    # ---------------------------------------
    # 4. Load documents from JSONL files
    # ---------------------------------------
    data_dir = Path("./data/longeval_sci_testing_2025_fulltext/documents/")

    if not data_dir.exists():
        logging.error("Data directory %s not found.", data_dir)
        sys.exit(1)

    client.bulk_insert_jsonl_dir(index_name=index_name, folder=data_dir)
    # ---------------------------------------
    # 5. Completed
    # ---------------------------------------
    logging.info("Ingestion completed.")


if __name__ == "__main__":
    main()
