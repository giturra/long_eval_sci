scientific_article_mapping = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,

        "analysis": {
            "analyzer": {
                "my_english": {
                    "type": "standard",
                    "stopwords": "_english_"
                }
            }
        }
    },

    "mappings": {
        "properties": {

            "id": {"type": "keyword"},

            "title": {
                "type": "text",
                "analyzer": "my_english",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },

            "abstract": {
                "type": "text",
                "analyzer": "my_english"
            },

            "fullText": {
                "type": "text",
                "analyzer": "my_english"
            },

            "authors": {
                "type": "nested",
                "properties": {
                    "name": {
                        "type": "text",
                        "analyzer": "my_english",
                        "fields": {"keyword": {"type": "keyword"}}
                    }
                }
            },

            "links": {
                "type": "nested",
                "properties": {
                    "type": {"type": "keyword"},
                    "url": {"type": "keyword"}
                }
            },

            "createdDate": {"type": "date"},
            "publishedDate": {"type": "date"},
            "updatedDate": {"type": "date"},

            "doi": {"type": "keyword"},
            "arxivId": {"type": "keyword"},
            "pubmedId": {"type": "keyword"},
            "magId": {"type": "keyword"},

            "oaiIds": {"type": "keyword"}
        }
    }
}
