# Cluster 38 – Long Eval Sci

This directory is part of the Long Eval Sci dataset and contains the documents of cluster 38 from the test partition. A cluster groups scientific papers related to a similar topic. In the case of cluster 38, the articles correspond to research in computer science (for example, graph algorithms, distributed systems or programming models), as can be deduced from the titles and abstracts of the included papers. The three main files are described below.

## Description of the files

`cluster_38_abstract_documents_sorted.jsonl`

**File type**: JSON Lines. Each line is an independent JSON object.

**Content**: each line contains the following information about a scientific article:

* id – numeric identifier of the paper

* title – title of the article

* abstract – textual abstract of the article. This field may contain line breaks and special characters.

* publishedDate – timestamp with the publication date in ISO 8601 format, or null if it is not available


* year – publication year as an integer, or null if it is not known.

**Ordering**: the objects are sorted by publication year (ascending), which makes it easy to traverse them chronologically.

**Typical use**: this file provides the metadata and abstracts needed for topic analysis, summarisation generation and evaluation of language models. Because each line is an independent document, it is straightforward to process sequentially.

`cluster_38_ids.txt`

**File type**: plain text.

**Content**: list of all identifiers (id) of the documents in this cluster. Each line contains a distinct integer.

**Typical use**: this file is useful for filtering or retrieving the original documents in the full database. It can also be used as a key to join other metadata (for example, to obtain the PDFs of the articles).

`cluster_38_year_stats.json`

**File type**: JSON (single object).

**Content**: a dictionary in which the keys are publication years (as strings) and the values represent the number of articles in the cluster published in that year


Example: the first entries are `{ "1878": 1, "1910": 1, "1918": 2, … }`, indicating that in 1878 there is one article and in 1918 there are two.

Typical use: this file makes it possible to analyse the temporal distribution of the documents in the cluster. It is useful for visualising the historical evolution of the topic or for reordering the documents by year.

## Auxiliary scripts

In addition to the data files, this directory includes utility scripts:

* `extract_cluster_ids.py` – reads a .jsonl file and extracts only the id fields, generating the corresponding IDs file.

* `order_cluster_by_years.py` – sorts the documents of a cluster by publication year and generates the sorted abstracts file.

These scripts are not necessary to use the data, but they show how cluster_38_ids.txt and cluster_38_abstract_documents_sorted.jsonl were generated.

## How to work with these data

1.  **Load the documents**: to process the documents, read cluster_38_abstract_documents_sorted.jsonl line by line and interpret each line as a JSON object. Languages such as Python provide libraries (json or pandas) that easily handle the JSONL format.

2. **Join with other sources**: use the identifiers from cluster_38_ids.txt to link with other collections or to retrieve full articles.

3. **Analyse the temporal distribution**: use cluster_38_year_stats.json to plot histograms or to identify periods with greater research activity.

## Considerations

* **Missing fields**: some articles lack a publication date or year (publishedDate or year are null). In those cases, the article may be old or the original database may not provide the information.

* **Language and topics**: the cluster contains research mostly in English, spanning different subfields of computer science. When analysing the abstracts, it is advisable to consider the technical terminology specific to the area.

This README provides a quick guide for understanding and using the files of cluster 38. See the general documentation of the Long Eval Sci project for more details about the methodology and the purpose of the dataset.