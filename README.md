# Tanpopo (たんぽぽ)

Tanpopo (たんぽぽ / Dandelion) is a lightweight, memory-efficient Japanese full-text search server built on top of [Tantivy](https://github.com/quickwit-oss/tantivy) and [Sudachi.rs](https://github.com/WorksApplications/sudachi.rs).
Just as Tantivy takes inspiration from the French word for dandelion, Tanpopo brings that same blazing speed to Japanese text search with a feather-light memory footprint and an Elasticsearch-compatible REST API subset.

## Features

- **Low Memory Footprint**:
  - Loads Sudachi's system and user dictionaries via `mmap` (`Arc<JapaneseDictionary>`) and shares them across all indices and threads with zero heap duplication.
  - Delegates index storage to OS page cache via Tantivy's `MmapDirectory` and caps per-index `IndexWriter` heap (`15 MB` default).
- **Japanese Morphological Analysis (Sudachi.rs)**:
  - Automatic discovery of system and user dictionaries in a directory (`TANPOPO_SUDACHI_DICT_DIR` or `./resources`).
  - Split modes `A` (short), `B` (middle), and `C` (long).
  - Spelling & inflection normalization via `normalized_form` (e.g., `打込む` ↔ `打ち込む`, `しょうゆ` ↔ `醤油`), `dictionary_form`, or `surface`.
  - Part-of-speech (`stop_tags`) and stopword filtering.
- **Elasticsearch-Compatible REST API Subset**:
  - `GET /`, `GET /_cluster/health`, `GET /_cat/indices`
  - `GET /_mapping`, `GET /:index/_mapping`
  - `POST /_analyze`, `POST /:index/_analyze`
  - `PUT /:index`, `GET /:index`, `DELETE /:index`, `POST /:index/_refresh`
  - `POST /:index/_doc`, `PUT /:index/_doc/:id`, `GET /:index/_doc/:id`, `DELETE /:index/_doc/:id`
  - `POST /_bulk`, `POST /:index/_bulk` (NDJSON)
  - `POST /_search`, `POST /:index/_search`, `GET /:index/_search?q=...` (`match`, `match_phrase`, `multi_match`, `term`, `terms`, `wildcard`, `range`, `bool`, `query_string`, `aggs`, `sort`, `docvalue_fields`)
  - `POST /_msearch`, `POST /:index/_msearch` (NDJSON)

- **Built-in Web UI**: a single-file search / analyze console at `http://localhost:9200/_ui` (no extra dependencies). Set `TANPOPO_UI=0` to disable it.

## Quick Start

### 1. Prepare Sudachi Dictionaries (System + Optional User Dictionaries)

Place Sudachi binary dictionaries (`.dic`) in `./resources` or point `TANPOPO_SUDACHI_DICT_DIR` to your dictionary directory.
Tanpopo will automatically detect the system dictionary (`system.dic`, `system_core.dic`, etc.) and register any other `.dic` files (e.g., `user.dic`, `custom_terms.dic`) as user-defined dictionaries:

```bash
mkdir -p resources
curl -L -o /tmp/sudachi-dic.zip https://d2ej7fkh96ddlu.cloudfront.net/sudachidict/sudachi-dictionary-latest-core.zip
unzip -j /tmp/sudachi-dic.zip "*/system_core.dic" -d resources/
mv resources/system_core.dic resources/system.dic

# (Optional) Drop any compiled user dictionaries (*.dic) into resources/
# e.g., resources/user.dic or resources/my_terms.dic
```

*(If omitted, Tanpopo starts with a minimal built-in fallback dictionary for testing and quick experimentation).*

### 2. Run the Server

```bash
TANPOPO_BIND="127.0.0.1:9200" \
TANPOPO_DATA_DIR="./data" \
TANPOPO_SUDACHI_DICT_DIR="./resources" \
TANPOPO_WRITER_HEAP_BYTES="15000000" \
cargo run --release
```

> **Note**: For backward compatibility, `TANPOPO_DICT_PATH="./resources/system.dic"` can still be used to specify an explicit system dictionary file.

### 3. API Examples

#### Analyze Japanese Text
```bash
curl -X POST http://localhost:9200/_analyze \
  -H 'Content-Type: application/json' \
  -d '{"text": "関西国際空港に行く", "mode": "A"}'
```

#### Create an Index with Custom Japanese Analyzer
```bash
curl -X PUT http://localhost:9200/articles \
  -H 'Content-Type: application/json' \
  -d '{
    "settings": {
      "analysis": {
        "analyzer": {
          "ja_search": {
            "mode": "A",
            "form": "normalized",
            "stop_tags": ["補助記号", "空白", "助詞", "助動詞"]
          }
        }
      }
    },
    "mappings": {
      "properties": {
        "title": { "type": "text", "analyzer": "ja_search" },
        "category": { "type": "keyword" },
        "views": { "type": "long" },
        "created_at": { "type": "date" }
      }
    }
  }'
```

#### Search Documents
```bash
curl -X POST http://localhost:9200/articles/_search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "bool": {
        "must": { "match": { "title": "空港" } },
        "filter": { "range": { "views": { "gte": 100 } } }
      }
    }
  }'
```

## MCP (Model Context Protocol)

Tanpopo is an MCP server for agentic search. Tools: `list_indices`, `search`, `get_document`, `analyze_text`.

- HTTP: `POST http://localhost:9200/mcp` (JSON-RPC 2.0)
- stdio: `tanpopo --mcp` (or `TANPOPO_MCP_STDIO=1`; logs go to stderr)

```json
{ "mcpServers": { "tanpopo": { "command": "tanpopo", "args": ["--mcp"], "env": { "TANPOPO_DATA_DIR": "./data" } } } }
```
