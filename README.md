# Tanpopo (たんぽぽ)

Tanpopo（たんぽぽ）は、[Tantivy](https://github.com/quickwit-oss/tantivy) と [Sudachi.rs](https://github.com/WorksApplications/sudachi.rs) を基盤とした、軽量かつ省メモリな日本語全文検索サーバです。
Tantivy の名前がフランス語の「タンポポ」に由来するように、Tanpopo はその高速性をそのままに、極めて少ないメモリ消費量と Elasticsearch 互換の REST API サブセットで日本語テキスト検索を実現します。

## ワークスペース構成

- **`crates/tanpopo`**: `tanpopo-bm25` と `tanpopo-vector` を Reciprocal Rank Fusion (RRF) および線形結合（Linear Combination）で統合するハイブリッド検索サーバ & MCP サーバ。
- **`crates/tanpopo-bm25`**: 単体動作可能な日本語 BM25 全文検索エンジン（Sudachi.rs + Tantivy）。`cargo run --release -p tanpopo-bm25` で独立して起動できます。
- **`crates/tanpopo-vector`**: 並列 SIMD 内積計算を備えた軽量な `mmap` SQ8（`i8`）ベクトルストア（1ドキュメント = 1ベクトル、`title` + `summary`、または `content`/`body` の先頭 `TANPOPO_DOC_EMBED_CHARS` 文字（デフォルト 4000）を埋め込み）と、OpenAI 互換の `/v1/embeddings` クライアント（`ruri` / `gemma` ネイティブプレフィックス対応）。旧チャンクベース版のベクトルストアは削除して再インデックスする必要があります。

## 主な特徴

- **省メモリ設計**:
  - Sudachi のシステム辞書・ユーザ辞書を `mmap`（`Arc<JapaneseDictionary>`）で読み込み、ヒープの複製なしに全インデックス・全スレッドで共有します。
  - Tantivy の `MmapDirectory` によりインデックスストレージを OS のページキャッシュへ委譲し、インデックスごとの `IndexWriter` ヒープ上限を設定します（デフォルト `15 MB`）。
- **日本語形態素解析 (Sudachi.rs)**:
  - 指定ディレクトリ（`TANPOPO_SUDACHI_DICT_DIR` または `./resources`）内のシステム辞書・ユーザ辞書の自動検出。
  - 分割モード `A`（短単位）、`B`（中単位）、`C`（長単位）に対応。
  - `normalized_form`（例: `打込む` ↔ `打ち込む`、`しょうゆ` ↔ `醤油`）、`dictionary_form`、`surface` による表記ゆれ・活用の正規化。
  - 品詞（`stop_tags`）およびストップワードによるフィルタリング。
- **Elasticsearch 互換 REST API サブセット**:
  - `GET /`, `GET /_cluster/health`, `GET /_cat/indices`
  - `GET /_mapping`, `GET /:index/_mapping`
  - `POST /_analyze`, `POST /:index/_analyze`
  - `PUT /:index`, `GET /:index`, `DELETE /:index`, `POST /:index/_refresh`
  - `POST /:index/_doc`, `PUT /:index/_doc/:id`, `GET /:index/_doc/:id`, `DELETE /:index/_doc/:id`
  - `POST /_bulk`, `POST /:index/_bulk` (NDJSON)
  - `POST /_search`, `POST /:index/_search`, `GET /:index/_search?q=...` (`match`, `match_phrase`, `multi_match`, `term`, `terms`, `wildcard`, `range`, `bool`, `query_string`, `aggs`, `sort`, `docvalue_fields`)
  - `POST /_msearch`, `POST /:index/_msearch` (NDJSON)
- **組み込み Web UI**: 追加依存なしで使える単一ファイルの検索・解析コンソールを `http://localhost:9200/_ui` で提供します。無効化する場合は `TANPOPO_UI=0` を設定してください。

## クイックスタート

### 1. Sudachi 辞書の準備（システム辞書 + 任意のユーザ辞書）

Sudachi のバイナリ辞書（`.dic`）を `./resources` に配置するか、`TANPOPO_SUDACHI_DICT_DIR` で辞書ディレクトリを指定します。
Tanpopo はシステム辞書（`system.dic`、`system_core.dic` など）を自動検出し、その他の `.dic` ファイル（例: `user.dic`、`custom_terms.dic`）をすべてユーザ定義辞書として登録します。

```bash
mkdir -p resources
curl -L -o /tmp/sudachi-dic.zip https://d2ej7fkh96ddlu.cloudfront.net/sudachidict/sudachi-dictionary-latest-core.zip
unzip -j /tmp/sudachi-dic.zip "*/system_core.dic" -d resources/
mv resources/system_core.dic resources/system.dic

# (任意) ビルド済みのユーザ辞書 (*.dic) を resources/ に配置
# 例: resources/user.dic や resources/my_terms.dic
```

*（省略した場合、Tanpopo はテストや簡易検証用の最小限の組み込みフォールバック辞書で起動します。）*

### 2. サーバの起動

```bash
TANPOPO_BIND="127.0.0.1:9200" \
TANPOPO_DATA_DIR="./data" \
TANPOPO_SUDACHI_DICT_DIR="./resources" \
TANPOPO_WRITER_HEAP_BYTES="15000000" \
TANPOPO_EMBED_URL="http://127.0.0.1:8080/v1/embeddings" \
TANPOPO_EMBED_MODEL="cl-nagoya/ruri-v3-310m" \
cargo run --release -p tanpopo
```

> **補足**: 純粋な BM25 モードで動かす場合は `TANPOPO_EMBED_URL` を省略するか、`cargo run --release -p tanpopo-bm25` を実行してください。後方互換性のため、`TANPOPO_DICT_PATH="./resources/system.dic"` によるシステム辞書ファイルの明示指定も引き続き利用できます。

### 3. API 使用例

#### 日本語テキストの形態素解析
```bash
curl -X POST http://localhost:9200/_analyze \
  -H 'Content-Type: application/json' \
  -d '{"text": "関西国際空港に行く", "mode": "A"}'
```

#### カスタム日本語アナライザ付きインデックスの作成
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

#### ドキュメントの検索
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

Tanpopo はエージェント検索向けの MCP サーバとしても動作します。提供ツール: `list_indices`、`search`（行番号付きの Sudachi ハイライト、`_source` の切り詰め/フィルタリング、`minimum_should_match`、トークン診断に対応）、`get_document`（`start_line`/`end_line`、`offset`/`length`、`outline` に対応）、`search_in_document`（Sudachi 正規化を用いたドキュメント内検索）、`analyze_text`。

- HTTP: `POST http://localhost:9200/mcp` (JSON-RPC 2.0)
- stdio: `tanpopo --mcp`（または `TANPOPO_MCP_STDIO=1`。ログは stderr に出力されます）

```json
{ "mcpServers": { "tanpopo": { "command": "tanpopo", "args": ["--mcp"], "env": { "TANPOPO_DATA_DIR": "./data" } } } }
```

## ライセンス

以下のいずれかのライセンスのもとで提供されます（選択可能）。

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) または http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) または http://opensource.org/licenses/MIT)
