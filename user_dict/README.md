# Sudachi ユーザ辞書ビルド環境

`tanpopo` は `resources/` ディレクトリ（または `TANPOPO_SUDACHI_DICT_DIR`）内の `system.dic` 以外の `*.dic` ファイルをすべてユーザ辞書として自動読み込みします。

## 使い方

1. `user_dict/user.csv` に単語を追加します（19列の Sudachi CSV 形式、`#` コメント・空行対応）。
   ※ `*.csv` を複数ファイルに分けて置くこともできます。
2. 以下のコマンドで `resources/user.dic` をビルドします。

```bash
uv run user_dict/build.py
```

3. `tanpopo` サーバを再起動すると `resources/user.dic` が自動的に読み込まれます。
   Web UI (`/_ui`) の解析タブや `POST /_analyze` でトークン分割と「ユーザ辞書」バッジを確認できます。

## オプション

```bash
# 出力先やシステム辞書を明示指定する場合
uv run user_dict/build.py -s resources/system.dic -o resources/custom.dic user_dict/user.csv
```
