# Python CSV Profiler Worker

This Worker exposes a small CSV profiling API for the automated feature engineering app.

## Local Development

Use Node 22+ for Wrangler:

```bash
source ~/.nvm/nvm.sh
nvm use 24
npm install
uv sync
npm run dev
```

The local Worker runs at:

```text
http://localhost:8787
```

## Endpoint

### `POST /profile-csv`

Request body: raw CSV text.

Example:

```bash
printf 'id,name,price\n1,avocado,2.5\n' \
  | curl -sS -X POST --data-binary @- http://localhost:8787/profile-csv
```

Response:

```json
{
  "rows": 1,
  "columns": [
    {
      "name": "id",
      "dtype": "number",
      "missingCount": 0,
      "sampleValues": ["1"]
    }
  ],
  "preview": [{ "id": "1", "name": "avocado", "price": "2.5" }],
  "warnings": []
}
```

## Notes

- The endpoint enforces the app's 10 MiB CSV limit.
- Empty headers are renamed to `unnamed_#` columns.
- Profiling is limited to the first 1,000 non-empty data rows.
- Python Workers must be run through `pywrangler` when Python packages are involved. The npm scripts wrap `uv run pywrangler ...`.
- `polars` is not included as a dependency because `pywrangler sync` cannot bundle its native runtime wheel for the current Python Workers/Pyodide runtime.
- The endpoint uses `pandas` for DataFrame-based profiling because it bundles successfully in Python Workers. If `pandas` is unavailable, it falls back to Python's standard `csv` module.
- To run the agent Worker with this Python Worker as a service binding during local development, start the primary Worker with both configs after syncing Python packages:

```bash
cd python-worker
uv run pywrangler sync
cd ..
source ~/.nvm/nvm.sh
nvm use 24
npx wrangler dev -c wrangler.jsonc -c python-worker/wrangler.jsonc
```
