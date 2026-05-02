from io import StringIO
import csv
import json

from workers import Response, WorkerEntrypoint

try:
    import pandas as pd
except ModuleNotFoundError:
    pd = None


MAX_CSV_SIZE_BYTES = 10 * 1024 * 1024
MAX_PROFILE_ROWS = 1000
PREVIEW_ROWS = 20


def json_response(payload, status=200):
    return Response(
        json.dumps(payload),
        status=status,
        headers={"content-type": "application/json"},
    )


def error_response(message, status=400):
    return json_response({"error": message}, status=status)


def normalize_headers(headers):
    normalized = []
    seen = set()
    warnings = []

    for index, header in enumerate(headers):
        name = str(header or "").strip()
        if not name:
            name = f"unnamed_{index + 1}"
            warnings.append(f"Empty header at position {index + 1} was renamed to {name}.")

        if name in seen:
            raise ValueError(f"Duplicate column name: {name}")

        seen.add(name)
        normalized.append(name)

    return normalized, warnings


def is_missing(value):
    return value is None or str(value).strip() == ""


def looks_number(value):
    try:
        float(str(value).strip())
        return True
    except ValueError:
        return False


def looks_boolean(value):
    return str(value).strip().lower() in {"true", "false", "0", "1", "yes", "no"}


def infer_type(values):
    present = [value for value in values if not is_missing(value)]
    if not present:
        return "empty"
    if all(looks_number(value) for value in present):
        return "number"
    if all(looks_boolean(value) for value in present):
        return "boolean"
    if any(looks_number(value) or looks_boolean(value) for value in present):
        return "mixed"
    return "string"


def profile_csv_with_stdlib(csv_text):
    reader = csv.reader(StringIO(csv_text))

    try:
        raw_headers = next(reader)
    except StopIteration as exc:
        raise ValueError("CSV has no header row.") from exc

    headers, warnings = normalize_headers(raw_headers)
    warnings.append(
        "Pandas is not available in the Python Workers runtime; used csv fallback."
    )
    rows = []
    field_count_warnings = 0

    for raw_row in reader:
        if len(rows) >= MAX_PROFILE_ROWS:
            warnings.append(f"Profile limited to first {MAX_PROFILE_ROWS} rows.")
            break

        if not raw_row or all(is_missing(value) for value in raw_row):
            continue

        if len(raw_row) != len(headers):
            field_count_warnings += 1

        padded = raw_row[: len(headers)] + [""] * max(0, len(headers) - len(raw_row))
        rows.append(dict(zip(headers, padded)))

    if not rows:
        raise ValueError("CSV has no data rows.")

    if field_count_warnings:
        warnings.append(
            f"{field_count_warnings} row{'s' if field_count_warnings != 1 else ''} had an inconsistent field count."
        )

    columns = []
    for header in headers:
        values = [row.get(header, "") for row in rows]
        samples = []
        for value in values:
            if not is_missing(value) and value not in samples:
                samples.append(value)
            if len(samples) == 5:
                break

        columns.append(
            {
                "name": header,
                "dtype": infer_type(values),
                "missingCount": sum(1 for value in values if is_missing(value)),
                "sampleValues": samples,
            }
        )

    return {
        "rows": len(rows),
        "columns": columns,
        "preview": rows[:PREVIEW_ROWS],
        "warnings": warnings,
        "engine": "python-csv",
    }


def dataframe_preview(dataframe):
    preview = dataframe.head(PREVIEW_ROWS).astype(object)
    return preview.where(preview.notna(), None).to_dict(orient="records")


def profile_csv_with_pandas(csv_text):
    reader = csv.reader(StringIO(csv_text))
    try:
        raw_headers = next(reader)
    except StopIteration as exc:
        raise ValueError("CSV has no header row.") from exc

    headers, warnings = normalize_headers(raw_headers)

    dataframe = pd.read_csv(  # pyright: ignore[reportOptionalMemberAccess]
        StringIO(csv_text),
        header=0,
        names=headers,
        nrows=MAX_PROFILE_ROWS,
    )

    if dataframe.empty:
        raise ValueError("CSV has no data rows.")

    columns = []
    for name, dtype in dataframe.dtypes.items():
        series = dataframe[name]
        sample_values = [
            str(value)
            for value in series.dropna().head(5).tolist()
            if str(value).strip()
        ]
        columns.append(
            {
                "name": name,
                "dtype": str(dtype),
                "missingCount": int(series.isna().sum()),
                "sampleValues": sample_values,
            }
        )

    return {
        "rows": len(dataframe),
        "columns": columns,
        "preview": dataframe_preview(dataframe),
        "warnings": warnings,
        "engine": "pandas",
    }


def profile_csv(csv_text):
    if pd is None:
        return profile_csv_with_stdlib(csv_text)
    return profile_csv_with_pandas(csv_text)


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        if request.method == "OPTIONS":
            return Response(
                None,
                headers={
                    "access-control-allow-origin": "*",
                    "access-control-allow-methods": "POST, OPTIONS",
                    "access-control-allow-headers": "content-type",
                },
            )

        url = request.url
        if request.method != "POST" or not url.endswith("/profile-csv"):
            return error_response("Not found", status=404)

        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_CSV_SIZE_BYTES:
            return error_response("CSV exceeds the 10 MiB upload limit.", status=413)

        try:
            csv_text = await request.text()
        except Exception:
            return error_response("Could not read request body.", status=400)

        if not csv_text.strip():
            return error_response("CSV file is empty.", status=400)

        if len(csv_text.encode("utf-8")) > MAX_CSV_SIZE_BYTES:
            return error_response("CSV exceeds the 10 MiB upload limit.", status=413)

        try:
            return json_response(profile_csv(csv_text))
        except Exception as exc:
            return error_response(f"CSV profiling failed: {exc}", status=400)
