from csv import reader
from io import StringIO
import json
from urllib.parse import unquote

import pandas as pd
from workers import Response, WorkerEntrypoint

MAX_INFERENCE_ROWS = 1000
MAX_PREVIEW_ROWS = 20


def json_response(payload, status=200):
    return Response(
        json.dumps(payload),
        status=status,
        headers={"content-type": "application/json"},
    )


def is_missing(value):
    return value is None or str(value).strip() == ""


def looks_boolean(value):
    return str(value).strip().lower() in {"true", "false", "0", "1", "yes", "no"}


def looks_number(value):
    trimmed = str(value).strip()
    if trimmed == "":
        return False
    try:
        float(trimmed)
        return True
    except ValueError:
        return False


def looks_date(value):
    trimmed = str(value).strip()
    if trimmed == "" or looks_number(trimmed):
        return False
    return not pd.isna(pd.to_datetime(trimmed, errors="coerce"))


def infer_column_type(values):
    present = [value for value in values if not is_missing(value)]
    if len(present) == 0:
        return "empty"

    checks = [
        ("number", looks_number),
        ("boolean", looks_boolean),
        ("date", looks_date),
    ]
    for inferred_type, check in checks:
        if all(check(value) for value in present):
            return inferred_type

    has_structured_signal = any(
        looks_number(value) or looks_boolean(value) or looks_date(value)
        for value in present
    )
    return "mixed" if has_structured_signal else "string"


def normalize_preview_value(value):
    if is_missing(value):
        return None
    return str(value)


def extract_headers(csv_text):
    rows = reader(StringIO(csv_text))
    try:
        raw_headers = next(rows)
    except StopIteration:
        raise ValueError("CSV has no usable header row.")

    generated_header_count = 0
    headers = []
    for index, header in enumerate(raw_headers):
        trimmed = header.strip()
        if trimmed:
            headers.append(trimmed)
        else:
            generated_header_count += 1
            headers.append(f"unnamed_{index + 1}")

    if len(headers) == 0:
        raise ValueError("CSV has no usable header row.")
    if len(set(headers)) != len(headers):
        raise ValueError("CSV contains duplicate column names.")

    return headers, generated_header_count


def build_dataset_summary(csv_text, file_name, file_size_bytes):
    headers, generated_header_count = extract_headers(csv_text)
    df = pd.read_csv(
        StringIO(csv_text),
        header=0,
        names=headers,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        nrows=MAX_INFERENCE_ROWS,
    )

    if len(df.index) == 0:
        raise ValueError("CSV has no data rows.")

    warnings = []
    if len(df.index) >= MAX_INFERENCE_ROWS:
        warnings.append(
            f"Schema was inferred from the first {MAX_INFERENCE_ROWS:,} rows."
        )
    if generated_header_count > 0:
        suffix = " was" if generated_header_count == 1 else "s were"
        warnings.append(
            f"{generated_header_count} empty column header{suffix} renamed to unnamed columns."
        )

    columns = []
    for column in headers:
        values = df[column].tolist()
        sample_values = []
        for value in values:
            if is_missing(value) or value in sample_values:
                continue
            sample_values.append(str(value))
            if len(sample_values) == 5:
                break

        columns.append(
            {
                "name": column,
                "inferredType": infer_column_type(values),
                "missingCount": sum(1 for value in values if is_missing(value)),
                "sampleValues": sample_values,
            }
        )

    preview_rows = []
    for row in df.head(MAX_PREVIEW_ROWS).to_dict(orient="records"):
        preview_rows.append(
            {column: normalize_preview_value(row.get(column)) for column in headers}
        )

    return {
        "fileName": file_name,
        "fileSizeBytes": file_size_bytes,
        "parsedRowCount": len(df.index),
        "columns": columns,
        "previewRows": preview_rows,
        "warnings": warnings,
    }


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        if request.method != "POST":
            return json_response({"error": "Upload a CSV with POST."}, status=405)

        csv_text = await request.text()
        file_name = unquote(request.headers.get("x-file-name") or "dataset.csv")
        file_size_bytes = int(request.headers.get("x-file-size") or len(csv_text))
        print(f"Received CSV upload: {file_name} ({file_size_bytes} bytes)")

        try:
            return json_response(
                build_dataset_summary(csv_text, file_name, file_size_bytes)
            )
        except Exception as error:
            return json_response({"error": str(error)}, status=400)
