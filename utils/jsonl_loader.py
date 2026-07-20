import json


def load_jsonl(file_path) -> list[dict]:
    records = []
    bad_lines = 0
    decoder = json.JSONDecoder()

    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
                continue
            except json.JSONDecodeError:
                pass

            try:
                record, end = decoder.raw_decode(stripped)
                if stripped[end:].strip():
                    bad_lines += 1
                else:
                    records.append(record)
            except json.JSONDecodeError:
                bad_lines += 1

    if bad_lines:
        print(f"Skipped {bad_lines} malformed line(s) in {file_path}.")

    return records
