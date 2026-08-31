import ast
import os
import json


def extract_chunks(file_path):
    #Extract function/class-level chunks from a single Python file.
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # skip files that fail to parse

    source_lines = source.splitlines()
    chunks = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start_line = node.lineno
            end_line = node.end_lineno
            code = "\n".join(source_lines[start_line - 1:end_line])

            chunks.append({
                "type": type(node).__name__,
                "name": node.name,
                "code": code,
                "file": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "docstring": ast.get_docstring(node) or ""
            })

    return chunks


def extract_all_chunks(source_dir):
    """Walk a directory and extract chunks from every .py file."""
    all_chunks = []
    for root, _, files in os.walk(source_dir):
        for fname in files:
            if fname.endswith(".py"):
                fpath = os.path.join(root, fname)
                all_chunks.extend(extract_chunks(fpath))
    return all_chunks


if __name__ == "__main__":
    chunks = extract_all_chunks("target_repo/src/flask")
    print(f"Extracted {len(chunks)} chunks total")

    os.makedirs("data", exist_ok=True)
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print("Saved to data/chunks.json")