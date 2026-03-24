import json
from pathlib import Path
from pypdf import PdfReader

DATA_DIR = Path("../data")
OUTPUT_FILE = Path("chunks.json")


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({"source": pdf_path.name, "page": i + 1, "text": text.strip()})

    return pages


def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def main():
    all_chunks = []

    for file in DATA_DIR.iterdir():
        if file.suffix.lower() == ".pdf":
            pages = extract_text_from_pdf(file)

            for page in pages:
                text_chunks = chunk_text(page["text"])

                for idx, chunk in enumerate(text_chunks):
                    all_chunks.append(
                        {
                            "id": f"{file.stem}_p{page['page']}_c{idx}",
                            "source": page["source"],
                            "page": page["page"],
                            "text": chunk,
                        }
                    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_chunks)} chunks to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
