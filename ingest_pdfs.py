import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from rag_pipeline import BASE_DIR, get_pdf_collection, ingest_pdf_urls, load_pdf_urls


DEFAULT_PDF_URLS = [
    "https://practitioners.slc.co.uk/media/2188/residency-lle-v10.pdf",
    "https://practitioners.slc.co.uk/media/2194/lle-guidance-chapter-tuition-fee-loan-entitlement-v10.pdf",
    "https://practitioners.slc.co.uk/media/2186/dsa-guidance-2627-v20.pdf",
    "https://practitioners.slc.co.uk/media/2197/sfe-assessing-eligibility-guidance-ay-2627-v20.pdf",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and index SLC PDF guidance into Chroma.")
    parser.add_argument(
        "--url-list",
        type=Path,
        default=BASE_DIR / "pdf_urls.txt",
        help="Optional text file containing one direct PDF URL per line.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY in .env before running ingestion.")

    pdf_urls = load_pdf_urls(DEFAULT_PDF_URLS, args.url_list)
    collection = get_pdf_collection()
    stored_count = ingest_pdf_urls(
        pdf_urls,
        api_key=api_key,
        collection=collection,
        batch_size=args.batch_size,
    )
    print(
        f"Upserted {stored_count} chunks from {len(pdf_urls)} PDFs; "
        f"the collection now contains {collection.count()} chunks."
    )


if __name__ == "__main__":
    main()