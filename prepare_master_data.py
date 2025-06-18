# prepare_master_data.py
import pandas as pd
import faiss
import argparse
from tqdm import tqdm
from matcher import embed_text, build_faiss_index

# --- Configuration ---
# These paths can be overridden by command-line arguments
DEFAULT_MASTER_FILE = "data/master_data.xlsx"
DEFAULT_INDEX_FILE = "faiss_index.bin"
DEFAULT_METADATA_FILE = "master_metadata.parquet"

# Define the columns that provide meaningful descriptions for embedding
# IMPORTANT: Adjust these column names to match your master_data.xlsx file!
COLUMNS_TO_USE = [
    "Item", 
    "Sales-Description", 
    "Dimension (complete)",
    "Product-Name", 
    "Tariff-Number-Description", 
    "customer", 
    "Catalog-Group-Code"
]

def main(args):
    """Main function to run the data preparation pipeline."""
    print(f"🔍 Loading master data from: {args.input}")
    try:
        # Load only necessary columns to save memory
        df = pd.read_excel(args.input, usecols=lambda col: col in COLUMNS_TO_USE)
    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found at '{args.input}'. Make sure it's in the 'data/' folder.")
        return
    except ValueError as e:
        print(f"❌ ERROR: Could not read Excel file. One or more columns in COLUMNS_TO_USE might be missing. Details: {e}")
        return
        
    print("🧹 Cleaning and preparing data...")
    df.fillna('', inplace=True)
    df.drop_duplicates(subset=['Item'], inplace=True, keep='first')
    df.reset_index(drop=True, inplace=True)

    print("🧠 Creating combined 'full_text' for embedding...")
    tqdm.pandas(desc="→ Combining text fields")
    df["full_text"] = df[COLUMNS_TO_USE].astype(str).progress_apply(
        lambda x: ' '.join(x.str.strip()), axis=1
    )

    print("📐 Embedding master data with SentenceTransformer (this may take a while)...")
    embeddings = embed_text(df["full_text"].tolist())

    print("🔎 Building FAISS index...")
    index = build_faiss_index(embeddings)

    print(f"💾 Saving FAISS index to {args.index_out}")
    faiss.write_index(index, args.index_out)

    print(f"💾 Saving metadata as Parquet to {args.meta_out}")
    df.to_parquet(args.meta_out)

    print("\n✅ Master data processing complete.")
    print(f"   - Index created: {args.index_out}")
    print(f"   - Metadata created: {args.meta_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare master data for AI Fastener Matcher.")
    parser.add_argument("-i", "--input", type=str, default=DEFAULT_MASTER_FILE, help="Path to the master data Excel file.")
    parser.add_argument("--index-out", type=str, default=DEFAULT_INDEX_FILE, help="Output path for the FAISS index file.")
    parser.add_argument("--meta-out", type=str, default=DEFAULT_METADATA_FILE, help="Output path for the metadata file.")
    
    args = parser.parse_args()
    main(args)