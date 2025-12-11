# Convert CSV to Parquet using only pandas
import pandas as pd
import gc
import os

def convert_csv_to_parquet_simple(csv_path, parquet_path, chunk_size=50000):
    """Convert CSV to Parquet using pandas only."""
    
    all_chunks = []
    chunk_count = 0
    
    print(f"Converting {csv_path} to Parquet...")
    
    for chunk in pd.read_csv(
        csv_path,
        chunksize=chunk_size,
        usecols=["start", "end", "id","Pfam_id", "Sequence"],
        dtype={
            'start': 'int32',
            'end': 'int32', 
            'Pfam_id': 'category',
            'Sequence': 'string'
        }

    ):
        chunk_count += 1
        all_chunks.append(chunk)
        
        if chunk_count % 20 == 0:
            print(f"Loaded {chunk_count} chunks...", end="\r")
            # Periodically combine and save to manage memory
            if len(all_chunks) >= 20:
                temp_df = pd.concat(all_chunks, ignore_index=True)
                temp_file = f"temp_combined_{chunk_count//20}.parquet"
                temp_df.to_parquet(temp_file, compression='snappy', index=False)
                
                del temp_df, all_chunks
                all_chunks = []
                gc.collect()
        
        del chunk
    
    # Handle remaining chunks
    if all_chunks:
        temp_df = pd.concat(all_chunks, ignore_index=True)
        temp_df.to_parquet("temp_final.parquet", compression='snappy', index=False)
        del temp_df, all_chunks
    
    # Combine all temp files
    import glob
    temp_files = glob.glob("temp_*.parquet")
    
    if temp_files:
        final_chunks = []
        for temp_file in temp_files:
            chunk_df = pd.read_parquet(temp_file)
            final_chunks.append(chunk_df)
            del chunk_df
        
        final_df = pd.concat(final_chunks, ignore_index=True)
        final_df.to_parquet(parquet_path, compression='snappy', index=False)
        
        # Clean up
        for temp_file in temp_files:
            os.remove(temp_file)
        
        print(f"✅ Conversion complete! Created {parquet_path}")
        print(f"Final dataset: {len(final_df)} rows")

# Run the conversion
convert_csv_to_parquet_simple("/global/research/students/sapelt/Masters/MasterThesis/Dataframes/v3/RemainingEntriesCompleteProteins.csv", "optimized_data.parquet")