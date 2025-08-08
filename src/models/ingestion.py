# src/ingestion.py
import pandas as pd
import os
import hashlib
from datetime import datetime
from pymongo import MongoClient, UpdateOne, InsertOne
from dotenv import load_dotenv
import time
from dateutil import parser as date_parser  # Renamed to avoid conflict

# Import configuration
from config import (
    RAW_DATA_DIR, 
    PRODUCTS_COLLECTION, 
    METADATA_COLLECTION,
    MIN_BATCH_SIZE,
    DEFAULT_DATA_FILE
)

def get_mongodb_connection():
    """Create and return a MongoDB connection using environment variables"""
    load_dotenv()
    
    username = os.getenv("MONGODB_USERNAME")
    password = os.getenv("MONGODB_PASSWORD")
    cluster = os.getenv("MONGODB_CLUSTER")
    database = os.getenv("MONGODB_DATABASE")
    
    connection_string = f"mongodb+srv://{username}:{password}@{cluster}/"
    client = MongoClient(connection_string)
    db = client.get_database(database)
    
    return db

def create_record_hash(row, key_fields):
    """Create a hash from the fields that make a record unique with better normalization"""
    # Create a normalized string from key fields
    components = []
    for field in key_fields:
        value = row[field]
        # Convert everything to string and strip whitespace
        value_str = str(value).strip()
        
        # Special handling for dates
        if field == 'first_visit_date':
            # Try to normalize date format
            try:
                # Handle different date formats
                parsed_date = date_parser.parse(value_str)
                value_str = parsed_date.strftime('%Y-%m-%d')  # Standardized format
            except Exception as e:
                # If parsing fails, use the original but still normalized
                pass
        
        components.append(value_str)
    
    # Join components with a delimiter to avoid problems
    unique_string = "|".join(components)
    return hashlib.md5(unique_string.encode()).hexdigest()

def get_latest_processed_metadata(db):
    """Get the latest processing metadata"""
    metadata_collection = db[METADATA_COLLECTION]
    latest_metadata = metadata_collection.find_one(
        sort=[("processed_at", -1)]
    )
    return latest_metadata

def save_processing_metadata(db, file_path, record_count, last_timestamp, processed_hashes):
    """Save metadata about this processing run"""
    metadata_collection = db[METADATA_COLLECTION]
    metadata = {
        "source_file": file_path,
        "processed_at": datetime.now(),
        "last_processed_timestamp": last_timestamp,
        "record_count": record_count,
        "processed_hashes": processed_hashes[:5] if processed_hashes else []  # Store a few sample hashes for reference
    }
    metadata_collection.insert_one(metadata)
    print(f"Saved metadata with timestamp {last_timestamp}")

def ingest_data(file_name=None, key_fields=None, min_batch_size=None, keep_hash=True):
    """
    Ingest data from a CSV file into MongoDB with improved deduplication
    
    Parameters:
    file_name (str): Name of the file in the raw data directory to ingest
    key_fields (list): List of fields that define a unique record
    min_batch_size (int): Minimum number of new records required to process
    keep_hash (bool): Whether to keep the hash column in the stored data
    
    Returns:
    dict: Statistics about the ingestion process
    """
    # Set defaults if not provided
    if file_name is None:
        file_name = DEFAULT_DATA_FILE
    
    if key_fields is None:
        # Default key fields based on the actual data schema
        key_fields = ['user_id', 'first_visit_date']
    
    if min_batch_size is None:
        min_batch_size = MIN_BATCH_SIZE
    
    # Full path to the file
    file_path = os.path.join(RAW_DATA_DIR, file_name)
    
    # Get database connection
    db = get_mongodb_connection()
    collection = db[PRODUCTS_COLLECTION]
    
    # Create a unique index on record_hash if it doesn't exist
    try:
        collection.create_index("record_hash", unique=True)
        print("Created or verified unique index on record_hash")
    except Exception as e:
        print(f"Note: {e}")
    
    # Print current collection stats
    print(f"Before ingestion: {collection.count_documents({})} documents in collection")
    
    # Load the CSV data
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records from {file_name}")
    print(f"Columns in the dataset: {list(df.columns)}")
    
    # --- FIX: Validate that key_fields exist in the DataFrame ---
    validated_key_fields = [field for field in key_fields if field in df.columns]
    if len(validated_key_fields) != len(key_fields):
        missing_fields = set(key_fields) - set(validated_key_fields)
        # You can either raise an error or log a warning and proceed with the existing fields
        logger.warning(f"Warning: The following key_fields were not found in the data and will be ignored: {missing_fields}")
        if not validated_key_fields:
            raise ValueError("None of the specified key_fields exist in the data. Cannot generate hash.")
    
    print(f"Using these columns for deduplication: {validated_key_fields}")
    # --- END FIX ---

    # Ensure the dataframe has a timestamp column
    if 'timestamp' not in df.columns:
        df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("Added timestamp column to the data")
    
    # Generate unique hash for each record - use the validated fields
    df['record_hash'] = df.apply(lambda row: create_record_hash(row, validated_key_fields), axis=1)
    print("Generated unique hash for each record")
    
    # (The rest of your function remains the same...)
    
    # ... (rest of the function)
    print(f"Sample hashes from current file: {df['record_hash'].head(5).tolist()}")
    
    print("DEBUG: Sample record hash generation:")
    for idx, row in df.head(3).iterrows():
        components = []
        for field in validated_key_fields: # Use validated fields here too
            value = row[field]
            value_str = str(value).strip()
            
            if field == 'first_visit_date':
                try:
                    parsed_date = date_parser.parse(value_str)
                    value_str = parsed_date.strftime('%Y-%m-%d')
                except:
                    pass
            
            components.append(value_str)
        
        unique_string = "|".join(components)
        hash_val = hashlib.md5(unique_string.encode()).hexdigest()
        print(f"Record {idx}: Raw values={row[validated_key_fields].tolist()}, Normalized='{unique_string}', Hash={hash_val}")
    
    print("Fetching existing records from database...")
    existing_hashes = set()
    
    cursor = collection.find({}, {"record_hash": 1, "_id": 0})
    for doc in cursor:
        if "record_hash" in doc:
            existing_hashes.add(doc["record_hash"])
    
    print(f"Found {len(existing_hashes)} existing record hashes in the database")
    
    new_records_df = df[~df['record_hash'].isin(existing_hashes)]
    print(f"After deduplication: {len(new_records_df)} truly new records found")
    
    if len(new_records_df) >= min_batch_size:
        print(f"Found {len(new_records_df)} new records to process (meets {min_batch_size} minimum)")
        df_to_process = new_records_df
    else:
        print(f"Only {len(new_records_df)} new records found (< {min_batch_size} required). No update performed.")
        df_to_process = pd.DataFrame()
    
    stats = {
        "file_processed": file_name,
        "total_records": len(df),
        "new_records": len(new_records_df),
        "inserted_records": 0,
        "success": False,
        "timestamp": datetime.now()
    }
    
    if not df_to_process.empty:
        if not keep_hash:
            insert_df = df_to_process.copy()
            processed_hashes = df_to_process['record_hash'].tolist()
            insert_df = insert_df.drop(columns=['record_hash'])
            print("Removed hash column from data before insertion")
        else:
            insert_df = df_to_process
            processed_hashes = df_to_process['record_hash'].tolist()
        
        records = insert_df.to_dict("records")
        
        print(f"Preparing to insert {len(records)} records in bulk...")
        bulk_operations = [UpdateOne({"record_hash": record["record_hash"]}, {"$set": record}, upsert=True) for record in records]
        
        batch_size = 500
        inserted_count = 0
        
        start_time = time.time()
        
        for i in range(0, len(bulk_operations), batch_size):
            batch = bulk_operations[i:i+batch_size]
            try:
                result = collection.bulk_write(batch, ordered=False)
                inserted_count += result.upserted_count
            except Exception as e:
                print(f"Error in bulk write: {e}")
        
        end_time = time.time()
        print(f"Bulk operation completed in {end_time - start_time:.2f} seconds")
        
        latest_timestamp = df_to_process['timestamp'].max()
        
        save_processing_metadata(db, file_path, inserted_count, latest_timestamp, processed_hashes)
        
        stats["inserted_records"] = inserted_count
        stats["success"] = True
        
        print(f"Upserted {inserted_count} documents into MongoDB")
        print(f"Collection now has {collection.count_documents({})} documents")
    else:
        print("No data was processed.")
    
    return stats

# Example of how to call this function
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest data into MongoDB")
    parser.add_argument("--file", type=str, help="File name to ingest")
    parser.add_argument("--min-batch", type=int, help="Minimum batch size")
    parser.add_argument("--key-fields", nargs='+', help="Fields to use for deduplication")
    parser.add_argument("--keep-hash", action="store_true", help="Keep hash column in stored data")
    parser.add_argument("--no-keep-hash", dest="keep_hash", action="store_false", help="Remove hash column before storing")
    parser.set_defaults(keep_hash=True)
    
    args = parser.parse_args()
    
    # Call the ingest function with arguments from command line or defaults
    stats = ingest_data(
        file_name=args.file,
        min_batch_size=args.min_batch,
        key_fields=args.key_fields,
        keep_hash=args.keep_hash
    )
    
    print(f"Ingestion completed. Stats: {stats}")