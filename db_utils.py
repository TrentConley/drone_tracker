# db_utils.py
import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector
import numpy as np
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

# --- Configuration from Environment Variables ---
DB_NAME = os.getenv("PG_DB", "face_drone_db") # Changed default db name
DB_USER = os.getenv("PG_USER", "user")
DB_PASSWORD = os.getenv("PG_PASSWORD", "password")
DB_HOST = os.getenv("PG_HOST", "localhost")
DB_PORT = os.getenv("PG_PORT", "5432")

# Default embedding dimension (InsightFace buffalo_l uses 512)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

# Cosine distance threshold (lower means stricter match)
# Adjust based on testing with your specific model and use case.
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.5"))

# --- Database Connection ---
def get_db_connection():
    """Establishes and returns a connection to the PostgreSQL database."""
    conn_string = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"
    try:
        conn = psycopg.connect(conn_string, row_factory=dict_row)
        register_vector(conn) # Register pgvector type handler
        logging.info(f"Successfully connected to database '{DB_NAME}' on {DB_HOST}:{DB_PORT}.")
        return conn
    except psycopg.OperationalError as e:
        logging.error(f"Database connection failed: {e}")
        logging.error("Check connection details (host, port, dbname, user, password) and ensure PostgreSQL server is running.")
        logging.error("Verify that the database exists and the user has connection permissions.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during database connection: {e}")
        return None

# --- Database Initialization ---
def initialize_database():
    """Connects and initializes the database table and extension."""
    conn = get_db_connection()
    if not conn:
        return False # Connection failed

    try:
        with conn.cursor() as cur:
            logging.info("Ensuring pgvector extension is enabled...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            logging.info(f"Creating table 'face_embeddings' if it doesn't exist (embedding dim: {EMBEDDING_DIM})...")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(255) NOT NULL UNIQUE, -- Ensure names are unique
                    embedding VECTOR({EMBEDDING_DIM}) NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """)

            logging.info("Checking for HNSW index on embeddings...")
            # Check if index exists before creating
            cur.execute("""
                SELECT indexname FROM pg_indexes
                WHERE tablename = 'face_embeddings' AND indexname = 'idx_face_embeddings_hnsw';
            """)
            index_exists = cur.fetchone()
            if not index_exists:
                logging.info("Creating HNSW index 'idx_face_embeddings_hnsw' for efficient similarity search...")
                # Consider adjusting HNSW parameters (m, ef_construction) based on dataset size and performance needs
                cur.execute(f"CREATE INDEX idx_face_embeddings_hnsw ON face_embeddings USING hnsw (embedding vector_cosine_ops);")
                logging.info("HNSW index created.")
            else:
                logging.info("HNSW index already exists.")

            conn.commit()
            logging.info("Database initialized successfully.")
            return True
    except psycopg.Error as e:
        logging.error(f"Database initialization error: {e}")
        conn.rollback() # Rollback changes on error
        return False
    finally:
        if conn:
            conn.close()
            logging.debug("Initialization connection closed.")

# --- Database Operations ---
def add_face(name: str, embedding: np.ndarray):
    """Adds or updates a face embedding in the database."""
    if embedding is None or not isinstance(embedding, np.ndarray) or embedding.ndim != 1:
        logging.error(f"Invalid embedding provided for name '{name}'. Must be a 1D NumPy array.")
        return False
    if embedding.shape[0] != EMBEDDING_DIM:
        logging.error(f"Embedding dimension mismatch for '{name}'. Expected {EMBEDDING_DIM}, got {embedding.shape[0]}.")
        return False

    conn = get_db_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cur:
            # Use INSERT ... ON CONFLICT to handle adding or updating based on unique name
            cur.execute("""
                INSERT INTO face_embeddings (name, embedding)
                VALUES (%s, %s)
                ON CONFLICT (name) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    created_at = CURRENT_TIMESTAMP; -- Optionally update timestamp on update
            """, (name, embedding.astype(np.float32))) # Ensure correct type for pgvector
            conn.commit()
            logging.info(f"Successfully added/updated face embedding for '{name}'.")
            return True
    except psycopg.Error as e:
        logging.error(f"Error adding/updating face embedding for '{name}': {e}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()
            logging.debug("Add/Update face connection closed.")

def find_similar_face(embedding_to_check: np.ndarray):
    """
    Finds the most similar face in the database using cosine distance.

    Args:
        embedding_to_check: The new face embedding (1D NumPy array).

    Returns:
        A tuple (name, distance) if a match within the threshold is found,
        otherwise (None, distance) if a face is found but outside threshold,
        or (None, None) if DB is empty or on error.
    """
    if embedding_to_check is None or not isinstance(embedding_to_check, np.ndarray) or embedding_to_check.ndim != 1:
        logging.error("Invalid embedding provided for similarity search.")
        return None, None
    if embedding_to_check.shape[0] != EMBEDDING_DIM:
        logging.error(f"Query embedding dimension mismatch. Expected {EMBEDDING_DIM}, got {embedding_to_check.shape[0]}.")
        return None, None

    conn = get_db_connection()
    if not conn:
        return None, None

    try:
        with conn.cursor() as cur:
            # <=> operator calculates cosine distance (0=identical, 1=orthogonal, 2=opposite)
            cur.execute(
                """
                SELECT name, embedding <=> %s AS distance
                FROM face_embeddings
                ORDER BY distance ASC
                LIMIT 1;
                """,
                (embedding_to_check.astype(np.float32),) # Ensure correct type
            )
            result = cur.fetchone()

            if result:
                name = result['name']
                distance = result['distance']
                logging.debug(f"Closest match found: {name} (Distance: {distance:.4f})")
                if distance is not None and distance < DISTANCE_THRESHOLD:
                    logging.info(f"Recognized face: {name} (Distance: {distance:.4f} < Threshold: {DISTANCE_THRESHOLD:.4f})")
                    return name, distance
                else:
                    logging.info(f"Closest match ({name}, Dist: {distance:.4f}) does not meet threshold {DISTANCE_THRESHOLD:.4f}.")
                    return None, distance # Match found, but not close enough
            else:
                logging.info("No faces found in the database for comparison.")
                return None, None # Database is empty

    except psycopg.Error as e:
        logging.error(f"Database error during similarity search: {e}")
        return None, None # Indicate error
    except ValueError as e:
        logging.error(f"Vector comparison error (likely type mismatch): {e}")
        return None, None
    finally:
        if conn:
            conn.close()
            logging.debug("Find similar face connection closed.")

def get_all_names():
    """Retrieves all registered names from the database."""
    conn = get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM face_embeddings ORDER BY name;")
            results = cur.fetchall()
            return [row['name'] for row in results]
    except psycopg.Error as e:
        logging.error(f"Error retrieving names from database: {e}")
        return []
    finally:
        if conn:
            conn.close()
            logging.debug("Get all names connection closed.")


def remove_face(name: str):
    """Removes a face entry from the database by name."""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM face_embeddings WHERE name = %s;", (name,))
            deleted_count = cur.rowcount
            conn.commit()
            if deleted_count > 0:
                logging.info(f"Successfully removed face entry for '{name}'.")
                return True
            else:
                logging.warning(f"No face entry found with the name '{name}' to remove.")
                return False
    except psycopg.Error as e:
        logging.error(f"Error removing face entry for '{name}': {e}")
        conn.rollback()
        return False
    finally:
        if conn:
            conn.close()
            logging.debug("Remove face connection closed.")

# --- Test Functions ---
if __name__ == '__main__':
    print("--- Database Utilities Test --- ")
    print("Attempting to initialize database...")
    if initialize_database():
        print("Database initialization successful (or already initialized).")

        # --- Test Adding/Updating Faces ---
        print("\nTesting add_face...")
        dummy_embedding1 = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        add_face("Test Person 1", dummy_embedding1)

        # Update the same person
        dummy_embedding1_updated = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        add_face("Test Person 1", dummy_embedding1_updated)

        # Add another person
        dummy_embedding2 = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        add_face("Test Person 2", dummy_embedding2)

        # --- Test Retrieving Names ---
        print("\nTesting get_all_names...")
        names = get_all_names()
        print(f"Registered names: {names}")
        assert "Test Person 1" in names
        assert "Test Person 2" in names

        # --- Test Finding Similar Face ---
        print("\nTesting find_similar_face...")
        # Test with the first person's updated embedding (should be a perfect match)
        print("Searching for Test Person 1 (expecting match with distance ~0)")
        name, distance = find_similar_face(dummy_embedding1_updated)
        if name:
            print(f"Found: {name}, Distance: {distance:.4f}")
        else:
            print(f"No match found (or distance {distance:.4f} >= threshold {DISTANCE_THRESHOLD})")
        assert name == "Test Person 1"

        # Test with a slightly modified embedding (should still potentially match)
        print("\nSearching with slightly modified embedding...")
        slightly_different_embedding = dummy_embedding1_updated + np.random.normal(0, 0.1, EMBEDDING_DIM).astype(np.float32)
        name, distance = find_similar_face(slightly_different_embedding)
        if name:
            print(f"Found: {name}, Distance: {distance:.4f}")
        else:
            print(f"No match found (or distance {distance:.4f} >= threshold {DISTANCE_THRESHOLD})")

        # Test with a completely random embedding (should not match)
        print("\nSearching with random embedding (expecting no match or high distance)...")
        random_embedding = np.random.rand(EMBEDDING_DIM).astype(np.float32)
        name, distance = find_similar_face(random_embedding)
        if name:
            print(f"Found: {name}, Distance: {distance:.4f}")
        else:
            print(f"No match found (or distance {distance:.4f} >= threshold {DISTANCE_THRESHOLD})")
        assert name is None or distance >= DISTANCE_THRESHOLD

        # --- Test Removing Faces ---
        print("\nTesting remove_face...")
        remove_face("Test Person 1")
        remove_face("Test Person 2")
        # Try removing non-existent face
        remove_face("Non Existent Person")

        # Verify removal
        names_after_removal = get_all_names()
        print(f"Registered names after removal: {names_after_removal}")
        assert "Test Person 1" not in names_after_removal
        assert "Test Person 2" not in names_after_removal

    else:
        print("Database initialization failed. Check connection details and PostgreSQL server status.")

    print("--- Test Finished ---") 