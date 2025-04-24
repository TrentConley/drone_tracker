# Drone Facial Recognition Application

This application uses a drone (Tello) to capture video, perform real-time facial recognition, and log recognized individuals.

## Features

*   Connects to and controls a Tello drone.
*   Streams video feed from the drone.
*   Detects faces in the video stream.
*   Uses `insightface` for facial embedding extraction.
*   Stores face embeddings in a PostgreSQL database using the `pgvector` extension.
*   Compares detected faces against the stored database to recognize individuals.
*   Logs recognition events.
*   Displays the video feed with bounding boxes and recognized names.

## Prerequisites

*   **Python:** Version 3.8+ recommended.
*   **PostgreSQL:** A running PostgreSQL server (version 11+ recommended for `pgvector`).
*   **Tello Drone:** A Ryze Tello drone connected to the same network as the computer running the application.
*   **Required Python Packages:** See `requirements.txt`.
*   **Face Recognition Models:** The application will download necessary models on first run (configured via environment variables).

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up PostgreSQL Database:**
    *   Ensure your PostgreSQL server is running.
    *   Create the database user specified in your environment settings (or use an existing one like `postgres`).
    *   Create the database (default name is `face_drone_db`):
        ```bash
        createdb face_drone_db
        # Or using psql:
        # psql -U your_postgres_user -c "CREATE DATABASE face_drone_db;"
        ```
    *   Connect to the newly created database:
        ```bash
        psql -d face_drone_db -U your_postgres_user
        ```
    *   Enable the `pgvector` extension within the database:
        ```sql
        CREATE EXTENSION IF NOT EXISTS vector;
        \q
        ```

5.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and fill in your specific configuration details:
        *   `DATABASE_URL`: Your PostgreSQL connection string (e.g., `postgresql://user:password@host:port/face_drone_db`).
        *   `USE_GPU`: Set to `true` if you have a compatible NVIDIA GPU and CUDA installed, otherwise `false`.
        *   `MODEL_PACK_NAME`: The `insightface` model pack to use (e.g., `buffalo_l`). See `insightface` documentation for options.
        *   Other optional settings like `FACE_MATCH_THRESHOLD`, etc.

## Running the Application

1.  **Ensure Drone is Connected:** Power on your Tello drone and connect your computer to its Wi-Fi network.
2.  **Activate Virtual Environment:**
    ```bash
    source venv/bin/activate # Or `venv\Scripts\activate` on Windows
    ```
3.  **Run the Main Script:**
    ```bash
    python main.py
    ```
    *   The application will attempt to connect to the drone, take off, start the video stream, and perform facial recognition.
    *   A window will pop up displaying the drone's video feed with detected faces and recognition results.

## Usage

*   **'q' Key:** Press 'q' while the video window is active to gracefully land the drone and exit the application.
*   **'e' Key:** Press 'e' for an emergency stop (cuts drone motors immediately). Use with caution.

## Adding Known Faces

Currently, adding known faces needs to be done manually or via a separate script. You would typically:
1.  Capture images of the person.
2.  Use the `FaceRecognizer`'s embedding generation capability (potentially in a separate utility script) to get the face embedding vector.
3.  Insert the person's name and their embedding vector into the `known_faces` table in the `face_drone_db` database.

## Troubleshooting

*   **`FATAL: database "face_drone_db" does not exist`**: Ensure you have created the database in PostgreSQL (Step 4).
*   **`Error: vector type not found in the database`** or similar: Make sure you have enabled the `pgvector` extension *within* the correct database (Step 4).
*   **Connection Issues:** Verify your computer is connected to the Tello drone's Wi-Fi. Check firewall settings.
*   **Performance:** If recognition is slow, consider using a smaller `MODEL_PACK_NAME`, ensuring `USE_GPU` is correctly set if applicable, or potentially resizing frames in `main.py` (currently commented out). 