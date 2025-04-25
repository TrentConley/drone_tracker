# Drone Person Following Application

This application uses a drone (Tello) to capture video, detect people using **YOLOv10**, track them using **ByteTrack**, recognize known faces using **InsightFace**, and follow a designated target person, maintaining a distance of approximately 0.5 meters.

## Features

*   Connects to and controls a Tello drone.
*   Streams video feed from the drone.
*   Detects people using **YOLOv10s**.
*   Tracks detected people across frames using **ByteTrack**.
*   Detects faces within person bounding boxes.
*   Uses **InsightFace** (`buffalo_l` by default) for facial embedding extraction.
*   Stores face embeddings in a PostgreSQL database using the `pgvector` extension.
*   Compares detected faces against the stored database to recognize individuals.
*   When an unrecognized person is detected, the application attempts to lock onto a recognized face within that person's bounding box.
*   Once locked, the application controls the drone (**yaw** and **forward/backward**) to maintain a target distance (approx. 0.5m) from the tracked person using bounding box width and camera parameters (`camera_utils.py`).
*   If the target is lost, the drone rotates (`SEARCH_YAW_VEL`) to find a recognized person again.
*   Displays the video feed with bounding boxes for people and recognized faces, highlighting the locked target and showing the estimated distance.
*   Includes a **webcam simulator** (`webcam_simulator.py`) to test the core logic without a drone.

## Prerequisites

*   **Python:** Version 3.8+ recommended.
*   **PostgreSQL:** A running PostgreSQL server (version 11+ recommended for `pgvector`).
*   **Tello Drone:** A Ryze Tello drone connected to the same network as the computer running the application.
*   **Required Python Packages:** See `requirements.txt`. Includes `ultralytics`, `insightface`, `djitellopy`, `psycopg2`, etc.
*   **`onnxruntime`:** Ensure you have the correct `onnxruntime` package installed. For NVIDIA GPU support, install `onnxruntime-gpu`. For Apple Silicon (MPS) support on macOS, install the standard `onnxruntime` (version 1.10+ recommended).
*   **ML Models:** The application will download necessary **InsightFace** models on first run (configured via `MODEL_PACK_NAME` in `.env`). You need to manually download the **YOLOv10s** weights (`yolov10s.pt`) - see Ultralytics documentation.
*   **(Optional) NVIDIA GPU:** CUDA Toolkit and cuDNN if using `USE_GPU=true` on Linux/Windows.
*   **(Optional) Apple Silicon Mac:** For MPS support (`USE_GPU=true` on macOS).

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Download YOLOv10 Weights:**
    *   Obtain the `yolov10s.pt` file from the official YOLOv10 releases (or train your own) and place it in the root project directory.

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
    ```

4.  **Install Python Dependencies:**
    ```bash
    # For CPU or Apple Silicon (MPS) on macOS:
    pip install -r requirements.txt

    # For NVIDIA GPU on Linux/Windows (ensure CUDA/cuDNN are installed first):
    # You might need to manually install onnxruntime-gpu first, check compatibility:
    # pip uninstall onnxruntime onnxruntime-gpu
    # pip install onnxruntime-gpu
    # pip install -r requirements.txt
    ```

5.  **Set up PostgreSQL Database:**
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

6.  **Configure Environment Variables:**
    *   Copy the example environment file:
        ```bash
        cp .env.example .env
        ```
    *   Edit the `.env` file and fill in your specific configuration details:
        *   `DATABASE_URL`: Your PostgreSQL connection string (e.g., `postgresql://user:password@host:port/face_drone_db`).
        *   `USE_GPU`: Set to `true` to attempt GPU acceleration (CUDA for NVIDIA, MPS/CoreML for macOS). Set to `false` to force CPU usage.
        *   `MODEL_PACK_NAME`: The `insightface` model pack to use (e.g., `buffalo_l`). See `insightface` documentation for options.
        *   `FACE_MATCH_THRESHOLD`: Adjust the cosine distance threshold for face recognition (default in `db_utils.py` if not set).

7.  **Add Known Faces:**
    *   Use the `add_person.py` script to add faces to the database:
        ```bash
        python add_person.py --name "Your Name"
        ```
    *   Follow the on-screen prompts to capture face images using your webcam.

## Running the Application

1.  **Ensure Drone is Connected:** Power on your Tello drone, ensure its battery is charged (>25% recommended), and connect your computer to its Wi-Fi network.
2.  **Activate Virtual Environment:**
    ```bash
    source venv/bin/activate # Or `venv\Scripts\activate` on Windows
    ```
3.  **Run the Main Drone Script:**
    ```bash
    python main.py
    ```
    *   The application will attempt to connect to the drone, take off, start the video stream, and perform person detection/tracking/recognition.
    *   Watch the console output during initialization to see if CUDA or CoreML (MPS) is being used.
    *   A window will pop up displaying the drone's video feed with detected people, faces, tracking IDs, and the current target/distance.

## Running the Webcam Simulator

1.  **Activate Virtual Environment:**
    ```bash
    source venv/bin/activate # Or `venv\Scripts\activate` on Windows
    ```
2.  **Run the Simulator Script:**
    ```bash
    python webcam_simulator.py
    ```
    *   This uses your built-in webcam (index 0) instead of the drone.
    *   It performs the same detection, tracking, recognition, and distance calculations.
    *   Instead of sending commands to a drone, it prints simulated RC commands (`SIM CMD: ...`) to the console.
    *   A window will pop up displaying the annotated webcam feed.

## Usage

*   **'q' Key:** Press 'q' while the video window is active to gracefully land the drone (in `main.py`) or exit the simulator (`webcam_simulator.py`).
*   **'e' Key:** Press 'e' for an emergency stop (cuts drone motors immediately). Only applicable in `main.py`. Use with caution.

## Troubleshooting

*   **`FATAL: database "face_drone_db" does not exist`**: Ensure you have created the database in PostgreSQL (Step 5).
*   **`Error: vector type not found in the database`** or similar: Make sure you have enabled the `pgvector` extension *within* the correct database (Step 5).
*   **`AttributeError: module 'camera_utils' has no attribute 'CAMERAS'`** (or similar): Ensure you are using the correct variable name (`CAMERA_PARAMS`) in your code if accessing parameters directly.
*   **Model Not Found:**
    *   `yolov10s.pt`: Make sure you have downloaded the weights and placed them in the project root directory (Step 2).
    *   InsightFace models: Check network connection; they should download automatically. Verify `MODEL_PACK_NAME` in `.env` is valid.
*   **GPU Issues (CUDA/MPS):**
    *   If `USE_GPU=true` but it falls back to CPU, check the logs for errors.
    *   **CUDA:** Ensure `onnxruntime-gpu` is installed, NVIDIA drivers, CUDA Toolkit, and cuDNN are correctly installed and compatible with your `onnxruntime-gpu` version.
    *   **MPS (macOS):** Ensure you have `onnxruntime` installed (v1.10+), you are on a Mac with Apple Silicon, and macOS is up-to-date. CoreML provider might have specific version requirements.
*   **Drone Connection Issues:** Verify your computer is connected to the Tello drone's Wi-Fi. Check firewall settings. Ensure the drone battery is sufficiently charged.
*   **Drone Takeoff Issues:** Check drone battery level (logs should show percentage). The drone might refuse takeoff if the battery is too low (e.g., < 20-25%).
*   **Performance:** If detection/tracking/recognition is slow, consider using a smaller `MODEL_PACK_NAME`, ensuring `USE_GPU` is correctly set and *active* (check logs). 