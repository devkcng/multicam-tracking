# FastAPI Project Setup Instructions

Follow these steps to set up and run the FastAPI project:

## 1. Create a Virtual Environment

1. Navigate to the project directory:

    ```bash
    cd multicam-tracking-be
    ```

2. Create a virtual environment:

    ```bash
    python3 -m venv venv
    ```

3. Activate the virtual environment:
    - On Linux/Mac:

      ```bash
      source venv/bin/activate
      ```

    - On Windows:

      ```bash
      venv\Scripts\activate
      ```

## 2. Install Requirements

1. Ensure the virtual environment is activated.
2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## 3. Run the FastAPI Project

1. Start the FastAPI server:

    ```bash
    uvicorn main:app --reload
    ```

    Replace `main` with the name of your main Python file if different.

2. Access the application:
    Open your browser and navigate to `http://127.0.0.1:8000`.

<!-- 3. API Documentation:
    - Swagger UI: `http://127.0.0.1:8000/docs`
    - ReDoc: `http://127.0.0.1:8000/redoc` -->
