# Agentic Backend for private 6G Orchestration and Deployment

This repository provides an Agentic Backend designed to help users deploy ai models on edge or cloud environments. The backend is built using FastAPI and Uvicorn, ensuring a scalable and high-performance API.

## Features

 - Deploy private 6G infrastructure seamlessly

 - Edge and cloud deployment support

 - Secure .env-based configuration management

 - Docker support for containerized deployment

## Prerequisites

Ensure you have the following installed before running the application:

- Python 3.8+

- pip (Python package manager)

- Docker (for containerized deployment)

- Git (optional, for cloning the repository)

Installation and Setup

1. Clone the Repository

```bash
git clone https://github.com/your-repo/agentic-backend.git
cd agentic-backend
```

2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate```
```

3. Install Dependencies

```bash
pip install -r requirements.txt
```

4. Configure the Environment Variables

Copy the example environment file and modify it as needed:

```bash
cp .env.example .env
```

Open .env and update the values with configurations, llm credentials, and other necessary details.

## Running the Application

Run with Uvicorn (Development Mode)

```bash
uvicorn main:app
```

API Docs (Swagger UI): `http://localhost:8000/docs`

API Redoc: `http://localhost:8000/redoc`

## Running with Docker

1. Build the Docker Image

```bash
docker build -t agentic-backend .
```

2. Run the Container with Your .env File

```bash
docker run --env-file .env -p 8000:8000 agentic-backend
```

3. Running in Detached Mode

```bash
docker run -d --env-file .env -p 8000:8000 agentic-backend
```

