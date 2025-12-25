# Interview Guide

## Overview
Interview Guide is a comprehensive platform designed to streamline the interview preparation process. It provides functionalities including job description ingestion, automated question generation, answer evaluation, personalized resource recommendations, profile tracking, multi-user support, and user preferences management. The platform can be accessed via a CLI-like mode or through the FastAPI backend described below. All backend code now lives inside the `backend/` directory to keep future frontend work separate.

## Features
- Job Description (JD) ingestion for tailored interview preparation
- Automated question generation based on job requirements
- Evaluation of candidate responses to track progress
- Personalized resource recommendations for skill enhancement
- Profile tracking to monitor improvement over time
- Multi-user support with user preferences management
- Clean FastAPI backend for custom UI integration

## Architecture
The backend is located inside `backend/`, with core modules under `backend/src/interview_guide/`:

- **Agents:** Orchestrate workflows and coordinate interactions between components.
- **Storage:** Manage user data, profiles, and session information.
- **Tools:** Implement core functionalities such as JD ingestion, question generation, and answer evaluation.
- **Graph:** Handle knowledge graph operations and relationships.
- **Router:** Manage routing of intents and requests.
- **Intents:** Define user intents and corresponding actions.
- **API:** FastAPI entry points live under `interview_guide.api`, exposing the orchestrated agent and all power tools.

## Setup

Use the `uv` package manager to install dependencies and synchronize the environment (run commands from the `backend/` directory):

```bash
cd backend
uv sync
```

## Configuration

Create a `.env` file in the project root (or within `backend/`) with the following environment variables:

```
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
INTERVIEW_GUIDE_DB=/absolute/path/to/interview_guide.db
```

Set `INTERVIEW_GUIDE_DB` to any persistent location (e.g., `/var/interview_guide/interview_guide.db`). SQLite stores all user sessions/scores in that file, so keeping it outside the repo ensures data survives backend restarts and deployments. On servers or containers, mount that path as persistent storage.

These API keys are required for accessing external services used in job description processing and other features.

## Running the Application

### Backend API Mode

To launch the FastAPI backend (which powers the upcoming UI), run:

```bash
cd backend
uv run uvicorn api_server:app --reload
```

`api_server.py` ensures the `src/` directory is on `PYTHONPATH`, so no extra configuration is required. Once running, all endpoints are available under `http://localhost:8000/api`.

### CLI-like Mode

For command-line interactions, such as working directly with the knowledge graph or agents, use:

```bash
cd backend
uv run python -m interview_guide.graph
```

### Frontend UI (static prototype)

A lightweight client lives in `frontend/`. Serve it locally (any static server works); for example:

```bash
cd frontend
python3 -m http.server 4173
```

Then visit `http://localhost:4173` in your browser while the backend is running on `http://localhost:8000`. The UI calls `/api/agent/execute` by default; adjust `window.API_BASE_URL` in `frontend/app.js` if you host the API elsewhere.

## Example Usage

### CLI Mode

You can interact with the system using the CLI-like mode by running:

```bash
cd backend
uv run python -m interview_guide.graph
```

Once running, you can enter queries such as:

- `Generate questions on SQL joins`
- `Here is the JD text: We are looking for a data analyst with experience in Python and SQL.`
- `Evaluate these answers: Candidate's response about polymorphism in OOP.`
- `Show my profile`
- `Update my prefs: free only, YouTube`

These commands allow you to generate interview questions, ingest job descriptions, evaluate candidate answers, view your profile, and update your preferences respectively.

### Backend API Mode

With the FastAPI server running, you can call endpoints directly. For example, to run a full agent turn:

```bash
curl -X POST http://127.0.0.1:8000/api/agent/execute \
  -H "Content-Type: application/json" \
  -d '{"query":"Give me resources for decision trees","user_id":"demo"}'
```

Power-tool endpoints remain available for admin or automation workflows. Their implementations reside under `backend/src/interview_guide/api/power_tools/`:

- `POST /api/jd/fetch`
- `POST /api/jd/profile`
- `POST /api/questions`
- `POST /api/evaluation`
- `POST /api/profile/summary`
- `POST /api/recommendations`
- `POST /api/sessions`
- `GET /api/sessions`
- `GET /api/sessions/{id}/questions`
- `GET /api/users`
- `GET /api/users/{user_id}`

## Deployment

Deploy the backend to your preferred host (e.g., cloud VM, container service) using the FastAPI server. Typical steps:

1. Ensure environment variables are set for API keys **and** `INTERVIEW_GUIDE_DB` (point it at a persistent location/volume).
2. Install dependencies inside `backend/` with `uv sync` (or bake them into a container image).
3. Start the API with `cd backend && uv run uvicorn api_server:app --host 0.0.0.0 --port 8000`.
4. Place the service behind your load balancer or gateway.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.
