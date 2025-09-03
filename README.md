# Interview Guide

## Overview
Interview Guide is a comprehensive platform designed to streamline the interview preparation process. It provides functionalities including job description ingestion, automated question generation, answer evaluation, personalized resource recommendations, profile tracking, multi-user support, and user preferences management. The platform is accessible via an intuitive Streamlit-based user interface as well as a CLI-like mode for advanced usage.

## Features
- Job Description (JD) ingestion for tailored interview preparation
- Automated question generation based on job requirements
- Evaluation of candidate responses to track progress
- Personalized resource recommendations for skill enhancement
- Profile tracking to monitor improvement over time
- Multi-user support with user preferences management
- Clean and interactive Streamlit UI for ease of use

## Architecture
The project is organized into modular components under `src/interview_guide/`:

- **Agents:** Orchestrate workflows and coordinate interactions between components.
- **Storage:** Manage user data, profiles, and session information.
- **Tools:** Implement core functionalities such as JD ingestion, question generation, and answer evaluation.
- **Graph:** Handle knowledge graph operations and relationships.
- **Router:** Manage routing of intents and requests.
- **Intents:** Define user intents and corresponding actions.
- **UI:** The user interface entry point is `app.py`, built with Streamlit.

## Setup

Use the `uv` package manager to install dependencies and synchronize the environment:

```bash
uv sync
```

## Configuration

Create a `.env` file in the project root with the following environment variables:

```
GOOGLE_API_KEY=your_google_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

These API keys are required for accessing external services used in job description processing and other features.

## Running the Application

### Streamlit UI Mode

To launch the interactive Streamlit user interface, run:

```bash
uv run streamlit run app.py
```

### CLI-like Mode

For command-line interactions, such as working directly with the knowledge graph or agents, use:

```bash
uv run python -m interview_guide.graph
```

## Example Usage

### CLI Mode

You can interact with the system using the CLI-like mode by running:

```bash
uv run python -m interview_guide.graph
```

Once running, you can enter queries such as:

- `Generate questions on SQL joins`
- `Here is the JD text: We are looking for a data analyst with experience in Python and SQL.`
- `Evaluate these answers: Candidate's response about polymorphism in OOP.`
- `Show my profile`
- `Update my prefs: free only, YouTube`

These commands allow you to generate interview questions, ingest job descriptions, evaluate candidate answers, view your profile, and update your preferences respectively.

### Streamlit UI Mode

Launch the interactive user interface with:

```bash
uv run streamlit run app.py
```

Within the UI, you can enter queries in the textbox to perform various actions, including:

- Job Description ingestion for personalized preparation
- Generating interview questions based on job requirements
- Evaluating candidate answers to track progress
- Viewing your user profile and progress
- Updating your preferences for recommended resources

This intuitive interface enables seamless interaction with all core functionalities of the Interview Guide platform.

## Deployment

This project can be deployed seamlessly on [Hugging Face Spaces](https://huggingface.co/spaces) to provide easy access and sharing. Simply upload the repository, ensure dependencies are listed, and configure environment variables in the Space settings.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License.