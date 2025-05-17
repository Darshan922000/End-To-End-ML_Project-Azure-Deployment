# End-To-End ML Project: Azure Deployment

This project demonstrates an end-to-end machine learning pipeline with Azure deployment capabilities. The project includes data exploration, model training, and a production-ready API deployment.

## Project Structure

```
├── src/                    # Source code
│   ├── components/        # ML pipeline components
│   ├── pipeline/         # Pipeline configuration
│   ├── utils.py          # Utility functions
│   ├── logger.py         # Logging configuration
│   └── exception.py      # Custom exception handling
├── Notebook/             # Jupyter notebooks
│   ├── Data/            # Dataset files
│   ├── EDA.ipynb        # Exploratory Data Analysis
│   └── Model Training.ipynb  # Model training notebook
├── logs/                 # Log files
├── data storage/         # Data storage directory
├── app.py               # FastAPI application
├── dockerfile           # Docker configuration
├── docker-compose.yml   # Docker compose configuration
├── pyproject.toml       # Project dependencies (Poetry)
└── poetry.lock         # Locked dependencies
```

## Setup Instructions

1. Clone the repository
2. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
3. Install dependencies:
   ```bash
   poetry install
   ```
4. Activate the virtual environment:
   ```bash
   poetry shell
   ```

## Running the Application

1. Start the application using Docker:
   ```bash
   docker-compose up --build
   ```
   
   Or run directly with Python:
   ```bash
   python app.py
   ```

2. The API will be available at `http://localhost:8000`

## Development

- Use the Jupyter notebooks in the `Notebook` directory for data exploration and model training
- The trained model and pipeline components are stored in the `src` directory
- Logs are stored in the `logs` directory

## Dependencies

The project uses Poetry for dependency management. Key dependencies include:
- FastAPI
- CatBoost
- Pandas
- NumPy
- Scikit-learn

See `pyproject.toml` for the complete list of dependencies.

## Docker Support

The project includes Docker support for containerized deployment:
- `dockerfile`: Defines the container environment
- `docker-compose.yml`: Orchestrates the services
- `.dockerignore`: Specifies files to exclude from the Docker build

## License

This project is licensed under the MIT License.