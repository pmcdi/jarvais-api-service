# Jarvais API Service

Microservice API for jarvAIs Analyzer, Trainer, and Inference services. Built with FastAPI for better performance, automatic API documentation, and improved type safety.

## Features

- **FastAPI Framework**: Modern, fast web framework with automatic API documentation
- **Automatic Validation**: Request/response validation using Pydantic models
- **Interactive API Docs**: Swagger UI and ReDoc documentation at `/docs` and `/redoc`
- **Rate Limiting**: Built-in rate limiting for production use
- **Redis Support**: Optional Redis backend for session storage
- **CORS Support**: Configurable CORS middleware
- **Health Checks**: Built-in health check endpoint
- **Async Support**: Full async/await support for better performance

## Quick Start

### 1. Install Dependencies

```bash
# Install pixi if you haven't already
curl -fsSL https://pixi.sh/install.sh | bash

# Install project dependencies
pixi install
```

#### OR install Docker

### 2. Run the Application

#### Development Mode
```bash
pixi run dev
```

#### Development Mode with Auto-reload
```bash
pixi run debug
```

#### Production Mode
```bash
pixi run prod
```

#### Using Pixi Shell
```bash
# Enter pixi environment
pixi shell

# Then run directly
python run_fastapi.py

# Or use uvicorn directly
uvicorn src.main:app --host 0.0.0.0 --port 8888 --reload
```

#### Via Docker
```bash
# Build the Docker image
docker build . 

# Spin up Docker compose and detach to run in background
docker compose up -d
```

### 3. Access the API

- **API Documentation**: http://localhost:8888/docs
- **Alternative Docs**: http://localhost:8888/redoc
- **Health Check**: http://localhost:8888/health

## API Endpoints

### File Upload
- **POST** `/upload` - Upload CSV file and create analyzer

### Visualizations
- **GET** `/visualization/{analyzer_id}/correlation_heatmap` - Get correlation heatmap
- **GET** `/visualization/{analyzer_id}/frequency_heatmap` - Get frequency heatmap
- **GET** `/visualization/{analyzer_id}/pie_chart` - Get pie chart
- **GET** `/visualization/{analyzer_id}/umap_scatterplot` - Get UMAP scatterplot

### Analyzer Management
- **GET** `/analyzers` - List all analyzers
- **GET** `/analyzers/{analyzer_id}` - Get analyzer info
- **DELETE** `/analyzers/{analyzer_id}` - Delete analyzer

### Trainer (Machine Learning)
- **POST** `/trainers` - Create and train a model
- **GET** `/trainers` - List all trainers
- **GET** `/trainers/{trainer_id}` - Get trainer info
- **GET** `/trainers/{trainer_id}/results` - Get training results
- **DELETE** `/trainers/{trainer_id}` - Delete trainer

### Inference (Predictions)
- **POST** `/trainers/{trainer_id}/infer` - Generate predictions from CSV file
- **POST** `/trainers/{trainer_id}/infer/json` - Generate predictions from JSON

### System
- **GET** `/health` - Health check

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_HOST` | `localhost` | Redis server host |
| `REDIS_PORT` | `6379` | Redis server port |
| `REDIS_DB` | `0` | Redis database number |
| `SESSION_TTL` | `3600` | Session timeout in seconds |
| `UPLOAD_FOLDER` | `uploads` | Upload directory |
| `MAX_CONTENT_LENGTH` | `104857600` | Max file size (100MB) |
| `ALLOWED_ORIGINS` | `*` | CORS allowed origins |
| `TRUSTED_HOSTS` | - | Trusted host middleware |
| `PORT` | `8888` | Server port |
| `HOST` | `0.0.0.0` | Server host |
| `RELOAD` | `false` | Enable auto-reload |
| `LOG_LEVEL` | `info` | Logging level |


### Available Tasks

| Task | Command | Description |
|------|---------|-------------|
| `dev` | `pixi run dev` | Start development server |
| `prod` | `pixi run prod` | Start production server |
| `test` | `pixi run test` | Run test suite |
| `docs` | `pixi run docs` | Echo CLI to help user find API docs |

## Docker Support

The existing Dockerfile should work with the FastAPI version. You can build and run:

```bash
docker build -t jarvais-api:latest .
docker compose up
```