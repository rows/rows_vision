# 📊 rows_vision

`rows_vision` is an open-source API service that extracts structured data from visual content like charts, receipts, and screenshots using vision-based classifiers and LLMs. It's built for fast local deployment, and works entirely in memory — no cloud storage required.

### Supported types: 
- 1: Line chart (single line)
- 2: Line chart (multiple lines)
- 3: Bar/column chart
- 4: Scatter plot
- 5: Pie or doughnut chart
- 6: Table
- 7: Receipt/Invoice
- 8: Other (e.g., infographic with extractable data)

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue" />
  <img src="https://img.shields.io/badge/flask-powered-brightgreen" />
  <img src="https://img.shields.io/badge/docker-ready-blue" />
  <img src="https://img.shields.io/badge/AI-multi--model-orange" />
</p>

---

## 🚀 Features

- **Multi-Model AI Support**: Choose from Anthropic Claude, OpenAI GPT-4, Google Gemini, or Groq models
- **Chart Analysis**: Extract data from line charts, bar charts, scatter plots, pie charts
- **Table & Receipt Processing**: Parse structured data from tables and receipts
- **Flexible Input**: Process images from URLs or local files
- **In-Memory Processing**: No cloud storage required - everything runs locally
- **Docker Ready**: Easy deployment with Docker containers
- **Production Ready**: Built-in health checks, logging, and error handling
- **Performance Metrics**: Optional timing information for monitoring

---

## 🧠 Example Use Case

Upload the URL of chart screenshot and receive a structured JSON like:

```json
{
    "result": [
        {
            "X": 130,
            "Y": "2000"
        },
        {
            "X": 90,
            "Y": "2005"
        },
        {
            "X": 30,
            "Y": "2010"
        },
        {
            "X": 25,
            "Y": "2015"
        },
        {
            "X": 60,
            "Y": "2020"
        },
        {
            "X": 110,
            "Y": "2025"
        }
    ]
}
```

---

## 🚀 Quick Start

### 🐳 Docker Deployment (Recommended)

```bash
# 1. Clone and setup
git clone https://github.com/yourname/rows_vision.git
cd rows_vision

# 2. Run setup script
chmod +x setup.sh
./setup.sh

# 3. Add your API keys to .env
nano .env  # Add at least one API key

# 4. Build and run with Docker
docker build -t rows-vision .
docker run -d --name rows-vision-api -p 8080:8080 --env-file .env rows-vision

# 5. Test the API (wait 30 seconds for startup)
sleep 30
curl http://localhost:8080/health
```

### 🐍 Local Python Development

```bash
# 1. Clone and setup
git clone https://github.com/yourname/rows_vision.git
cd rows_vision
chmod +x setup.sh && ./setup.sh

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies and run
pip install -r requirements.txt
nano .env  # Add API keys
python main.py
```

**Why Docker?** | Docker | Local Python
---|---|---
**Setup Time** | 5 minutes | 10-15 minutes
**Dependencies** | Automatic | Manual
**Consistency** | Same everywhere | "Works on my machine"
**Production Ready** | Yes | Needs additional setup

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with your API credentials:

```env
# Required: At least one AI API key
API_KEY_ANTHROPIC=sk-ant-your-key-here
API_KEY_OPENAI=sk-your-key-here
API_KEY_GEMINI=AIzaSy-your-key-here
API_KEY_GROQ=gsk_your-key-here

# Optional: Model Configuration
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
OPENAI_MODEL=gpt-4o
GEMINI_MODEL=gemini-2.0-flash
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct

# Optional: Server Settings
HOST=0.0.0.0
PORT=8080
DEBUG=false
LOG_LEVEL=INFO
MAX_FILE_SIZE=10485760  # 10MB
```

### Supported AI Models

| Model | Classification | Extraction | Notes |
|-------|---------------|------------|-------|
| `anthropic` | ✅ | ✅ | Claude Sonnet, high accuracy |
| `openai` | ✅ | ✅ | GPT-4o, good performance |
| `google` | ✅ | ✅ | Gemini Flash, fast processing |
| `groq` | ✅ | ✅ | Llama, cost-effective |

---

## 🔌 API Endpoints

### `POST /api/run`

Process an image from a URL.

**Request:**
```bash
curl -X POST 'http://localhost:8080/api/run' \
--header 'Content-Type: application/json' \
--data '{
  "image_url": "https://pbs.twimg.com/media/GoCeF4wbwAE24ln?format=jpg&name=large",
  "model_classification": "anthropic",
  "model_extraction": "anthropic",
  "time_outputs": true
}'
```

**Python Example:**
```python
import requests

url = "http://localhost:8080/api/run"
payload = {
    "image_url": "https://pbs.twimg.com/media/GoCeF4wbwAE24ln?format=jpg&name=large",
    "model_classification": "anthropic",
    "model_extraction": "anthropic",
    "time_outputs": True
}

response = requests.post(url, json=payload)
print(response.json())
```

**Response:**
```json
{
  "result": [
    {"Month": "January", "Sales": 1000, "Profit": 200}
  ],
  "metrics": {
    "total_time": 2.345
  }
}
```

### `POST /api/run-file`

Process an image from URL or local file path.

```json
{
  "image_url": "https://example.com/chart.png",
  // OR
  "file_path": "/path/to/local/image.jpg",
  "model_classification": "anthropic",
  "model_extraction": "anthropic"
}
```

---

## 🚀 Production Deployment

### Docker Compose (Recommended)

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  rows-vision:
    build: .
    container_name: rows-vision-api
    ports:
      - "8080:8080"
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
docker-compose up -d
```

### Cloud Deployment

**Google Cloud Run:**
```bash
gcloud run deploy rows-vision --source . --platform managed --allow-unauthenticated
```

**AWS ECS / Digital Ocean / Others:**
Use the Docker image built above with your preferred container orchestration platform.

### Traditional Deployment

```bash
# Using Gunicorn (production WSGI server)
pip install gunicorn
gunicorn --bind 0.0.0.0:8080 --workers 4 --timeout 120 main:app
```

---

## 🔍 Monitoring & Health

```bash
# Health check
curl http://localhost:8080/health

# Docker container status
docker ps
docker logs rows-vision-api --tail 50 -f

# Resource monitoring
docker stats rows-vision-api
```

---

## 🏗 Technical Details

**Supported Formats:** PNG, JPG, JPEG, GIF, WEBP, HEIC  
**Chart Types:** Line, Bar, Scatter, Pie, Tables, Receipts  
**Processing:** In-memory, no file storage required  
**Architecture:** Flask API + AI model backends  

---

## 📁 Project Structure

```
rows_vision/
├── src/                     # Source code
│   ├── config.py           # Configuration
│   ├── image_analyzer.py   # Data extraction
│   ├── image_classifier.py # Image classification
│   └── rows_vision.py      # Main orchestrator
├── prompts/                # AI prompt templates
├── main.py                 # Application entry point
├── requirements.txt        # Dependencies
├── Dockerfile             # Container definition
├── setup.sh               # Automated setup script
└── .env.example          # Environment template
```

---

## 🚧 To-Do

- Support user prompt for finer operations
- ~~Improve error handling~~ ✅ **Done**
- ~~Docker deployment~~ ✅ **Done** 
- ~~Production-ready logging~~ ✅ **Done**
- Support for batch processing
- PDF processing improvements

---

## 🐛 Troubleshooting

**Missing API Keys:**
```bash
# Check if keys are loaded
docker run --env-file .env rows-vision python -c "import os; print('Keys loaded:', bool(os.getenv('API_KEY_ANTHROPIC')))"
```

**Container Issues:**
```bash
# Check logs
docker logs rows-vision-api

# Debug mode
docker run -it --env-file .env -e DEBUG=true rows-vision
```

**Port Conflicts:**
```bash
# Use different port
docker run -d -p 8081:8080 --env-file .env rows-vision
```

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

---

## 🙌 Contributions

PRs and issues are welcome. Please fork the repo and submit changes via pull request.

---

## 📣 Maintainer

Created by [@asamagaio](https://github.com/asamagaio) at Rows.