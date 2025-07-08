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
        ["Model", "Model Size (parameters)", "LiveCodeBench Pass@1 (%)"],
        ["DeepCoder(ours)", "16B", 60.8],
        ["o3-mini (low)", "N/A", 61.2],
        ["o1", "N/A", 59.5],
        ["R1-Distilled-32B", "32B", 57.2],
        ["R1-Distilled-14B", "14B", 53.0]
    ]
}
```

---

## 🚀 Quick Start

### 🐳 Docker Deployment (Recommended)

```bash
# 1. Clone and setup
git clone https://github.com/rows/rows_vision.git
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

**Linux/macOS:**
```bash
# 1. Clone and setup
git clone https://github.com/rows/rows_vision.git
cd rows_vision
chmod +x setup.sh && ./setup.sh

# 2. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies and run
pip install -r requirements.txt
nano .env  # Add API keys
python main.py
```

**Windows (PowerShell):**
```powershell
# 1. Clone repository
git clone https://github.com/rows/rows_vision.git
cd rows_vision

# 2. Copy environment template
Copy-Item ".env.example" ".env"

# 3. Edit .env file with your API keys
notepad .env

# 4. Create and activate virtual environment
python -m venv venv
venv\Scripts\Activate.ps1

# 5. Install dependencies and run
pip install -r requirements.txt
python main.py
```

**Windows (Git Bash - Alternative):**
```bash
# If you have Git Bash installed, you can use the Linux/macOS commands:
chmod +x setup.sh
./setup.sh
# Then follow the Linux/macOS steps above
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

**🎯 Unified Output Format:** All endpoints return data in the same format - an array where the first row contains headers and subsequent rows contain data values.

**🚀 Endpoint Comparison:**

| Endpoint | Use Case | Speed | Features |
|----------|----------|-------|----------|
| `/api/run` | General purpose | Medium | Two-step analysis, dual models |
| `/api/run-file` | Local files | Medium | Same as run + local file support |
| `/api/run-one-shot` | Tables/receipts | Fastest | Direct extraction, single step |
| `/api/classify-with-instructions` | Custom extraction | Fast | Custom instructions, single model |

### `POST /api/run`

Process an image from a URL using two-step analysis (classification + extraction).

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
    ["Month", "Sales", "Profit"],
    ["January", 1000, 200],
    ["February", 1200, 300],
    ["March", 950, 180]
  ],
  "metrics": {
    "total_time": 2.345
  }
}
```

### `POST /api/run-file`

Process an image from URL or local file path. Same as `/api/run` but supports local files.

**Request:**
```json
{
  "image_url": "https://example.com/chart.png",
  // OR
  "file_path": "/path/to/local/image.jpg",
  "model_classification": "anthropic",
  "model_extraction": "anthropic",
  "time_outputs": false
}
```

**Response:** Same format as `/api/run` endpoint.

### `POST /api/run-one-shot`

Process an image with direct data extraction (skips secondary analysis). Fastest option for tables, receipts, and charts with clear data labels.

**Request:**
```bash
curl -X POST 'http://localhost:8080/api/run-one-shot' \
--header 'Content-Type: application/json' \
--data '{
  "image_url": "https://example.com/table.png",
  "model_classification": "google",
  "model_extraction": "google",
  "time_outputs": true
}'
```

**Features:**
- **Fastest Processing**: Skips secondary analysis pipeline
- **Direct Extraction**: Uses classification results directly
- **Best For**: Tables, receipts, and charts with clear labels
- **Same Format**: Returns same output format as other endpoints

**Response:**
```json
{
  "result": [
    ["Product", "Price", "Stock"],
    ["Laptop", "$999", "25"],
    ["Mouse", "$29", "150"],
    ["Keyboard", "$79", "80"]
  ],
  "metrics": {
    "total_time": 1.8
  }
}
```

### `POST /api/classify-with-instructions`

Process an image with custom instructions using a single AI model. This endpoint combines classification and extraction in one step using system/user message structure.

**Request:**
```bash
curl -X POST 'http://localhost:8080/api/classify-with-instructions' \
--header 'Content-Type: application/json' \
--data '{
  "image_url": "https://example.com/chart.png",
  "instructions": "Extract only the revenue data from this chart, focusing on Q1-Q4 values",
  "model": "google",
  "time_outputs": true
}'
```

**Python Example:**
```python
import requests

url = "http://localhost:8080/api/classify-with-instructions"
payload = {
    "image_url": "https://example.com/chart.png",
    "instructions": "Extract only the revenue data from this chart, focusing on Q1-Q4 values",
    "model": "google",  # or "openai", "anthropic"
    "time_outputs": True,
    "include_name": False  # optional - set to True to include chart name
}

response = requests.post(url, json=payload)
print(response.json())
```

**Request Parameters:**
- `image_url` (required): URL of the image to process
- `file_path` (alternative): Local file path (use instead of image_url)
- `instructions` (optional): Custom instructions for data extraction (if empty, passes only image)
- `model` (required): AI model to use (`google`, `openai`, or `anthropic`)
- `time_outputs` (optional): Include timing metrics in response
- `include_name` (optional): Include chart name in response (default: false)

**Response (default format - data points only):**
```json
{
  "result": [
    ["Quarter", "Revenue"],
    ["Q1", "150000"],
    ["Q2", "180000"],
    ["Q3", "220000"],
    ["Q4", "280000"]
  ],
  "metrics": {
    "total_time": 3.2
  }
}
```

**Response (with include_name=true):**
```json
{
  "result": {
    "name": "Revenue Chart Q1-Q4",
    "data_points": [
      ["Quarter", "Revenue"],
      ["Q1", "150000"],
      ["Q2", "180000"],
      ["Q3", "220000"],
      ["Q4", "280000"]
    ]
  },
  "metrics": {
    "total_time": 3.2
  }
}
```

**Key Features:**
- **Single Model Processing**: No ensemble, direct results
- **Custom Instructions**: Tailor extraction to specific needs (optional)
- **System/User Prompts**: Uses advanced prompt structure
- **Supported Models**: Google Gemini, OpenAI, and Anthropic Claude
- **Combined Operation**: Classification and extraction in one call
- **Simplified Output**: Returns data points array directly (optional name parameter)
- **Flexible Input**: Works with or without custom instructions

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
│   ├── main.py             # Flask application
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

- ~~Support user prompt for finer operations~~ ✅ **Done**
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

## 📄 Research Paper

This work is based on research studying multimodal large language models for visual data extraction from charts and tables. 

**📖 Paper**: [Rows Vision: Multimodal Large Language Models for Visual Data Extraction](./paper/RowsVision_WhitePaper.pdf) *(White Paper)*

<!-- Once uploaded to arXiv, replace with:
**📖 Paper**: [Rows Vision: Multimodal Large Language Models for Visual Data Extraction](https://arxiv.org/abs/XXXX.XXXXX) *(arXiv preprint)*
-->

**🎯 Citation**:
```bibtex
@techreport{samagaio2025rowsvision,
  title={Rows Vision: Multimodal Large Language Models for Visual Data Extraction},
  author={Samagaio, {\'A}lvaro Mendes and Cruz, Henrique},
  institution={Rows.com},
  address={Porto, Portugal},
  year={2025},
  type={White Paper},
  note={Available at: \url{https://github.com/rows/rows_vision/blob/main/paper/RowsVision_WhitePaper.pdf}}
}
```

**For arXiv submission (when ready):**
```bibtex
@misc{samagaio2025rowsvision,
  title={Rows Vision: Multimodal Large Language Models for Visual Data Extraction},
  author={Samagaio, {\'A}lvaro Mendes and Cruz, Henrique},
  year={2025},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
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