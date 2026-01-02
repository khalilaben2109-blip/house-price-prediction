# 🚀 Deployment Guide

This guide explains how to deploy the House Price Prediction project to various platforms.

## 📋 Table of Contents
- [Local Development](#local-development)
- [Streamlit Cloud](#streamlit-cloud)
- [Heroku](#heroku)
- [Docker](#docker)
- [GitHub Pages](#github-pages)

## 🏠 Local Development

### Quick Start
```bash
git clone https://github.com/YOUR_USERNAME/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
python start.py
```

### Development Server
```bash
# Main interface
streamlit run app.py

# Database interface
streamlit run database_app.py --server.port 8502
```

## ☁️ Streamlit Cloud

### Prerequisites
- GitHub repository
- Streamlit Cloud account

### Steps
1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select repository: `house-price-prediction`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Environment Variables** (if needed)
   - Add any required secrets in Streamlit Cloud settings

### Configuration
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
enableCORS = false
enableXsrfProtection = false
```

## 🚀 Heroku

### Prerequisites
- Heroku account
- Heroku CLI installed

### Files Required

**Procfile**:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

**runtime.txt**:
```
python-3.9.16
```

**setup.sh**:
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

### Deployment Steps
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Open app
heroku open
```

## 🐳 Docker

### Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - PYTHONPATH=/app
```

### Commands
```bash
# Build image
docker build -t house-price-prediction .

# Run container
docker run -p 8501:8501 house-price-prediction

# Using docker-compose
docker-compose up
```

## 📄 GitHub Pages (Static Documentation)

For hosting documentation and static content:

### Setup
1. Create `docs/` folder
2. Add documentation files
3. Enable GitHub Pages in repository settings
4. Select source: `docs/` folder

### GitHub Actions Workflow
`.github/workflows/docs.yml`:
```yaml
name: Deploy Documentation

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install mkdocs mkdocs-material
    
    - name: Build docs
      run: mkdocs build
    
    - name: Deploy to GitHub Pages
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./site
```

## 🔧 Environment Variables

### Required Variables
```bash
# Optional: Database configuration
DATABASE_URL=sqlite:///data/house_prices.db

# Optional: API keys for external data sources
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### Setting Variables

**Streamlit Cloud**:
- Go to app settings → Secrets
- Add variables in TOML format

**Heroku**:
```bash
heroku config:set DATABASE_URL=your_database_url
```

**Docker**:
```bash
docker run -e DATABASE_URL=your_database_url -p 8501:8501 house-price-prediction
```

## 📊 Performance Optimization

### For Production
1. **Caching**:
   ```python
   @st.cache_data
   def load_data():
       # Your data loading logic
       pass
   ```

2. **Resource Limits**:
   - Limit dataset size for web deployment
   - Use model compression techniques
   - Implement lazy loading

3. **Database Optimization**:
   - Use connection pooling
   - Implement proper indexing
   - Regular database maintenance

## 🔍 Monitoring

### Health Checks
```python
# Add to your Streamlit app
def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}
```

### Logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## 🚨 Troubleshooting

### Common Issues

**Memory Issues**:
- Reduce dataset size
- Use data sampling
- Implement pagination

**Slow Loading**:
- Add caching decorators
- Optimize data loading
- Use progress bars

**Port Issues**:
- Check port availability
- Use environment variables for ports
- Configure firewall settings

### Debug Mode
```bash
# Run with debug information
streamlit run app.py --logger.level=debug
```

## 📝 Deployment Checklist

- [ ] Code tested locally
- [ ] Requirements.txt updated
- [ ] Environment variables configured
- [ ] Database migrations completed
- [ ] Static files optimized
- [ ] Security settings reviewed
- [ ] Performance tested
- [ ] Monitoring configured
- [ ] Backup strategy implemented
- [ ] Documentation updated

## 🔗 Useful Links

- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-cloud)
- [Heroku Python Guide](https://devcenter.heroku.com/articles/getting-started-with-python)
- [Docker Documentation](https://docs.docker.com/)
- [GitHub Actions](https://docs.github.com/en/actions)

---

For more help, check the [main README](README.md) or open an [issue](https://github.com/YOUR_USERNAME/house-price-prediction/issues).