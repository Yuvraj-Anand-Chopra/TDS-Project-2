# Deployment Guide

This guide covers deploying the LLM Analysis Quiz Solver to various platforms.

## Pre-deployment Checklist

- All tests passing: pytest tests/ -v
- Environment variables configured in .env
- Gemini API key is valid and has quota
- GitHub repository is public with MIT LICENSE
- All sensitive data removed from codebase
- Docker image builds successfully

## Local Testing

Before deployment, test locally:

```bash
docker build -t llm-quiz:test .
docker run -p 8000:8000 --env-file .env llm-quiz:test
```

Visit http://localhost:8000/docs to verify.

## Deployment Platforms

### Render.com (Recommended)

1. Create Render Account at https://render.com
2. Go to Dashboard → New → Web Service
3. Select your repository
4. Name: llm-analysis-quiz
5. Environment: Docker
6. Set environment variables
7. Click "Deploy"

### Heroku

1. Install Heroku CLI
2. heroku login
3. heroku create llm-analysis-quiz
4. heroku config:set GEMINI_API_KEY=your_key
5. git push heroku main

### AWS EC2

1. Launch Ubuntu 22.04 LTS instance
2. Install Python 3.11, Docker
3. Clone repository
4. docker-compose up -d
5. Configure NGINX reverse proxy

### DigitalOcean App Platform

1. Connect GitHub repository
2. Select docker-compose.yml
3. Add environment variables
4. Deploy

## Monitoring & Logging

### View Logs

Render:
```bash
render logs your-service-id
```

Heroku:
```bash
heroku logs --tail
```

### Health Checks

```bash
curl https://your-domain.com/health
```

## Security Best Practices

1. Use HTTPS only
2. Enable SSL/TLS certificate
3. Protect environment variables
4. Rotate API keys regularly
5. Monitor access logs
6. Keep dependencies updated

## Rollback Procedure

If deployment has issues:

Render: Go to Dashboard → Deploys → Select previous version → Redeploy
Heroku: heroku releases, heroku rollback v2
GitHub: Push new commit to previous version
