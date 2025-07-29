# 🚀 Deployment Guide for Webhook Submission

## Quick Deployment Options

### Option 1: Render (Recommended for this hackathon)
1. Push your code to GitHub
2. Go to [render.com](https://render.com)
3. Connect your GitHub repository
4. Create a new "Web Service"
5. Use these settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python -m uvicorn main_simple:app --host 0.0.0.0 --port $PORT`
   - **Environment**: `PYTHONPATH=/opt/render/project/src`

Your webhook URL will be: `https://your-app-name.onrender.com/hackrx/run`

### Option 2: Railway
1. Install Railway CLI: `npm install -g @railway/cli`
2. Login: `railway login`
3. Deploy: `railway up`
4. Your webhook URL: `https://your-app.railway.app/hackrx/run`

### Option 3: Vercel (Serverless)
1. Install Vercel CLI: `npm install -g vercel`
2. Deploy: `vercel --prod`
3. Your webhook URL: `https://your-project.vercel.app/hackrx/run`

### Option 4: Heroku
1. Install Heroku CLI
2. Create app: `heroku create your-app-name`
3. Deploy: `git push heroku main`
4. Your webhook URL: `https://your-app-name.herokuapp.com/hackrx/run`

### Option 5: ngrok (For testing)
1. Install ngrok: `https://ngrok.com/download`
2. Start your local server: `python -m uvicorn main_simple:app --host 0.0.0.0 --port 8001`
3. Expose with ngrok: `ngrok http 8001`
4. Use the ngrok URL: `https://abc123.ngrok.io/hackrx/run`

## Current Local Testing
Your server is running locally at: `http://localhost:8001/hackrx/run`

## Required Headers for Webhook
```
Content-Type: application/json
Accept: application/json
Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72
```

## Test Command
```bash
curl -X POST "https://your-deployed-url.com/hackrx/run" \
  -H "Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": ["What is the grace period for premium payment?"]
  }'
```

## Quick ngrok Setup (Fastest for immediate submission)

1. Download ngrok from https://ngrok.com/download
2. Run your local server
3. In a new terminal: `ngrok http 8001`
4. Copy the https URL (e.g., https://abc123.ngrok.io)
5. Your webhook URL: `https://abc123.ngrok.io/hackrx/run`

This will make your local server publicly accessible immediately!
