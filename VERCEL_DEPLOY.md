# 🚀 Vercel Deployment Instructions

## Quick Deploy to Vercel

### Method 1: Vercel CLI (Recommended)

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Deploy from your project directory**:
   ```bash
   cd /home/mounish/Documents/Hackathons/hackrx/hackrx_bajaj
   vercel --prod
   ```

3. **Follow the prompts**:
   - Link to existing project or create new one
   - Keep default settings
   - Deploy!

### Method 2: GitHub + Vercel Dashboard

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Deploy to Vercel"
   git push origin main
   ```

2. **Deploy via Vercel Dashboard**:
   - Go to [vercel.com](https://vercel.com)
   - Import your GitHub repository
   - Deploy with default settings

## Your Webhook URL

After deployment, your webhook URL will be:
```
https://your-project-name.vercel.app/hackrx/run
```

## Testing Your Deployment

1. **Health Check**:
   ```bash
   curl https://your-project-name.vercel.app/health
   ```

2. **System Info**:
   ```bash
   curl https://your-project-name.vercel.app/system/info
   ```

3. **Full Test**:
   ```bash
   curl -X POST "https://your-project-name.vercel.app/hackrx/run" \
     -H "Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72" \
     -H "Content-Type: application/json" \
     -d '{
       "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
       "questions": ["What is the grace period for premium payment?"]
     }'
   ```

## Important Notes

- **Serverless Limitations**: The deployment is optimized for Vercel's serverless environment
- **Cold Starts**: First request may take longer due to ML model loading
- **Size Limits**: Documents are limited to 10MB for serverless performance
- **Timeout**: Requests timeout after 30 seconds on Vercel

## Alternative: Use ngrok for Immediate Testing

If you need an immediate webhook URL for testing:

1. **Install ngrok**: Download from https://ngrok.com/download
2. **Start your local server**:
   ```bash
   cd /home/mounish/Documents/Hackathons/hackrx/hackrx_bajaj
   PYTHONPATH=/home/mounish/Documents/Hackathons/hackrx/hackrx_bajaj /home/mounish/Documents/Hackathons/hackrx/hackrx_bajaj/venv/bin/python -m uvicorn main_simple:app --host 0.0.0.0 --port 8001
   ```
3. **Expose with ngrok**:
   ```bash
   ngrok http 8001
   ```
4. **Use the ngrok URL**: `https://abc123.ngrok.io/hackrx/run`

## File Structure for Vercel

```
hackrx_bajaj/
├── api/
│   └── index.py          # Main serverless function
├── vercel.json           # Vercel configuration
├── requirements.txt      # Python dependencies
└── README.md
```

## Webhook URL Format

Your final webhook URL will be:
```
https://[your-project-name].vercel.app/hackrx/run
```

Replace `[your-project-name]` with your actual Vercel project name.
