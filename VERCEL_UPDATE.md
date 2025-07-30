# 🚀 Updated Vercel Deployment with Gemini API

## Step 1: Push to GitHub

```bash
# Add all changes
git add .

# Commit with a descriptive message
git commit -m "Add Gemini API key support and fix 403 errors"

# Push to GitHub
git push origin main
```

## Step 2: Update Vercel Environment Variables

1. **Go to Vercel Dashboard**: https://vercel.com/dashboard
2. **Select your project**: `hackrx-bajaj`
3. **Go to Settings** → **Environment Variables**
4. **Add the following environment variables**:

   ```
   GEMINI_API_KEY = AIzaSyBBi1RQgXXxh4CvbByxAdTB9yhZqxIqyBQ
   ```

   Optional (if you have OpenAI key):
   ```
   OPENAI_API_KEY = your-openai-api-key
   ```

## Step 3: Trigger Redeploy

### Method 1: Automatic (from GitHub push)
- Your deployment will automatically update when you push to main branch

### Method 2: Manual Redeploy
- In Vercel dashboard, go to your project
- Click "Deployments" tab
- Click "Redeploy" on the latest deployment

## Step 4: Test Your Updated Deployment

```bash
curl -X POST "https://hackrx-bajaj.vercel.app/hackrx/run" \
  -H "Authorization: Bearer ca6914a6c8df9d1ce075149c3ab9f060e666c75940576e37a98b3cf0e9092c72" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": ["What is the grace period for premium payment?"]
  }'
```

## Expected Result
- ✅ No more 403 Forbidden errors
- ✅ API responds with proper answers
- ✅ Gemini API fallback works correctly

## Your Webhook URL
```
https://hackrx-bajaj.vercel.app/hackrx/run
```

## Troubleshooting

If you still get 403 errors:
1. Check that environment variables are properly set in Vercel
2. Verify the API key is correct
3. Check the deployment logs in Vercel dashboard

## Next Steps
1. Run the commands above to push to GitHub
2. Set environment variables in Vercel
3. Test the deployment
4. Submit your webhook URL for the hackathon
