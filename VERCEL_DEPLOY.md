# Vercel Deployment Guide

## Files Created for Vercel

1. **`vercel.json`** - Vercel configuration for routing
2. **`api/index.py`** - Serverless function handler
3. **`.vercelignore`** - Files to exclude from deployment

## Deployment Steps

### 1. Install Vercel CLI (if not already installed)
```bash
npm install -g vercel
```

### 2. Login to Vercel
```bash
vercel login
```

### 3. Deploy
```bash
vercel
```

For production deployment:
```bash
vercel --prod
```

### 4. Or Deploy via GitHub
- Push your code to GitHub
- Connect your repository to Vercel
- Vercel will automatically deploy on push

## Important Notes

1. **Excel Files**: Make sure `projects_data.xlsx` is in your repository root. Vercel will include it in the deployment.

2. **Environment Variables**: If you need any environment variables, add them in Vercel dashboard:
   - Go to your project settings
   - Add environment variables under "Environment Variables"

3. **File Size Limits**: 
   - Vercel has limits on function size (50MB uncompressed)
   - If pandas/numpy cause issues, you might need to use lighter alternatives

4. **Cold Starts**: Serverless functions have cold starts. First request might be slower.

## Troubleshooting

### If deployment fails:
1. Check that all dependencies are in `requirements.txt`
2. Ensure `mangum` is included (required for FastAPI on Vercel)
3. Check Vercel build logs for errors

### If API doesn't work:
1. Check that routes in `vercel.json` match your API paths
2. Verify `api/index.py` is correctly importing the app
3. Check browser console for CORS errors

## Testing Locally with Vercel

```bash
vercel dev
```

This will run your app locally with Vercel's routing.

