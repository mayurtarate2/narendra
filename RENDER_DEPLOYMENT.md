# 🚀 Deploy HackRX Document Q&A to Render

This guide will help you deploy your high-performance document Q&A system to Render.com.

## 📋 Prerequisites

1. **GitHub Repository**: Your code should be in a GitHub repository
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)

## 🔧 Deployment Steps

### 1. **Push to GitHub**
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. **Create Render Web Service**

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Configure the service:

   **Basic Settings:**
   - **Name**: `hackrx-document-qa`
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Runtime**: `Python 3`

   **Build & Deploy:**
   - **Build Command**: `./build.sh`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`

### 3. **Environment Variables**

Add these in the Render dashboard under **Environment**:

| Variable | Value | Notes |
|----------|--------|-------|
| `OPENAI_API_KEY` | `sk-your-key-here` | **Required** - Your OpenAI API key |
| `BEARER_TOKEN` | `your-secure-token` | **Optional** - For API authentication |
| `PYTHON_VERSION` | `3.12.0` | Python version |

### 4. **Health Check**
- **Health Check Path**: `/hackrx/health`

## 🎯 Post-Deployment

### Your API Endpoints:
- **Main API**: `https://your-app.onrender.com/hackrx/run`
- **Health Check**: `https://your-app.onrender.com/hackrx/health`
- **API Docs**: `https://your-app.onrender.com/docs`

### Test Your Deployment:
```bash
curl https://your-app.onrender.com/hackrx/health
```

## 📊 Performance Features Included:

✅ **FAISS Vector Search** - Ultra-fast similarity matching  
✅ **Async I/O** - Non-blocking operations  
✅ **Embedding Caching** - Reduces OpenAI API calls  
✅ **Multithreading** - CPU-intensive task optimization  
✅ **SQLite Optimization** - High-performance database  
✅ **Auto-scaling** - Render handles traffic spikes  

## 🔒 Security Notes:

- Your `.env` file is excluded from deployment
- Set environment variables in Render dashboard
- Use HTTPS endpoints (automatic on Render)
- Bearer token authentication is enabled

## 🐛 Troubleshooting:

### Build Fails:
- Check build logs in Render dashboard
- Ensure all dependencies are in `requirements.txt`

### Service Won't Start:
- Check start command: `uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1`
- Verify health check endpoint: `/hackrx/health`

### Performance Issues:
- Consider upgrading to higher Render plan
- Monitor logs for bottlenecks
- Check OpenAI API rate limits

## 💡 Tips:

1. **Free Tier**: Render free tier sleeps after 15 minutes of inactivity
2. **Upgrades**: For production, consider paid plans for better performance
3. **Monitoring**: Use Render's built-in metrics and logging
4. **Custom Domain**: Add your own domain in Render settings

Your high-performance document Q&A system is now ready for production! 🎉
