# Deploying to Render (Free)

## Quick Steps

1. **Push your code to GitHub**
   - Create a new repository on GitHub
   - Push the `clean_temp` folder contents to it

2. **Deploy on Render**
   - Go to [render.com](https://render.com) and sign up (free)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository and branch
   - Configure:
     - **Name**: disease-simulator (or any name)
     - **Environment**: Python 3
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app`
   - Click "Create Web Service"

3. **Wait for deployment** (takes ~5 minutes first time)

4. **Your app will be live at**: `https://your-app-name.onrender.com`

## Alternative: Railway

1. Go to [railway.app](https://railway.app) and sign up
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects Flask - it should work automatically!
5. Your app will be live at: `https://your-app-name.up.railway.app`

## Notes

- Render free tier: spins down after 15 min of inactivity (first request may be slow)
- Railway free tier: similar behavior
- Both are perfect for demos and testing!

