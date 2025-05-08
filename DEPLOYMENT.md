# Deployment Guide for Primavera Companion

This guide provides step-by-step instructions for deploying the Primavera Companion application to different hosting platforms.

## Table of Contents

1. [Deploying to Render](#deploying-to-render)
2. [Deploying to Heroku](#deploying-to-heroku)
3. [Deploying to Railway](#deploying-to-railway)
4. [Using Docker](#using-docker)
5. [Environment Variables](#environment-variables)

## Deploying to Render

[Render](https://render.com) is a unified cloud platform that offers free web services with SSL, a global CDN, and continuous deployment from GitHub.

### Steps:

1. Create a Render account and log in
2. Click "New" and select "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `primavera-companion` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Under "Advanced", add the following environment variables:
   - `SECRET_KEY`: A random string for Flask sessions (generate one with `python -c "import secrets; print(secrets.token_hex(16))"`)
6. Click "Create Web Service"
7. After deployment, upload your Primavera playlist to the `/data` directory through the Render dashboard or configure your application to download it during startup

**Render URL**: Your application will be available at `https://primavera-companion.onrender.com` (or your custom name)

## Deploying to Heroku

[Heroku](https://heroku.com) is a platform as a service (PaaS) that enables developers to build, run, and operate applications entirely in the cloud.

### Prerequisites:
- Heroku CLI installed
- Heroku account
- Git repository initialized

### Steps:

1. Create a `Procfile` in your project root:
   ```
   web: gunicorn app:app
   ```

2. Login to Heroku CLI:
   ```bash
   heroku login
   ```

3. Create a new Heroku app:
   ```bash
   heroku create primavera-companion
   ```

4. Set environment variables:
   ```bash
   heroku config:set SECRET_KEY=your_secret_key_here
   ```

5. Push your code to Heroku:
   ```bash
   git push heroku main
   ```

6. After deployment, upload your Primavera playlist using Heroku CLI or use a cloud storage solution to host your data

**Heroku URL**: Your application will be available at `https://primavera-companion.herokuapp.com`

## Deploying to Railway

[Railway](https://railway.app) is a modern platform for building, shipping, and monitoring applications.

### Steps:

1. Create a Railway account and log in
2. Create a new project and select "Deploy from GitHub repo"
3. Connect your GitHub repository
4. Add environment variables:
   - `SECRET_KEY`: Your secret key for Flask
5. Railway will automatically detect the requirements.txt file and build your application
6. Set the start command in the "Settings" tab:
   ```
   gunicorn app:app
   ```
7. Your application will be deployed automatically
8. Upload your Primavera playlist to a persistent storage location that your app can access

**Railway URL**: Your application will be available at the URL provided by Railway

## Using Docker

You can also deploy Primavera Companion using Docker for better portability and consistency across environments.

### Create a Dockerfile:

Create a file named `Dockerfile` in your project root:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/uploads /app/results /app/data

# Default environment variable
ENV SECRET_KEY=default_secret_key_change_me

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

### Build and run the Docker image:

```bash
# Build the image
docker build -t primavera-companion .

# Run the container
docker run -p 5000:5000 -e SECRET_KEY=your_secret_key_here -v /path/to/your/data:/app/data primavera-companion
```

Your application will be available at `http://localhost:5000`

## Environment Variables

Configure these environment variables for your deployment:

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `SECRET_KEY` | Flask secret key for session security | Yes | None |
| `PORT` | Port to run the application on | No | 5000 |
| `DEBUG` | Enable Flask debug mode | No | False |

## Preparing the Primavera Playlist

For any deployment option, you'll need to have the Primavera lineup data in CSV format with audio features. This file should be placed in the `data` directory with the name `primavera.csv`.

### Structure of `primavera.csv`:

The CSV should include the following columns:
- Artist Name(s)
- Genres
- Popularity
- Danceability
- Energy
- Key
- Loudness
- Mode
- Speechiness
- Acousticness
- Instrumentalness
- Liveness
- Valence
- Tempo
- Time Signature

You can extract this data from Spotify using tools like [Exportify](https://exportify.net) or the Spotify API.

## Troubleshooting

If you encounter issues during deployment:

1. Check application logs in your hosting platform dashboard
2. Verify that all environment variables are set correctly
3. Ensure the Primavera playlist CSV is in the correct location and format
4. Check that the uploads and results directories have proper write permissions
5. Verify that your hosting platform supports file uploads and temporary storage

For specific platform errors, consult the documentation for your chosen hosting service.