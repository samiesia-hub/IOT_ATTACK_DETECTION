# IoT model replay demo on Ubuntu

Copy this entire folder to your Ubuntu server. It includes the trained model bundles and the held-out test data. The demo replays labelled dataset records; it does not monitor live network traffic. Only use these trusted model files, because loading joblib files can execute code.

The trained models and dataset are excluded from Git. A fresh Git clone does not contain them; use the prepared local folder from the training computer, or add those files yourself before building.

The server needs Docker Engine and the Docker Compose plugin, plus internet access during the first build to download the Python image and dependencies. It must have enough free disk space for the image. The included models were trained with Python 3.13 and the pinned packages in `demo-requirements.txt`; on architectures without compatible package wheels, the build may fail. A container memory limit of 1 GB and one CPU are configured; actual resource usage and timings depend on the host.

In a terminal on the server, change to this folder and run:

```bash
docker compose up -d --build
docker compose ps
docker compose logs --tail 50
```

The web app listens on all server network interfaces on port 8501. On the server, run `hostname -I` to find its LAN address, then open `http://SERVER_IP:8501` on a device connected to that network. The app has no login; anyone who can reach that port can use it. Do not forward port 8501 through your router or expose it publicly.

Alternatively, from your Windows computer, open PowerShell and replace the placeholders with your server username and IP address:

```powershell
ssh -L 8501:127.0.0.1:8501 USER@SERVER_IP
```

Keep the SSH window open and visit **http://localhost:8501** in a browser on your Windows computer. To stop the app without deleting its image, run `docker compose down` from this folder on the server. To restart it later, run `docker compose up -d`.

If port 8501 is already used on the server, change the left port in `compose.yaml`, for example `8502:8501`, and visit `http://SERVER_IP:8502`. Inference timing shown in the page is illustrative; it excludes feature extraction and is not a controlled edge benchmark.
