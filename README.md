# Footee Vision

Footee Vision is an experimental soccer computer-vision web app for turning a
player highlight reel into reviewable clips and a first-stage movement profile.
Users can upload a reel, split it into scenes, remove likely cutaways, select and
track a player, and receive an archetype with similar-player film suggestions.
Three preprocessed sample reels are also included for visitors who do not have a
video ready; opening a sample uses static Vercel assets and no Azure inference.

- [Open the live app](https://footee-highlights.vercel.app/)
- [Check the live API](https://footee-vision-api.mangoplant-7b9bb2f4.eastus.azurecontainerapps.io/health)
- [Read the detailed project overview](project_overview.md)

## Current workflow

1. Open one of three instant sample reels for read-only clip review, or upload a
   soccer highlight video for the full workflow.
2. For a new upload, start scene detection and follow live progress for TransNetV2 cuts, cutaway
   filtering, and thumbnail generation.
3. Review the gameplay clips, detect players in a frame, and select the player
   to follow.
4. Track that player through one or more clips, adding a manual anchor if the
   tracker loses the identity.
5. Enter the player's positions and footedness to generate a movement-based
   archetype, supporting evidence, confidence, and deduplicated film examples.

## Technology

- **Frontend:** React 19, TypeScript, Vite, and Tailwind CSS on Vercel
- **Backend:** FastAPI, Python 3.12, OpenCV, PyTorch, TransNetV2, and
  Ultralytics YOLO11 on Azure Container Apps
- **Container delivery:** Docker and Azure Container Registry
- **Storage:** user uploads and derived artifacts are temporary; the three
  intentionally public demo reels and their precomputed results are static
  frontend assets

## Project structure

```text
footee-highlights/
  frontend/              React application, API client, and static demos
    public/samples/      Three preprocessed read-only sample reels
  backend/               FastAPI routes and computer-vision pipeline
  Dockerfile             Production backend image
  deploy-azure.ps1       Azure build and deployment script
  project_overview.md    Detailed product and engineering documentation
```

## Run locally

Start the backend from the repository root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The API runs at `http://localhost:8000`, its
interactive documentation is at `http://localhost:8000/docs`, and the frontend
uses the local API by default. To use another backend, create
`frontend/.env.local`:

```dotenv
VITE_API_BASE_URL=https://your-backend.example.com
```

## Production deployment

The frontend is deployed from GitHub to Vercel. The CPU-heavy backend runs as a
Docker container on Azure Container Apps with 2 vCPU, 4 GiB of memory, one
worker, and scale-to-zero enabled.

After signing in to Azure and starting Docker Desktop, deploy or update the
backend with:

```powershell
.\deploy-azure.ps1
```

Set the printed backend URL as `VITE_API_BASE_URL` in Vercel, without a trailing
slash, and redeploy the frontend.

## Privacy and current limitations

Video files, thumbnails, detection caches, clips, and result JSON are temporary.
The client requests cleanup when a session ends, while the backend expires
abandoned data after one hour by default. Azure uses ephemeral `/tmp` storage;
the project does not provide permanent user video storage.

The three bundled sample reels are the explicit exception: they are public,
persistent demo assets rather than user uploads. Only publish sample footage
that you have permission to distribute. Static demos avoid backend compute, but
serving video still uses normal frontend hosting bandwidth.

Footee Vision is an early-stage development aid, not a definitive scouting or
performance assessment. Scene boundaries can be imperfect, player tracking can
lose or switch identities, and the current profile is based primarily on
frame-by-frame movement rather than complete tactical or on-ball context.

See [project_overview.md](project_overview.md) for the architecture, API,
pipeline design, deployment configuration, privacy lifecycle, and roadmap.
