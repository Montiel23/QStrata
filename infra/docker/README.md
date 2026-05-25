## QStrata EDA Docker Environment — v0.1

### Build
```
docker compose --env-file .env -f infra/docker/docker-compose.yml build
```

### Start
```
docker compose --env-file .env -f infra/docker/docker-compose.yml up
```

### Open Jupyter
Open browser: http://localhost:8888
No token required (local research mode).

⚠️  Security note: Jupyter token is disabled for local research convenience.
    Use localhost only. Do not expose port 8888 publicly.

### Dataset mount
Copy `.env.example` to `.env` and set your dataset path:
```
cp .env.example .env
# Edit .env: DATASET_PATH=/your/path/to/vindr-spinexr
```
Inside the notebook use: `/data/vindr-spinexr/...`

⚠️  Always use `--env-file .env` explicitly.
    Docker Compose may not automatically resolve `.env` from the repo root
    when using `-f infra/docker/docker-compose.yml`. Without `--env-file .env`,
    `DATASET_PATH` will not be resolved and the mount falls back to `../../data`.

### Config check
```
docker compose --env-file .env -f infra/docker/docker-compose.yml config
```

### Stop
```
docker compose --env-file .env -f infra/docker/docker-compose.yml down
```

### Notes
- CPU-only environment. No GPU required.
- QML libraries not included. This is EDA only.
- Torch not included. Will be added in Slice 3 for baseline modeling.
- Repo root is mounted at `/workspace`. Edits are live.
