## QStrata EDA Docker Environment — v0.1

### Build
```
docker compose -f infra/docker/docker-compose.yml build
```

### Start
```
docker compose -f infra/docker/docker-compose.yml up
```

### Open Jupyter
Open browser: http://localhost:8888
No token required (local research mode).

⚠️  Security note: Jupyter token is disabled for local research convenience.
    Use localhost only. Do not expose port 8888 publicly.

### Dataset mount
By default mounts `../../data` to `/data` inside the container.
To use a custom dataset path:
```
DATASET_PATH=/your/path/to/vindr-spinexr \
  docker compose -f infra/docker/docker-compose.yml up
```
Inside the notebook use: `/data/vindr-spinexr/...`

### Stop
```
docker compose -f infra/docker/docker-compose.yml down
```

### Notes
- CPU-only environment. No GPU required.
- QML libraries not included. This is EDA only.
- Torch not included. Will be added in Slice 3 for baseline modeling.
- Repo root is mounted at `/workspace`. Edits are live.
