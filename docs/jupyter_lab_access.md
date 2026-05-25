# JupyterLab Access — QStrata GPU Container

Local JupyterLab access for interactive notebook work inside the `qstrata-gpu` Docker environment.

---

## 1. Prerequisites

- `qstrata-gpu` container image built (`docker compose build` run at least once)
- Port **8889** free on the host (host 8889 → container 8888)
- External drive containing the VinDr-SpineXR dataset mounted at `/media/mike/Datasets/` on the host
- Docker Compose file: `infra/docker/docker-compose.gpu.yml`

**Port note:** The host-side port is **8889**, not 8888. This is intentional — it avoids conflict with any other local Jupyter server on 8888. The container listens on 8888 internally.

**Auth note:** No token is required. `--IdentityProvider.token=''` is set in the compose command, so the browser opens directly without a login prompt.

---

## 2. Starting the Container

JupyterLab starts automatically as the container's default command. Run from the repo root:

```bash
# Option A — use the helper script
bash scripts/start_jupyter_lab.sh

# Option B — use compose directly
docker compose -f infra/docker/docker-compose.gpu.yml up -d qstrata-gpu
```

Check the container is running:

```bash
docker compose -f infra/docker/docker-compose.gpu.yml ps
```

---

## 3. Launching JupyterLab

JupyterLab starts automatically when the container starts — no separate launch step is needed.

If you need to run JupyterLab in an additional session (e.g., for a second instance), you can exec into the running container:

```bash
docker compose -f infra/docker/docker-compose.gpu.yml exec qstrata-gpu \
  jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --notebook-dir=/workspace
```

---

## 4. Accessing in the Browser

Open:

```
http://localhost:8889
```

No token or password is required — the browser opens directly to the JupyterLab file browser.

The notebook directory root is `/workspace`, which maps to the repo root on the host:
`/home/mike/research-projects/QStrata`

Notebooks are under `/workspace/notebooks/`.

---

## 5. Verifying Dataset Access Inside a Notebook

Run in any notebook cell:

```python
import os
print(os.listdir('/datasets/vindr-spinexr'))
```

Expected output:
```
['LICENSE.txt', 'SHA256SUMS.txt', '_speed_test', 'annotations',
 'physionet.org', 'supplemental_file_DICOM_tags_SpineXR.pdf',
 'test_images', 'train_images', 'urls_aria2.txt']
```

The dataset is mounted **read-only** (`/media/mike/Datasets/vindr-spinexr:/datasets/vindr-spinexr:ro`). Write operations inside the container will fail — this is intentional.

---

## 6. Stopping JupyterLab

**From the browser:** File → Shut Down from the JupyterLab menu, or close the tab.

**From the terminal:** Stop the container (this also stops JupyterLab):

```bash
docker compose -f infra/docker/docker-compose.gpu.yml down
```

To stop and remove the container without touching volumes:

```bash
docker compose -f infra/docker/docker-compose.gpu.yml stop qstrata-gpu
```

---

## 7. Troubleshooting

**Port 8889 not responding:**
```bash
# Check container status
docker compose -f infra/docker/docker-compose.gpu.yml ps

# Check JupyterLab logs
docker compose -f infra/docker/docker-compose.gpu.yml logs qstrata-gpu
```

**Dataset not accessible (`/datasets/vindr-spinexr` not found):**
- Verify the external drive is mounted on the host: `ls /media/mike/Datasets/vindr-spinexr`
- If unmounted, mount it and recreate the container: `docker compose -f infra/docker/docker-compose.gpu.yml up -d`

**DICOM files not readable:**
- The container has `pydicom`, `pylibjpeg`, and `pylibjpeg-openjpeg` installed.
- VinDr-SpineXR images use JPEG 2000 compression — these packages are required.
- Verify inside the container: `docker compose -f infra/docker/docker-compose.gpu.yml exec qstrata-gpu pip show pydicom`

**Rebuild the container image (after Dockerfile changes):**
```bash
docker compose -f infra/docker/docker-compose.gpu.yml build qstrata-gpu
docker compose -f infra/docker/docker-compose.gpu.yml up -d qstrata-gpu
```
