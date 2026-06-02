# Running Vomero on a VM with the gVisor sandbox (Strategy A)

This is the recommended setup when **users select arbitrary data at runtime**:
the data is untrusted, so every question's model-authored code runs in a
throwaway gVisor container with **no network and no access to your API key**.
Even a prompt injection hidden in someone's dataset can't steal credentials or
call out.

```
┌─────────────── Kubernetes ───────────────┐        ┌──────────── VM (this guide) ───────────┐
│  your frontend / API pods                 │  HTTPS │  Vomero service (uvicorn, app.py)       │
│   └── call http://vomero/runs  ───────────┼───────▶│   holds the API key; talks to the LLM   │
└───────────────────────────────────────────┘        │   └── per question: docker run --runsc  │
                                                       │        (no network, no key, ro corpus)  │
                                                       │  /data       datasets (read-only)        │
                                                       │  /var/lib/vomero/workspace  results (rw) │
                                                       └──────────────────────────────────────────┘
```

Why a VM and not a pod: Strategy A launches a gVisor container **per step** via
`docker run`, which needs a real container runtime. A VM has one; a normal pod
doesn't (you'd need a privileged Docker-in-Docker sidecar). Running Vomero
directly on the VM host — *not* itself in a container — lets it drive the host's
Docker cleanly.

You need: a Linux VM you control, on a private network your cluster can reach.
Treat it as dedicated to Vomero.

---

## 1. Install Docker + gVisor

```bash
# Docker (Debian/Ubuntu shown; use your distro's instructions):
curl -fsSL https://get.docker.com | sudo sh

# gVisor (runsc) — see https://gvisor.dev/docs/user_guide/install/ for the
# apt/yum repo lines, then register it with Docker and restart:
sudo runsc install
sudo systemctl restart docker

# Verify gVisor works:
docker run --rm --runtime=runsc hello-world
# Pre-pull the sandbox image so the first request isn't slow:
docker pull python:3.11-slim
```

## 2. Install Vomero on the VM (on the host, not in a container)

```bash
sudo useradd --system --create-home --home-dir /opt/vomero vomero
sudo usermod -aG docker vomero            # so it can run `docker run`

sudo -u vomero -H bash -lc '
  cd /opt/vomero
  git clone <YOUR_VOMERO_REPO_URL> repo && cd repo
  python3 -m venv /opt/vomero/.venv
  /opt/vomero/.venv/bin/pip install . "fastapi>=0.110" "uvicorn[standard]>=0.29"
'
# The systemd unit expects WorkingDirectory=/opt/vomero with deploy/app.py
# reachable; symlink the repo there for simplicity:
sudo ln -s /opt/vomero/repo/deploy /opt/vomero/deploy
```

## 3. Mount the datasets (read-only) and create the workspace (writable)

- **`/data`** — your dataset store, mounted **read-only**. How you mount it is
  your platform's choice (NFS, an object-storage FUSE mount, a cloud disk).
  Example for NFS:

  ```bash
  sudo mkdir -p /data
  sudo mount -o ro,nfsvers=4 nfs-server:/datasets /data
  # make it permanent in /etc/fstab
  ```

  Each request's `dataset` field is a **subfolder** of `/data` (e.g. a request
  with `"dataset": "ds-123"` reads `/data/ds-123`).

- **`/var/lib/vomero/workspace`** — where the model writes result files. Must be
  writable by the `vomero` user. If you want your frontend to retrieve results,
  put this on a share the frontend can also read (or add a download endpoint).

  ```bash
  sudo install -d -o vomero -g vomero /var/lib/vomero/workspace
  ```

## 4. Configure and start the service

```bash
sudo install -d /etc/vomero
sudo cp /opt/vomero/repo/deploy/vm/vomero.env.example /etc/vomero/vomero.env
sudo nano /etc/vomero/vomero.env          # set OPENAI_API_KEY, model, sizing
sudo chmod 600 /etc/vomero/vomero.env     # the key lives here

sudo cp /opt/vomero/repo/deploy/vm/vomero.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vomero
systemctl status vomero
```

Smoke-test locally on the VM:

```bash
curl -s localhost:8000/healthz
curl -s -XPOST localhost:8000/runs -H 'content-type: application/json' \
  -d '{"dataset":"ds-123","question":"What is in this dataset?"}'
# then stream GET /runs/<run_id>/events  (Server-Sent Events)
```

Confirm the model code is really sandboxed — ask a question whose code prints
the kernel, or check the logs: each run does `docker run --runtime=runsc`. A run
that tries to reach the network from inside its code gets connection errors
(the sandbox is `--network none`).

## 5. Lock down the VM's network

The model's sandbox already has **no network**. The only thing that needs egress
is the Vomero host process talking to your LLM provider. So:

- **Egress:** allow outbound `443` to your LLM provider only; deny the rest.
- **Ingress:** allow `8000` **only** from your cluster's egress IP/range
  (a security group / firewall rule). Don't expose 8000 to the internet.

```bash
# Example with ufw (adjust to your provider/cluster CIDRs):
sudo ufw default deny incoming
sudo ufw default deny outgoing
sudo ufw allow out 53                       # DNS
sudo ufw allow out 443                       # HTTPS to the LLM provider (tighten to its range)
sudo ufw allow from <CLUSTER_EGRESS_CIDR> to any port 8000 proto tcp
sudo ufw enable
```

## 6. Point Kubernetes at the VM

Apply [`../k8s/vomero-endpoint.yaml`](../k8s/vomero-endpoint.yaml) (set the VM's
private IP in it). Your frontend pods then call `http://vomero/runs` in-cluster,
and Kubernetes routes that to the VM. See the comments in that file for the
DNS/ExternalName variant.

---

## Operating notes

- **Capacity:** each in-flight question runs its own container, and recursion
  (`rlm()`) can add up to `VOMERO_MAX_DEPTH` more. Size the VM's CPU/RAM for
  `peak concurrent runs × VOMERO_SANDBOX_MEMORY`, with headroom. Lower
  `VOMERO_MAX_STEPS` / `VOMERO_MAX_DEPTH` to cap cost and load.
- **Latency:** gVisor cold-start is paid once per run (then reused across that
  run's steps). The first request after boot also pulls/initialises the image.
- **Results location:** files land in
  `${VOMERO_WORKSPACE_ROOT}/<user_id>/<session_id>/`. They persist until the
  workspace is cleaned; the session's *variables* are dropped after
  `VOMERO_SESSION_TTL` idle seconds, but the files remain.
- **Scaling out:** run several VMs behind the Service's endpoint list, but note
  conversation history + per-session variables live in one VM's memory — route a
  given `session_id` to the same VM (or move history to a shared store).
- **Updates:** `git pull` in the repo, `pip install .` in the venv, then
  `sudo systemctl restart vomero` (in-flight runs drain within `TimeoutStopSec`).
