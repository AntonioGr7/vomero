# Deploying Vomero

Your case: **users select arbitrary data at runtime** (downloaded into a folder,
then used as the corpus). That data is untrusted, so the deployment is built to
keep your LLM credentials and network away from model-authored code.

## Chosen architecture — Strategy A: sandbox on a VM, frontend in Kubernetes

```
Kubernetes (your frontend)  ──HTTPS──▶  VM running Vomero with the gVisor sandbox
                                          • holds the API key, calls the LLM
                                          • per question: docker run --runtime=runsc
                                            (no network, no key, corpus read-only)
                                          • /data        = datasets (read-only)
                                          • workspace    = result files (writable)
```

Why: untrusted data means a prompt injection in a dataset could make model code
try to read your API key or call out. The per-step gVisor sandbox runs that code
with **no network and no secrets** (your key stays in the Vomero host process and
is reached only over an internal RPC socket), so the injection has nothing to
steal. That sandbox is launched with `docker run`, which needs a real container
runtime — easiest on a VM, awkward and privileged inside a pod.

### Follow these, in order

1. **VM setup → [`vm/README.md`](vm/README.md)** — install Docker + gVisor,
   install Vomero, mount datasets (read-only) and a writable workspace,
   configure ([`vm/vomero.env.example`](vm/vomero.env.example)), run as a service
   ([`vm/vomero.service`](vm/vomero.service)), and lock down the VM's firewall.
2. **Reach it from the cluster → [`k8s/vomero-endpoint.yaml`](k8s/vomero-endpoint.yaml)** —
   exposes the VM in-cluster as `http://vomero`; your frontend calls that.
3. **Constrain the frontend → [`k8s/networkpolicy.yaml`](k8s/networkpolicy.yaml)** —
   lets your frontend reach only the Vomero VM (+ DNS).

The HTTP API the VM serves (`POST /runs`, the SSE event stream, `POST .../reply`)
is implemented by [`app.py`](app.py) and documented in
[../docs/serving.md](../docs/serving.md). Each request carries a `dataset`
(a subfolder of the read-only data root) and a `question`.

## Input vs output, explicitly

- **Input (datasets)** are mounted **read-only**. The model reads them; in the
  gVisor sandbox the corpus is always read-only, so one user's run can't tamper
  with the source data.
- **Output (result files)** the model produces go to the **writable workspace**
  (`VOMERO_WORKSPACE_ROOT`), in a per-session subfolder. The text answer comes
  back over the API; files are collected from the workspace.

## Files here

```
deploy/
  app.py                       the FastAPI service the VM runs (corpus chosen per request)
  vm/README.md                 ← START HERE: the VM runbook (Strategy A)
  vm/vomero.env.example        VM configuration (API key, sandbox sizing, paths)
  vm/vomero.service            systemd unit
  k8s/vomero-endpoint.yaml     makes the VM reachable in-cluster as `vomero`
  k8s/networkpolicy.yaml       frontend → VM egress lockdown
  k8s/secret.example.yaml      (only for the fallback below)
  ── fallback, NOT for untrusted data ──
  Dockerfile                   image for the in-cluster Strategy B pod
  k8s/deployment.yaml          in-cluster, in-process pod (model code shares the pod's secrets)
  k8s/service.yaml             Service for that pod
  k8s/runtimeclass.yaml        gVisor RuntimeClass (if you sandbox the pod itself)
```

> The `Dockerfile` + `k8s/deployment.yaml` + `k8s/service.yaml` are a **fallback**
> for *trusted* data only (Strategy B: in-process inside a pod, where model code
> can read the pod's API key). They're kept for completeness; don't use them for
> the user-selected-data case. Background: [../docs/deployment.md](../docs/deployment.md) §4.
