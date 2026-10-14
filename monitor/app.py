"""One-second Kubernetes worker resource monitor with a small browser dashboard."""

from __future__ import annotations

from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import ssl
from threading import Lock, Thread
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


PORT = int(os.environ.get("PORT", "8080"))
INTERVAL_SECONDS = float(os.environ.get("POLL_INTERVAL_SECONDS", "1"))
RETENTION_SAMPLES = int(os.environ.get("RETENTION_SAMPLES", "3600"))
NAMESPACE_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/namespace")
TOKEN_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
NAMESPACE = os.environ.get("POD_NAMESPACE") or (NAMESPACE_PATH.read_text().strip() if NAMESPACE_PATH.exists() else "default")
METRICS_URL = (
    "https://kubernetes.default.svc/apis/metrics.k8s.io/v1beta1/"
    f"namespaces/{quote(NAMESPACE)}/pods?labelSelector={quote('app=nanogpt-worker')}"
)

samples: dict[str, deque[dict[str, float]]] = defaultdict(lambda: deque(maxlen=RETENTION_SAMPLES))
training_samples: deque[dict[str, float]] = deque(maxlen=RETENTION_SAMPLES)
samples_lock = Lock()
last_error: str | None = None


def quantity_to_cpu_millicores(value: str) -> float:
    if value.endswith("n"):
        return float(value[:-1]) / 1_000_000
    if value.endswith("u"):
        return float(value[:-1]) / 1_000
    if value.endswith("m"):
        return float(value[:-1])
    return float(value) * 1000


def quantity_to_mebibytes(value: str) -> float:
    units = {"Ki": 1024, "Mi": 1024**2, "Gi": 1024**3, "K": 1000, "M": 1000**2, "G": 1000**3}
    for suffix, multiplier in units.items():
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * multiplier / (1024**2)
    return float(value) / (1024**2)


def poll_metrics() -> None:
    global last_error
    while True:
        started = time.time()
        try:
            token = TOKEN_PATH.read_text().strip()
            request = Request(METRICS_URL, headers={"Authorization": f"Bearer {token}"})
            context = ssl.create_default_context(cafile=CA_PATH)
            with urlopen(request, context=context, timeout=5) as response:
                payload = json.load(response)

            now = time.time()
            with samples_lock:
                for pod in payload.get("items", []):
                    trainer = next((item for item in pod.get("containers", []) if item["name"] == "trainer"), None)
                    if trainer is None:
                        continue
                    usage = trainer["usage"]
                    samples[pod["metadata"]["name"]].append(
                        {
                            "timestamp": now,
                            "cpuMillicores": quantity_to_cpu_millicores(usage["cpu"]),
                            "memoryMiB": quantity_to_mebibytes(usage["memory"]),
                        }
                    )
            last_error = None
        except (OSError, HTTPError, URLError, ValueError, KeyError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(max(0, INTERVAL_SECONDS - (time.time() - started)))


DASHBOARD = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>nanoGPT worker resources</title>
<style>body{background:#10151c;color:#edf2f7;font:16px system-ui;margin:2rem;max-width:1100px}h1{margin-bottom:.2rem}.muted{color:#aab7c4}section{background:#1a222d;border-radius:8px;margin:1.5rem 0;padding:1rem}canvas{width:100%;height:260px;background:#121922}#status{color:#aab7c4}</style></head>
<body><h1>nanoGPT worker resources</h1><p class="muted">Samples every second from the Kubernetes Metrics API. Retained in this monitor Pod's memory.</p><p id="status">Loading…</p>
<section><h2>CPU by worker (millicores)</h2><canvas id="cpu" width="1000" height="260"></canvas></section>
<section><h2>Memory by worker (MiB)</h2><canvas id="memory" width="1000" height="260"></canvas></section>
<section><h2>Training loss by step</h2><canvas id="loss" width="1000" height="260"></canvas></section>
<section><h2>Step duration by step (ms)</h2><canvas id="duration" width="1000" height="260"></canvas></section>
<section><h2>Training throughput by step (tokens/second)</h2><canvas id="throughput" width="1000" height="260"></canvas></section>
<script>
const colors=['#62d9ff','#ffbd59','#ad7cff','#69db7c','#ff7eb6','#f06595'];
const formatValue=v=>v>=100?Math.round(v):v>=10?v.toFixed(1):v.toFixed(2);
function draw(id, workers, key) {
  const c=document.getElementById(id),x=c.getContext('2d'),w=c.width,h=c.height,left=50,bottom=38,right=8,top=28;
  x.clearRect(0,0,w,h); const series=Object.values(workers), points=series.flat();
  if(!points.length){x.fillStyle='#aab7c4';x.fillText('No worker samples yet.',left,h/2);return}
  const minT=Math.min(...points.map(a=>a.timestamp)), maxT=Math.max(...points.map(a=>a.timestamp)), maxV=Math.max(1,...points.map(a=>a[key]));
  const plotW=w-left-right,plotH=h-top-bottom,pxFor=t=>left+plotW*(maxT===minT?.5:(t-minT)/(maxT-minT)),pyFor=v=>top+plotH-(plotH*v/maxV);
  x.strokeStyle='#53606e';x.beginPath();x.moveTo(left,top);x.lineTo(left,h-bottom);x.lineTo(w-right,h-bottom);x.stroke();
  Array.from({length:5},(_,i)=>maxV*i/4).forEach(v=>{const py=pyFor(v);x.strokeStyle='#2d3946';x.beginPath();x.moveTo(left,py);x.lineTo(w-right,py);x.stroke();x.fillStyle='#aab7c4';x.textAlign='right';x.fillText(formatValue(v),left-6,py+4)});
  const ticks=maxT===minT?[minT]:Array.from({length:5},(_,i)=>minT+(maxT-minT)*i/4);
  ticks.forEach(t=>{const px=pxFor(t);x.strokeStyle='#2d3946';x.beginPath();x.moveTo(px,top);x.lineTo(px,h-bottom);x.stroke();x.fillStyle='#aab7c4';x.textAlign='center';x.fillText(new Date(t*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}),px,h-12)});
  Object.entries(workers).forEach(([name,values],i)=>{x.strokeStyle=colors[i%colors.length];x.lineWidth=2;x.beginPath();values.forEach((a,j)=>{const px=pxFor(a.timestamp),py=pyFor(a[key]);j?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.fillStyle=colors[i%colors.length];x.fillRect(left+i*170,10,10,10);x.fillStyle='#edf2f7';x.textAlign='left';x.fillText(name,left+14+i*170,19)})
}
function drawStep(id, values, key) {
  const c=document.getElementById(id),x=c.getContext('2d'),w=c.width,h=c.height,left=50,bottom=38,right=8,top=12;
  x.clearRect(0,0,w,h);if(!values.length){x.fillStyle='#aab7c4';x.fillText('No training samples yet.',left,h/2);return}
  const minS=Math.min(...values.map(a=>a.step)),maxS=Math.max(...values.map(a=>a.step)),maxV=Math.max(1,...values.map(a=>a[key]));
  const plotW=w-left-right,plotH=h-top-bottom,pxFor=s=>left+plotW*(maxS===minS?.5:(s-minS)/(maxS-minS)),pyFor=v=>top+plotH-(plotH*v/maxV);
  x.strokeStyle='#53606e';x.beginPath();x.moveTo(left,top);x.lineTo(left,h-bottom);x.lineTo(w-right,h-bottom);x.stroke();Array.from({length:5},(_,i)=>maxV*i/4).forEach(v=>{const py=pyFor(v);x.strokeStyle='#2d3946';x.beginPath();x.moveTo(left,py);x.lineTo(w-right,py);x.stroke();x.fillStyle='#aab7c4';x.textAlign='right';x.fillText(formatValue(v),left-6,py+4)});
  const ticks=maxS===minS?[minS]:Array.from({length:5},(_,i)=>minS+(maxS-minS)*i/4);
  ticks.forEach(s=>{const px=pxFor(s);x.strokeStyle='#2d3946';x.beginPath();x.moveTo(px,top);x.lineTo(px,h-bottom);x.stroke();x.fillStyle='#aab7c4';x.textAlign='center';x.fillText(`step ${Math.round(s)}`,px,h-12)});
  x.strokeStyle='#62d9ff';x.lineWidth=2;x.beginPath();values.forEach((a,i)=>{const px=pxFor(a.step),py=pyFor(a[key]);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();
}
async function refresh(){try{const r=await fetch('/api/samples');const d=await r.json();document.getElementById('status').textContent=d.error?'Metrics API: '+d.error:`${Object.keys(d.workers).length} worker Pods; updated ${new Date(d.now*1000).toLocaleTimeString()}`;draw('cpu',d.workers,'cpuMillicores');draw('memory',d.workers,'memoryMiB');drawStep('loss',d.training,'loss');drawStep('duration',d.training,'durationMilliseconds');drawStep('throughput',d.training,'throughputTokensPerSecond')}catch(e){document.getElementById('status').textContent=e}}
refresh();setInterval(refresh,1000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/api/samples":
            with samples_lock:
                body = json.dumps({"now": time.time(), "error": last_error, "workers": {name: list(values) for name, values in samples.items()}, "training": list(training_samples)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        elif self.path == "/":
            body = DASHBOARD.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        else:
            self.send_error(404)
            return
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/api/training":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            sample = {
                "step": int(payload["step"]),
                "loss": float(payload["loss"]),
                "durationMilliseconds": float(payload["durationSeconds"]) * 1000,
                "throughputTokensPerSecond": float(payload["throughputTokensPerSecond"]),
            }
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            self.send_error(400, "Expected numeric training metrics JSON")
            return
        with samples_lock:
            training_samples.append(sample)
        self.send_response(204)
        self.end_headers()

    def log_message(self, *_args: object) -> None:
        pass


Thread(target=poll_metrics, daemon=True).start()
print(f"Monitoring worker metrics in namespace {NAMESPACE} every {INTERVAL_SECONDS:g}s", flush=True)
ThreadingHTTPServer(("", PORT), Handler).serve_forever()
