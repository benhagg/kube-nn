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
<script>
const colors=['#62d9ff','#ffbd59','#ad7cff','#69db7c','#ff7eb6','#f06595'];
function draw(id, workers, key) {
  const c=document.getElementById(id),x=c.getContext('2d'),w=c.width,h=c.height,left=50,bottom=38,right=8,top=28;
  x.clearRect(0,0,w,h); const series=Object.values(workers), points=series.flat();
  if(!points.length){x.fillStyle='#aab7c4';x.fillText('No worker samples yet.',left,h/2);return}
  const minT=Math.min(...points.map(a=>a.timestamp)), maxT=Math.max(...points.map(a=>a.timestamp)), maxV=Math.max(1,...points.map(a=>a[key]));
  const plotW=w-left-right,plotH=h-top-bottom,pxFor=t=>left+plotW*(maxT===minT?.5:(t-minT)/(maxT-minT)),pyFor=v=>top+plotH-(plotH*v/maxV);
  x.strokeStyle='#53606e';x.beginPath();x.moveTo(left,top);x.lineTo(left,h-bottom);x.lineTo(w-right,h-bottom);x.stroke();
  x.fillStyle='#aab7c4';x.textAlign='right';x.fillText(Math.ceil(maxV),left-6,top+5);x.fillText('0',left-6,h-bottom+5);
  const ticks=maxT===minT?[minT]:Array.from({length:5},(_,i)=>minT+(maxT-minT)*i/4);
  ticks.forEach(t=>{const px=pxFor(t);x.strokeStyle='#2d3946';x.beginPath();x.moveTo(px,top);x.lineTo(px,h-bottom);x.stroke();x.fillStyle='#aab7c4';x.textAlign='center';x.fillText(new Date(t*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit',second:'2-digit'}),px,h-12)});
  Object.entries(workers).forEach(([name,values],i)=>{x.strokeStyle=colors[i%colors.length];x.lineWidth=2;x.beginPath();values.forEach((a,j)=>{const px=pxFor(a.timestamp),py=pyFor(a[key]);j?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();x.fillStyle=colors[i%colors.length];x.textAlign='left';x.fillText(name,left+i*170,18)})
}
async function refresh(){try{const r=await fetch('/api/samples');const d=await r.json();document.getElementById('status').textContent=d.error?'Metrics API: '+d.error:`${Object.keys(d.workers).length} worker Pods; updated ${new Date(d.now*1000).toLocaleTimeString()}`;draw('cpu',d.workers,'cpuMillicores');draw('memory',d.workers,'memoryMiB')}catch(e){document.getElementById('status').textContent=e}}
refresh();setInterval(refresh,1000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/api/samples":
            with samples_lock:
                body = json.dumps({"now": time.time(), "error": last_error, "workers": {name: list(values) for name, values in samples.items()}}).encode()
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

    def log_message(self, *_args: object) -> None:
        pass


Thread(target=poll_metrics, daemon=True).start()
print(f"Monitoring worker metrics in namespace {NAMESPACE} every {INTERVAL_SECONDS:g}s", flush=True)
ThreadingHTTPServer(("", PORT), Handler).serve_forever()
