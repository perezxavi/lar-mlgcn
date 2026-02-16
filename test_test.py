# bench_inference.py
import argparse
import time
from statistics import mean, stdev
import numpy as np
import torch

# --------------------------
# Utils
# --------------------------
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def p95(xs):
    xs = sorted(xs)
    k = int(0.95 * (len(xs) - 1))
    return xs[k]

def human_mb(bytes_):
    return f"{bytes_ / 1e6:.1f} MB"

def set_tf32_if_available():
    if torch.cuda.is_available():
        # En GPUs Ampere+ esto acelera FP32 matmul/conv sin (normalmente) degradar métricas
        torch.set_float32_matmul_precision("high")

def maybe_compile(model, backend: str):
    """
    backend: 'none' | 'eager' | 'inductor'
    """
    if not hasattr(torch, "compile"):
        print("torch.compile no disponible (PyTorch < 2.0). Continuando en eager.")
        return model

    if backend == "none":
        return model
    if backend == "eager":
        print("Usando torch.compile backend='eager' (no requiere Triton).")
        return torch.compile(model, backend="eager")
    if backend == "inductor":
        print("Usando torch.compile backend='inductor' (requiere Triton compatible).")
        try:
            return torch.compile(model, backend="inductor")
        except Exception as e:
            print("Fallo al usar inductor; continuando en eager. Motivo:", e)
            return model
    raise ValueError("backend debe ser: none | eager | inductor")

# --------------------------
# Model (ejemplo)
# Sustituye por tu modelo real si quieres
# --------------------------
class Dummy(torch.nn.Module):
    def __init__(self, d, C):
        super().__init__()
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, C),
        )
    def forward(self, x):  # [N, d] -> [N, C]
        return self.ff(x)

# --------------------------
# Implementaciones a comparar
# --------------------------
@torch.inference_mode()
def predict_proba_original(model, Xnp: np.ndarray, device: torch.device):
    """
    Tu versión original: convierte todo X a tensor en device y ejecuta de golpe.
    """
    model.eval()
    Xt = torch.tensor(Xnp, dtype=torch.float32, device=device)
    return torch.sigmoid(model(Xt)).cpu().numpy()

@torch.inference_mode()
def predict_proba_new(model,
                      Xnp: np.ndarray,
                      device: torch.device,
                      batch_size: int = 8192,
                      use_amp: bool = True):
    """
    Versión optimizada: batching, pin_memory, non_blocking y AMP opcional (CUDA).
    """
    model.eval()

    tcpu = torch.from_numpy(Xnp)  # cero copia en CPU
    outs = []
    is_cuda = (device.type == "cuda")

    # Contexto AMP solo en CUDA
    amp_ctx = torch.amp.autocast('cuda',dtype=torch.float16) if (use_amp and is_cuda) else torch.no_grad()
    with amp_ctx:
        for i in range(0, len(tcpu), batch_size):
            xb = tcpu[i:i + batch_size].to(device, non_blocking=True)
            probs = torch.sigmoid_(model(xb)).float().cpu()  # sigmoid in-place + cast a float32 en CPU
            outs.append(probs)
    return torch.vstack(outs).numpy()



@torch.inference_mode()
def predict_proba_new_nocopy(model, Xnp, device, batch_size=65536, use_amp=True):
    model.eval()
    tdev = torch.from_numpy(Xnp).to(device)  # una sola copia H→D
    outs = []
    ctx = torch.amp.autocast("cuda", dtype=torch.float16) if (use_amp and device.type=="cuda") else torch.no_grad()
    with ctx:
        for i in range(0, len(tdev), batch_size):
            xb = tdev[i:i+batch_size]
            probs = torch.sigmoid_(model(xb)).float().cpu()
            outs.append(probs)
    return torch.vstack(outs).numpy()
# --------------------------
# Benchmark helpers
# --------------------------
def measure_mem_cuda(fn):
    if not torch.cuda.is_available():
        return None
    torch.cuda.reset_peak_memory_stats()
    _ = fn()
    sync()
    peak = torch.cuda.max_memory_allocated()
    return peak

def bench(fn, label, nproc_per_iter, warmup=5, repeats=20):
    # Warmup
    for _ in range(warmup):
        _ = fn()
        sync()

    # Memoria pico (CUDA)
    mem_peak = measure_mem_cuda(fn)

    # Timing
    lat_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = fn()
        sync()
        lat_ms.append((time.perf_counter() - t0) * 1000)

    thr_list = [nproc_per_iter / (t / 1000.0) for t in lat_ms]
    print(f"\n== {label} ==")
    print(f" Latencia: media {mean(lat_ms):.2f} ms | p95 {p95(lat_ms):.2f} ms | stdev {stdev(lat_ms):.2f} ms")
    print(f" Throughput medio: {mean(thr_list):,.0f} muestras/s")
    if mem_peak is not None:
        print(f" Memoria pico CUDA: {human_mb(mem_peak)}")
    return mean(thr_list)

# --------------------------
# Main
# --------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark inferencia: original vs optimizada")
    parser.add_argument("--N", type=int, default=200_000, help="nº de muestras")
    parser.add_argument("--d", type=int, default=300, help="nº de features")
    parser.add_argument("--C", type=int, default=50, help="nº de clases (multilabel)")
    parser.add_argument("--batch-size", type=int, default=8192, help="batch_size para versión optimizada")
    parser.add_argument("--repeats", type=int, default=20, help="repeticiones cronometradas")
    parser.add_argument("--warmup", type=int, default=5, help="iteraciones de calentamiento")
    parser.add_argument("--use-amp", action="store_true", help="activar AMP (CUDA) en versión optimizada")
    parser.add_argument("--backend", choices=["none", "eager", "inductor"], default="none",
                        help="torch.compile backend (inductor requiere Triton)")
    parser.add_argument("--seed", type=int, default=0, help="semilla RNG")
    args = parser.parse_args()

    # Semillas
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device y TF32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    set_tf32_if_available()

    # Datos sintéticos (sustituye por tus datos si quieres)
    print(f"Generando datos: N={args.N}, d={args.d}, C={args.C}")
    X = np.random.randn(args.N, args.d).astype(np.float32)

    # Modelo (sustituye por el tuyo si ya lo tienes)
    model = Dummy(args.d, args.C).to(device).eval()
    model = maybe_compile(model, args.backend)

    # Bench original (todo de golpe)
    thr_orig = bench(
        lambda: predict_proba_original(model, X, device),
        f"ORIGINAL (todo de una vez, backend={args.backend})",
        nproc_per_iter=args.N,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    # Bench optimizada (batched)
    thr_new = bench(
        lambda: predict_proba_new(model, X, device, batch_size=args.batch_size, use_amp=args.use_amp),
        f"NUEVA (batch={args.batch_size}, AMP={args.use_amp}, backend={args.backend})",
        nproc_per_iter=args.N,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    thr_new_nocopy= bench(
        lambda: predict_proba_new_nocopy(model, X, device, batch_size=args.batch_size, use_amp=args.use_amp),
        f"NUEVA NO COPY(batch={args.batch_size}, AMP={args.use_amp}, backend={args.backend})",
        nproc_per_iter=args.N,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    print(f"\nSpeedup NUEVA vs ORIGINAL: ×{thr_new / thr_orig:.2f}")

    # Validación rápida de precisión numérica (subset para ir rápido)
    subset = min(50_000, args.N)
    probs_o = predict_proba_original(model, X[:subset], device)
    probs_n = predict_proba_new(model, X[:subset], device, batch_size=args.batch_size, use_amp=args.use_amp)
    probs_n_nc = predict_proba_new_nocopy(model, X[:subset], device, batch_size=args.batch_size, use_amp=args.use_amp)
    max_abs_err = float(np.max(np.abs(probs_o - probs_n)))
    print(f"Diferencia máx. abs (subset={subset}): {max_abs_err:.3e}")

    max_abs_err_nc = float(np.max(np.abs(probs_o - probs_n_nc)))
    print(f"Diferencia máx. abs no copy (subset={subset}): {max_abs_err_nc:.3e}")

if __name__ == "__main__":
    main()

