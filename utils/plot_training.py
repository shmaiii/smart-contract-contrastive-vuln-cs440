import argparse, re, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

'''
This utility function is here because we only realized the need to parse loss graphs after training was finished.
This is a script to parse the logs we obtained from training into plots
'''

BATCH = re.compile(r"Epoch\s+(\d+)/\d+:.*loss=([\d.]+)")
VAL   = re.compile(r"validation.*?f1=([\d.]+).*?pr_auc=([\d.]+).*?roc_auc=([\d.]+)")
ANSI  = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

def smooth(v, w):
    out, n = [], len(v)
    for i in range(n):
        lo, hi = max(0, i-w//2), min(n, i+w//2+1)
        out.append(sum(v[lo:hi])/(hi-lo))
    return out

def parse(path):
    seen, order, vals, ep_starts = {}, [], [], {}
    with open(path, errors="replace") as f:
        for raw in f:
            line = ANSI.sub("", raw)
            m = BATCH.search(line)
            if m:
                ep, loss = int(m.group(1)), float(m.group(2))
                cnt = re.search(r"\|\s*(\d+)/\d+", line)
                bn = int(cnt.group(1)) if cnt else 0
                k = (ep, bn)
                if k not in seen:
                    order.append(k)
                    ep_starts.setdefault(ep, len(seen))
                seen[k] = loss
                continue
            vm = VAL.search(line)
            if vm:
                vals.append((len(seen)-1, float(vm.group(1)), float(vm.group(2)), float(vm.group(3))))
    losses = [seen[k] for k in order]
    ep_idx = {ep: sum(1 for k in order if k[0] < ep) for ep in ep_starts}
    return losses, ep_idx, vals

def plot(losses, ep_idx, val_pts, w, out):
    x, s = list(range(len(losses))), smooth(losses, w)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9))
    ax1.plot(x, losses, color="#aec6e8", lw=0.4, alpha=0.6, label="batch loss (raw)")
    ax1.plot(x, s, color="#1f77b4", lw=2, label=f"smoothed w={w}")
    ymax = max(losses)*1.05
    for ep, idx in sorted(ep_idx.items()):
        ax1.axvline(idx, color="gray", ls="--", lw=1)
        ax1.text(idx+len(x)*0.005, ymax*0.97, f"Epoch {ep}", fontsize=9, color="gray")
    ax1.set(xlim=(0,len(x)), ylim=(0,ymax), xlabel="Batch (global)", ylabel="Loss", title="Training Loss per Batch")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    if val_pts:
        vx=[p[0] for p in val_pts]; vf=[p[1] for p in val_pts]; vp=[p[2] for p in val_pts]; vr=[p[3] for p in val_pts]
        for v2,mk,col,lab in [(vf,"o","#d62728","F1"),(vp,"s","#2ca02c","PR-AUC"),(vr,"^","#ff7f0e","ROC-AUC")]:
            ax2.plot(vx,v2,marker=mk,ms=9,lw=2,color=col,label=lab)
            for xi,yi in zip(vx,v2): ax2.annotate(f"{yi:.4f}",(xi,yi),textcoords="offset points",xytext=(6,5),fontsize=9,color=col)
        for ep,idx in sorted(ep_idx.items()):
            ax2.axvline(idx,color="gray",ls="--",lw=1)
            ax2.text(idx+len(losses)*0.005,0.808,f"Epoch {ep}",fontsize=9,color="gray")
        ax2.set(xlim=(0,len(losses)),ylim=(0.80,1.01))
    ax2.set(xlabel="Batch (global)",ylabel="Score",title="Validation Metrics (end of each epoch)")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out, dpi=150, bbox_inches="tight"); print(f"Saved -> {out}")

parser = argparse.ArgumentParser()
parser.add_argument("log"); parser.add_argument("--out",default="training_curves.png"); parser.add_argument("--smooth",type=int,default=200)
args = parser.parse_args()
losses, ep_idx, val_pts = parse(args.log)
print(f"Parsed {len(losses)} steps, {len(val_pts)} val points")
if not losses: sys.exit(1)
plot(losses, ep_idx, val_pts, args.smooth, args.out)
