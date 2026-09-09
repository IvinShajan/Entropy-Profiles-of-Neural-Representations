"""Training harness for the network population.

Trains small MLP / CNN models on MNIST, Fashion-MNIST and a small grayscale
CIFAR-10, sweeping architectures, width, dropout, weight decay and initialisation
scale, with fixed seeds. Saves weights + metrics. Optionally saves per-epoch
checkpoints (for the training-dynamics experiment).

Layer activations are captured with forward hooks and (optionally) the raw
activation matrices of the probe set are saved for offline analysis.

Run:  python3 train_networks.py            # launch the whole sweep
      python3 train_networks.py --only N   # train only config index N
"""
import os, json, argparse, time, hashlib, sys

# Cap BLAS / OpenMP threading BEFORE torch is imported so forked pool workers
# inherit the intended limits (avoids gross CPU oversubscription).
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TORCH_THREADS", "2")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
NET_DIR = os.path.join(BASE, "results", "nets")
os.makedirs(NET_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def load_dataset(name: str, train: bool):
    if name == "mnist":
        ds = torchvision.datasets.MNIST(DATA_DIR, train=train, download=True,
                                        transform=T.ToTensor())
    elif name == "fashion":
        ds = torchvision.datasets.FashionMNIST(DATA_DIR, train=train, download=True,
                                               transform=T.ToTensor())
    elif name == "cifar-small":
        transform = T.Compose([
            T.Resize((16, 16)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
        ])
        ds = torchvision.datasets.CIFAR10(DATA_DIR, train=train, download=True,
                                          transform=transform)
    elif name == "cifar10":
        transform = T.Compose([
            T.RandomCrop(32, padding=4) if train else T.Lambda(lambda x: x),
            T.RandomHorizontalFlip() if train else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])
        ds = torchvision.datasets.CIFAR10(DATA_DIR, train=train, download=True,
                                          transform=transform)
    elif name == "cifar100":
        if train:
            transform = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        else:
            transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            ])
        ds = torchvision.datasets.CIFAR100(DATA_DIR, train=train, download=True,
                                           transform=transform)
    elif name == "svhn":
        split = "train" if train else "test"
        ds = torchvision.datasets.SVHN(DATA_DIR, split=split, download=True,
                                       transform=T.ToTensor())
    else:
        raise ValueError(name)
    return ds


def _balanced_subset(ds, n, n_classes=10):
    """Balanced subset with n//n_classes images per class."""
    per = max(1, n // n_classes)
    idx = []
    for c in range(n_classes):
        ci = [i for i, (_, y) in enumerate(ds) if int(y) == c]
        rng = np.random.default_rng(42)
        idx.extend(list(rng.choice(ci, min(per, len(ci)), replace=False)))
    return Subset(ds, idx)


def n_classes_of(dataset):
    return {"mnist": 10, "fashion": 10, "cifar-small": 10, "cifar10": 10,
            "cifar100": 100, "svhn": 10}[dataset]


# ---------------------------------------------------------------------------
# Models (blocks are the layers whose activations we study)
# ---------------------------------------------------------------------------

def mlp(dims, dropout=0.0, init_scale=1.0):
    layers = []
    for i in range(len(dims) - 1):
        lin = nn.Linear(dims[i], dims[i + 1])
        if init_scale != 1.0:
            nn.init.normal_(lin.weight, std=init_scale / np.sqrt(dims[i]))
            lin.bias.data.zero_()
        layers.append(lin)
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    net = nn.Sequential(nn.Flatten(), *layers)
    return net, [dims[i + 1] for i in range(len(dims) - 1)]


def cnn2(dropout=0.0, init_scale=1.0, in_channels=1):
    """conv32-pool-conv64-pool-adapool-flatten-fc256-fc10."""
    def cb(in_c, out_c, k=3, stride=1, pad=1):
        m = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, stride, pad, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        return m
    net = nn.Sequential(
        cb(in_channels, 32), nn.MaxPool2d(2),            # block 0  -> 32x14x14 (28px) / 8x8 (16px)
        cb(32, 64), nn.MaxPool2d(2),           # block 1  -> 64x7x7 / 4x4
        nn.AdaptiveAvgPool2d(2), nn.Flatten(), # block 2  -> 256 features
        nn.Linear(64 * 2 * 2, 256), nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(256, 10))                    # block 3  -> logits
    if init_scale != 1.0:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=init_scale * 0.1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_scale * 0.1)
    return net, [None, None, 256, 10]


def cnn3(dropout=0.0, init_scale=1.0, in_channels=1):
    """conv32-pool-conv64-pool-conv128-gap-fc10."""
    def cb(in_c, out_c, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, k, stride, pad, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
    net = nn.Sequential(
        cb(in_channels, 32), nn.MaxPool2d(2),            # block 0 -> 32x14x14 / 8x8
        cb(32, 64), nn.MaxPool2d(2),           # block 1 -> 64x7x7 / 4x4
        cb(64, 128), nn.MaxPool2d(2),          # block 2 -> 128x3x3 / 2x2
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), # block 3 -> 128
        nn.Linear(128, 10))                    # block 4
    if init_scale != 1.0:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=init_scale * 0.1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=init_scale * 0.1)
    return net, [None, None, None, 128, 10]


class ResNet9(nn.Module):
    """Compact ResNet-9 for small images (CIFAR-100/SVHN at 32x32).

    Named children so that analyze.get_reps hooks each stage (stem, 4 blocks,
    GAP) as a Sequential containing Conv2d. ~2.6M params."""
    def __init__(self, num_classes=100, in_channels=3):
        super().__init__()
        def conv3x3(i, o, stride=1):
            return nn.Conv2d(i, o, 3, stride=stride, padding=1, bias=False)
        class Block(nn.Module):
            def __init__(self, i, o, stride=1):
                super().__init__()
                self.bn1 = nn.BatchNorm2d(i)
                self.c1 = conv3x3(i, o, stride)
                self.bn2 = nn.BatchNorm2d(o)
                self.c2 = conv3x3(o, o)
                self.short = None
                if stride != 1 or i != o:
                    self.short = nn.Conv2d(i, o, 1, stride=stride, bias=False)
            def forward(self, x):
                o = F.relu(self.bn1(x))
                o = F.relu(self.bn2(self.c1(o)))
                o = self.c2(o)
                return o + (self.short(x) if self.short else x)
        self.stem = nn.Sequential(conv3x3(in_channels, 64),
                                  nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.b1 = nn.Sequential(Block(64, 128, stride=2))
        self.b2 = nn.Sequential(Block(128, 128))
        self.b3 = nn.Sequential(Block(128, 256, stride=2))
        self.b4 = nn.Sequential(Block(256, 256))
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.gap(x)
        return self.fc(x)


def resnet9(dropout=0.0, init_scale=1.0, num_classes=100, in_channels=3):
    """Return (ResNet9, layer_dims): one entry per hooked feature layer."""
    net = ResNet9(num_classes=num_classes, in_channels=in_channels)
    if init_scale != 1.0:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=init_scale * 0.1)
    return net, [64, 128, 128, 256, 256, 256]


def resnet18(dropout=0.0, init_scale=1.0, num_classes=100):
    """ResNet-18 (torchvision) with a CIFAR-adapted stem for 32x32 inputs.

    Replaces the 7x7/stride-2 stem and initial max-pool with a 3x3/stride-1
    conv so small images are not over-reduced; keeps the four stage blocks.
    Returns (net, layer_dims) with one entry per stage block plus the
    post-pool feature layer (the layers whose activations we study)."""
    from torchvision.models import resnet18 as _rn
    net = _rn(num_classes=num_classes)
    net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    net.maxpool = nn.Identity()
    block_dims = [64, 128, 256, 512, 512]
    return net, block_dims


def wideresnet(dropout=0.0, init_scale=1.0, num_classes=100, depth=16, widen=4):
    """Simple WideResNet-style block net for CIFAR-100 (CPU-friendly)."""
    def conv3x3(i, o):
        return nn.Conv2d(i, o, 3, padding=1, bias=False)
    class Block(nn.Module):
        def __init__(self, i, o, stride=1):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(i)
            self.c1 = conv3x3(i, o); self.c1.stride = stride
            self.bn2 = nn.BatchNorm2d(o)
            self.c2 = conv3x3(o, o)
            self.short = None
            if stride != 1 or i != o:
                self.short = nn.Conv2d(i, o, 1, stride=stride, bias=False)
        def forward(self, x):
            o = F.relu(self.bn1(x))
            o = self.c1(o)
            o = F.relu(self.bn2(o))
            o = self.c2(o)
            if self.short is not None:
                x = self.short(x)
            return o + x
    layers, k = [], 32 * widen
    strides = [1, 2, 2, 2]
    for si, s in enumerate(strides):
        layers.append(Block(3 if si == 0 else k // 2, k, s))
        for _ in range(depth // 4 - 1):
            layers.append(Block(k, k))
        k *= 2
    net = nn.Sequential(
        conv3x3(3, 32 * widen),
        *layers,
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(16 * 4 * widen, num_classes))
    return net, [None] * (depth // 4 * 4) + [16 * 4 * widen, num_classes]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def configs():
    """The experiment population. Each config is a dict."""
    cfgs = []
    i = 0
    def add(dataset, arch, epochs, dropout=0.0, wd=0.0, lr=1e-3,
            init_scale=1.0, seed=0, width=1, label=None, max_train=None,
            num_classes=10, depth=16, widen=4):
        nonlocal i
        cfgs.append(dict(id=i, dataset=dataset, arch=arch, epochs=epochs,
                         dropout=dropout, wd=wd, lr=lr, init_scale=init_scale,
                         seed=seed, width=width, label=label or f"{arch}-{i}",
                         max_train=max_train, num_classes=num_classes,
                         depth=depth, widen=widen))
        i += 1

    # MNIST population: architectures x regularization
    for arch in ["mlp2", "mlp4", "cnn2", "cnn3"]:
        add("mnist", arch, 12, label=f"mnist-{arch}-base", seed=0)
    for arch in ["mlp2", "cnn2"]:
        add("mnist", arch, 12, dropout=0.3, label=f"mnist-{arch}-drop", seed=1)
        add("mnist", arch, 12, dropout=0.5, label=f"mnist-{arch}-drop5", seed=2)
        add("mnist", arch, 12, wd=5e-4, label=f"mnist-{arch}-wd", seed=3)
        add("mnist", arch, 12, lr=5e-4, label=f"mnist-{arch}-lr5e4", seed=4)
        add("mnist", arch, 12, lr=1e-4, label=f"mnist-{arch}-lr1e4", seed=5)
        add("mnist", arch, 12, init_scale=0.01, label=f"mnist-{arch}-init01", seed=6)
    # wide / narrow variants
    add("mnist", "mlp2w", 12, width=2, label="mnist-mlp2-wide", seed=7)
    add("mnist", "mlp2n", 12, width=0.5, label="mnist-mlp2-narrow", seed=8)
    # seed variability for a single config
    for s in range(9, 14):
        add("mnist", "cnn2", 12, seed=s, label=f"mnist-cnn2-seed{s}")
    # other datasets
    for arch in ["mlp2", "cnn2"]:
        add("fashion", arch, 12, label=f"fashion-{arch}-base", seed=0)
        add("fashion", arch, 12, dropout=0.3, label=f"fashion-{arch}-drop", seed=1)
    add("cifar-small", "mlp2", 15, label="cifar-mlp2-base", seed=0)
    add("cifar-small", "cnn2", 15, label="cifar-cnn2-base", seed=0)

    # --- scale-up population (Nature upgrade) ---
    # CIFAR-100 / SVHN with compact ResNets and small CNNs. ResNet-18/WideResNet
    # are intractable on CPU, so we use ResNet-9 (~2.6M params) which is a
    # modern, residual-architecture proxy and still ~10x cheaper.
    # Each ResNet-9 config: 20 epochs @ 25k stratified images (~2h CPU).
    def rn9(dsname, reg=None, seed=0, label=None, epochs=15):
        add(dsname, "resnet9", epochs,
            **(dict(dropout=0.3) if reg == "drop" else
               dict(wd=5e-4) if reg == "wd" else
               dict(lr=5e-4) if reg == "lr" else {}),
            num_classes=100 if dsname == "cifar100" else 10,
            label=label or f"{dsname}-rn9-{reg or 'base'}{'-s%d' % seed if seed else ''}",
            seed=seed, max_train=20000)
    rn9("cifar100", seed=0)
    rn9("cifar100", "wd", seed=1)
    rn9("cifar100", "drop", seed=2)
    rn9("cifar100", "lr", seed=3)
    rn9("svhn", seed=0)
    rn9("svhn", "wd", seed=1)
    rn9("svhn", "drop", seed=2)
    # cheap CNN baselines at scale (40 epochs, full train set)
    def cnn(dsname, reg=None, seed=0, label=None, arch="cnn3", epochs=40):
        add(dsname, arch, epochs,
            **(dict(dropout=0.3) if reg == "drop" else dict(wd=5e-4) if reg == "wd" else {}),
            num_classes=100 if dsname == "cifar100" else 10,
            label=label or f"{dsname}-{arch}-{reg or 'base'}{'-s%d' % seed if seed else ''}",
            seed=seed)
    cnn("cifar100", arch="cnn3")
    cnn("cifar100", "wd", arch="cnn3")
    cnn("cifar100", "drop", arch="cnn3")
    cnn("cifar100", arch="cnn2")
    cnn("svhn", arch="cnn3")
    # CIFAR-10 color (32x32x3): cheap extra diversity, already-downloaded data
    def c10(reg=None, seed=0, label=None, arch="cnn3", max_train=None):
        add("cifar10", arch, 25,
            **(dict(dropout=0.3) if reg == "drop" else dict(wd=5e-4) if reg == "wd" else {}),
            num_classes=10, label=label or f"cifar10-{arch}-{reg or 'base'}{'-s%d' % seed if seed else ''}",
            seed=seed, max_train=max_train)
    c10(arch="cnn3")
    c10("wd", arch="cnn3")
    c10("drop", arch="cnn3")
    c10(arch="cnn2")
    c10(arch="resnet9", max_train=20000)
    c10(seed=1)
    return cfgs


def build_model(cfg):
    if cfg["arch"] in ("mlp2", "mlp2w", "mlp2n", "mlp4"):
        in_dim = {"mnist": 784, "fashion": 784, "cifar-small": 256}[cfg["dataset"]]
    else:
        in_dim = None
    if cfg["arch"] == "mlp2":
        dims = [in_dim, 256 * cfg.get("width", 1), 10]
        net, layer_dims = mlp(dims, cfg["dropout"], cfg["init_scale"])
    elif cfg["arch"] == "mlp2w":
        dims = [in_dim, 512, 10]
        net, layer_dims = mlp(dims, cfg["dropout"], cfg["init_scale"])
    elif cfg["arch"] == "mlp2n":
        dims = [in_dim, 128, 10]
        net, layer_dims = mlp(dims, cfg["dropout"], cfg["init_scale"])
    elif cfg["arch"] == "mlp4":
        dims = [in_dim, 512, 256, 128, 10]
        net, layer_dims = mlp(dims, cfg["dropout"], cfg["init_scale"])
    elif cfg["arch"] == "cnn2":
        in_ch = 1 if cfg["dataset"] in ("mnist", "fashion", "cifar-small") else 3
        net, layer_dims = cnn2(cfg["dropout"], cfg["init_scale"],
                               in_channels=cfg.get("in_channels", in_ch))
    elif cfg["arch"] == "cnn3":
        in_ch = 1 if cfg["dataset"] in ("mnist", "fashion", "cifar-small") else 3
        net, layer_dims = cnn3(cfg["dropout"], cfg["init_scale"],
                               in_channels=cfg.get("in_channels", in_ch))
    elif cfg["arch"] == "resnet18":
        net, layer_dims = resnet18(cfg["dropout"], cfg["init_scale"],
                                   cfg.get("num_classes", 100))
    elif cfg["arch"] == "resnet9":
        net, layer_dims = resnet9(cfg["dropout"], cfg["init_scale"],
                                  cfg.get("num_classes", 100),
                                  cfg.get("in_channels", 3))
    elif cfg["arch"] == "wideresnet":
        net, layer_dims = wideresnet(cfg["dropout"], cfg["init_scale"],
                                     cfg.get("num_classes", 100),
                                     cfg.get("depth", 16), cfg.get("widen", 4))
    else:
        raise ValueError(cfg["arch"])
    return net, layer_dims


def train_one(cfg, out_dir=NET_DIR, save_checkpoints=False):
    t0 = time.time()
    if os.path.exists(os.path.join(out_dir, f"model_{cfg['label']}.pt")) and \
       os.path.exists(os.path.join(out_dir, f"meta_{cfg['label']}.json")):
        print(f"SKIP {cfg['label']} (already trained)", flush=True)
        return cfg["label"]
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    net, layer_dims = build_model(cfg)

    ds = load_dataset(cfg["dataset"], train=True)
    n_train = len(ds)
    if cfg.get("max_train"):
        m = min(cfg["max_train"], n_train)
        idx = np.random.RandomState(0).choice(n_train, m, replace=False)
        ds = Subset(ds, idx)
    dtest = load_dataset(cfg["dataset"], train=False)

    bs = 128
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0)
    dlt = DataLoader(dtest, batch_size=512, num_workers=0)

    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg["epochs"] // 2, gamma=0.3)
    lossf = nn.CrossEntropyLoss()

    def acc(net_, loader):
        correct = tot = 0
        with torch.no_grad():
            for x, y in loader:
                out = net_(x)
                correct += (out.argmax(1) == y).sum().item()
                tot += y.size(0)
        return correct / tot

    log = {"train_acc": [], "test_acc": [], "epoch_time": []}
    final_only = cfg.get("eval_final_only", True)
    for ep in range(cfg["epochs"]):
        net.train()
        te = time.time()
        for x, y in dl:
            opt.zero_grad()
            loss = lossf(net(x), y)
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        last_ep = (ep + 1 == cfg["epochs"])
        if not final_only or last_ep:
            log["train_acc"].append(acc(net, dl))
            log["test_acc"].append(acc(net, dlt))
            log["epoch_time"].append(time.time() - te)
            print(f"  [{cfg['label']}] ep{ep+1} train={log['train_acc'][-1]:.3f} "
                  f"test={log['test_acc'][-1]:.3f} ({time.time()-te:.0f}s)",
                  flush=True)
        elif (ep + 1) % 3 == 0:
            print(f"  [{cfg['label']}] ep{ep+1} (no eval) "
                  f"({time.time()-te:.0f}s)", flush=True)
        if save_checkpoints and (ep + 1) % 2 == 0:
            torch.save(net.state_dict(), os.path.join(
                out_dir, f"ckpt_{cfg['label']}_ep{ep + 1}.pt"))

    torch.save(net.state_dict(), os.path.join(out_dir, f"model_{cfg['label']}.pt"))
    cfg_out = dict(cfg, layer_dims=layer_dims, train_examples=len(ds))
    with open(os.path.join(out_dir, f"meta_{cfg['label']}.json"), "w") as f:
        json.dump(cfg_out, f, indent=2)
    with open(os.path.join(out_dir, f"log_{cfg['label']}.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"DONE {cfg['label']} in {time.time()-t0:.0f}s", flush=True)
    return cfg["label"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=-1)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--range", type=str, default="")
    args = ap.parse_args()

    all_cfgs = configs()
    if args.range:
        a, b = map(int, args.range.split("-"))
        sel = all_cfgs[a:b]
    elif args.only >= 0:
        sel = [all_cfgs[args.only]]
    else:
        sel = all_cfgs

    print(f"=== training {len(sel)} configs "
          f"(total registered {len(all_cfgs)}) ===")
    if args.parallel > 1 and len(sel) > 1:
        from multiprocessing import Pool
        n = min(args.parallel, len(sel))
        n_threads = max(1, int(os.cpu_count() or 8) // n)
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
        os.environ["TORCH_THREADS"] = str(n_threads)

        def init_worker():
            torch.set_num_threads(n_threads)

        print(f"workers={n} threads/worker={n_threads}", flush=True)
        with Pool(n, initializer=init_worker) as pool:
            pool.map(train_one, sel, chunksize=1)
    else:
        for c in sel:
            train_one(c)
