"""Metric suite: accuracy, generalization gap, calibration (ECE), adversarial
robustness (FGSM, PGD).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def predict_probs(net, x):
    net.eval()
    with torch.no_grad():
        logits = net(x)
    return torch.softmax(logits, dim=1).numpy()


def accuracy(net, x, y):
    p = predict_probs(net, x)
    return float((p.argmax(1) == y.numpy()).mean())


def expected_calibration_error(net, x, y, n_bins=15):
    """Expected Calibration Error with the standard equal-width binning."""
    p = predict_probs(net, x)
    conf = p.max(1)
    acc = (p.argmax(1) == y.numpy()).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if i == 0:
            mask = (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(conf)) * abs(acc[mask].mean() - conf[mask].mean())
    return float(ece)


def fgsm_attack(net, x, y, eps, clamp=(0, 1)):
    net.eval()
    xin = x.clone().requires_grad_(True)
    logits = net(xin)
    loss = F.cross_entropy(logits, y)
    grad = torch.autograd.grad(loss, xin)[0]
    xadv = xin + eps * torch.sign(grad)
    xadv = torch.clamp(xadv, *clamp)
    with torch.no_grad():
        out = net(xadv)
    return float((out.argmax(1) == y).float().mean())


def pgd_attack(net, x, y, eps, alpha, steps, clamp=(0, 1)):
    net.eval()
    xadv = x.clone()
    for _ in range(steps):
        xadv.requires_grad_(True)
        logits = net(xadv)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, xadv)[0]
        xadv = (xadv + alpha * torch.sign(grad)).detach()
        delta = torch.clamp(xadv - x, -eps, eps)
        xadv = torch.clamp(x + delta, *clamp)
    with torch.no_grad():
        out = net(xadv)
    return float((out.argmax(1) == y).float().mean())


def full_metric_suite(net, x, y):
    """All headline metrics for a (probe_x, probe_y) pair. Returns dict."""
    p = predict_probs(net, x)
    acc = float((p.argmax(1) == y.numpy()).mean())
    ece = expected_calibration_error(net, x, y)
    f01 = fgsm_attack(net, x, y, 0.1)
    f03 = fgsm_attack(net, x, y, 0.3)
    pgd = pgd_attack(net, x, y, 0.3, 0.06, 10)
    return dict(acc=acc, ece=ece, fgsm01=f01, fgsm03=f03, pgd=pgd)
