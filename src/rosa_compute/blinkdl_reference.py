import torch


def rosa_slow_ref(q, k, v):
    """
    BlinkDL reference ROSA route search oracle on integer symbol lists.
    q, k, v: list of ints of length T
    returns: (idx, ln) lists of length T
    """
    n = len(q)
    idx = [0] * n
    ln = [0] * n
    for i in range(n):
        found = False
        for w in range(i + 1, 0, -1):
            t = q[i + 1 - w : i + 1]
            for j in range(i - w, -1, -1):
                if k[j : j + w] == t:
                    s = j + w
                    idx[i] = v[s]
                    ln[i] = w
                    found = True
                    break
            if found:
                break
    return idx, ln

def blinkdl_rosa_4bit_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, emb: torch.Tensor = None) -> torch.Tensor:
    """
    BlinkDL ROSA-4bit reference implementation.
    q, k, v: [B, T, C] where C is divisible by 4.
    emb: optional [1, 1, C] parameter. If provided, applies BlinkDL learned embedding reconstruction (+emb for 1, -emb for 0).
    """
    B, T, C = q.shape
    bits = 4
    assert C % bits == 0, f"Channel size C={C} must be divisible by {bits}"
    G = C // bits

    qb = (q > 0).to(torch.uint8).cpu()
    kb = (k > 0).to(torch.uint8).cpu()
    vb = (v > 0).to(torch.uint8).cpu()

    if emb is not None:
        ee = emb.detach().cpu()
    else:
        ee = None

    out = torch.zeros((B, T, C), dtype=q.dtype)

    for b in range(B):
        for g in range(G):
            qsym = [0] * T
            ksym = [0] * T
            vsym = [0] * T
            for bb in range(bits):
                ch = g * bits + bb
                qsym = [qsym[t] | (int(qb[b, t, ch]) << bb) for t in range(T)]
                ksym = [ksym[t] | (int(kb[b, t, ch]) << bb) for t in range(T)]
                vsym = [vsym[t] | (int(vb[b, t, ch]) << bb) for t in range(T)]

            idx, ln = rosa_slow_ref(qsym, ksym, vsym)

            for t in range(T):
                if ln[t] > 0:
                    sym = idx[t]
                    for bb in range(bits):
                        ch = g * bits + bb
                        bit = (sym >> bb) & 1
                        if ee is not None:
                            sign = 1.0 if bit == 1 else -1.0
                            out[b, t, ch] = sign * ee[0, 0, ch].item()
                        else:
                            out[b, t, ch] = float(bit)
                else:
                    for bb in range(bits):
                        ch = g * bits + bb
                        out[b, t, ch] = 0.0

    return out.to(q.device)
