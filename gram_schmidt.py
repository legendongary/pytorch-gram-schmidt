import torch


def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(1)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[:, k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    a = torch.randn(5, 2, requires_grad=True)
    b = gram_schmidt(a)
    c = b.sum()
    c.backward()
    print(b.matmul(b.t()))
    print(a.grad)
