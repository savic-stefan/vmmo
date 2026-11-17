import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from math import pi
import time

A_h = 100   
A_w = 78    
K_images = 30
sigma_noise_true = 0.06
n_restarts = 3
max_iters = 100
tol_elbo = 1e-4

def try_infer_shape_1d(arr):
    L = arr.size
    s = int(np.sqrt(L))
    for d in range(s, 7, -1):
        if L % d == 0:
            return d, L // d
    return s, int(np.ceil(L / s))

def load_noisy_csv(path):
    arr = np.loadtxt(path, delimiter=",")
    if arr.ndim == 1:
        H, W = try_infer_shape_1d(arr)
        if H * W != arr.size:
            pad = H * W - arr.size
            arr = np.concatenate([arr, np.zeros(pad)])
        return arr.reshape((H, W))
    return arr

def load_photo_gray(path, min_h=A_h, min_w=A_w):
    im = Image.open(path).convert('L')
    arr = np.asarray(im).astype(np.float32)/255.0
    H, W = arr.shape
    if H < min_h or W < min_w:
        scale = max(min_h/H, min_w/W, 1.0)
        newH = int(np.ceil(H*scale))
        newW = int(np.ceil(W*scale))
        im = im.resize((newW, newH), Image.BILINEAR)
        arr = np.asarray(im).astype(np.float32)/255.0
    return arr

def add_impulse_noise(image, p=0.02):
    out = image.copy()
    num = image.size
    nnoisy = int(p * num)
    if nnoisy == 0:
        return out
    coords = np.random.choice(num, nnoisy, replace=False)
    out.flat[coords] = np.random.choice([0.0, 1.0], size=nnoisy)
    return out

def possible_positions(H, W, Ah, Aw):
    rows = list(range(0, H - Ah + 1))
    cols = list(range(0, W - Aw + 1))
    positions = [(r,c) for r in rows for c in cols]
    return positions, len(rows), len(cols)

def logsumexp(a, axis=None):
    a_max = np.max(a, axis=axis, keepdims=True)
    res = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    if axis is None:
        return res.squeeze()
    return res.squeeze()

# модель: лог p(x | pos, theta)
def log_p_x_given_pos(image, B, F, sigma2, pos):
    H, W = image.shape
    Ah, Aw = F.shape
    r,c = pos
    fr0, fr1 = r, r+Ah
    fc0, fc1 = c, c+Aw
    mask = np.ones_like(image, dtype=bool)
    mask[fr0:fr1, fc0:fc1] = False
    diff_out = image[mask] - B[mask]
    sse = np.sum(diff_out*diff_out)
    region = image[fr0:fr1, fc0:fc1]
    diff_in = region - F
    sse += np.sum(diff_in*diff_in)
    n = image.size
    logp = -0.5 * n * np.log(2*pi*sigma2) - sse/(2.0*sigma2)
    return logp

def precompute_logpx(image, B, F, sigma2, positions):
    P = len(positions)
    out = np.empty(P, dtype=np.float64)
    for i,pos in enumerate(positions):
        out[i] = log_p_x_given_pos(image, B, F, sigma2, pos)
    return out

# E-step
def E_step(images, B, F, sigma2, A_prior, positions, hard=False):
    K = len(images)
    P = len(positions)
    A_flat = A_prior.ravel()
    logA = np.log(A_flat + 1e-20)
    logpx = np.zeros((K,P))
    for k,img in enumerate(images):
        logpx[k,:] = precompute_logpx(img, B, F, sigma2, positions)
    log_joint = logpx + logA[np.newaxis,:]
    if hard:
        gamma = np.zeros_like(log_joint)
        idx = np.argmax(log_joint, axis=1)
        gamma[np.arange(K), idx] = 1.0
    else:
        lse = logsumexp(log_joint, axis=1)  # (K,)
        gamma = np.exp(log_joint - lse[:,None])
    return gamma, log_joint

# M-step (строго порядок: A -> F -> B -> sigma2)
def M_step(images, gamma, positions, H, W, Ah, Aw):
    K,P = gamma.shape
    A_flat = np.sum(gamma, axis=0) / K
    rows = sorted(set([p[0] for p in positions]))
    cols = sorted(set([p[1] for p in positions]))
    R = len(rows); C = len(cols)
    A_prior = A_flat.reshape((R,C))
    sumF = np.zeros((Ah,Aw), dtype=np.float64)
    weightF = np.zeros((Ah,Aw), dtype=np.float64)
    for k,img in enumerate(images):
        for p_idx,(r,c) in enumerate(positions):
            w = gamma[k,p_idx]
            if w==0: continue
            sumF += w * img[r:r+Ah, c:c+Aw]
            weightF += w
    F = np.copy(sumF)
    nzf = weightF>0
    if np.any(nzf):
        F[nzf] = sumF[nzf]/weightF[nzf]
    if np.any(~nzf):
        F[~nzf] = np.mean([im.mean() for im in images])
    sumB = np.zeros((H,W), dtype=np.float64)
    weightB = np.zeros((H,W), dtype=np.float64)
    for k,img in enumerate(images):
        for p_idx,(r,c) in enumerate(positions):
            w = gamma[k,p_idx]
            if w==0: continue
            mask = np.ones((H,W), dtype=bool)
            mask[r:r+Ah, c:c+Aw] = False
            sumB[mask] += w * img[mask]
            weightB[mask] += w
    B = np.copy(sumB)
    nz = weightB > 0
    if np.any(nz):
        B[nz] = sumB[nz]/weightB[nz]
    if np.any(~nz):
        B[~nz] = np.mean([im.mean() for im in images])
    sse = 0.0
    denom = 0.0
    for k,img in enumerate(images):
        for p_idx,(r,c) in enumerate(positions):
            w = gamma[k,p_idx]
            if w==0: continue
            mask = np.ones((H,W), dtype=bool)
            mask[r:r+Ah, c:c+Aw] = False
            diff_out = img[mask] - B[mask]
            sse += w * np.sum(diff_out*diff_out)
            diff_in = img[r:r+Ah, c:c+Aw] - F
            sse += w * np.sum(diff_in*diff_in)
            denom += w * (H*W)
    sigma2 = sse / (denom + 1e-16)
    return A_prior, F, B, sigma2

# ELBO
def compute_ELBO(gamma, log_joint):
    eps = 1e-20
    g = np.clip(gamma, eps, 1.0)
    return np.sum(g * (log_joint - np.log(g)))

# EM algorithm
def EM(images, Ah, Aw, n_iters=max_iters, tol=tol_elbo, init_mode='from_image', hard=False, verbose=True):
    K = len(images)
    H,W = images[0].shape
    positions, R, C = possible_positions(H,W,Ah,Aw)
    P = len(positions)
    A_prior = np.ones((R,C))/(R*C)
    B = np.mean(np.stack(images, axis=0), axis=0)
    if init_mode=='from_image':
        r0 = (H-Ah)//2; c0 = (W-Aw)//2
        F = images[0][r0:r0+Ah, c0:c0+Aw].copy()
    else:
        F = np.random.normal(loc=B.mean(), scale=0.05, size=(Ah,Aw))
    sigma2 = np.var(np.stack(images).ravel())
    prev_elbo = -np.inf
    for it in range(n_iters):
        gamma, log_joint = E_step(images, B, F, sigma2, A_prior, positions, hard=hard)
        elbo_before = compute_ELBO(gamma, log_joint)
        if verbose:
            print(f"Iter {it:03d} ELBO(before M)={elbo_before:.6f} sigma2={sigma2:.6e}")
        A_prior, F, B, sigma2 = M_step(images, gamma, positions, H, W, Ah, Aw)
        gamma2, log_joint2 = E_step(images, B, F, sigma2, A_prior, positions, hard=hard)
        elbo_after = compute_ELBO(gamma2, log_joint2)
        if verbose:
            print(f"       ELBO(after M)={elbo_after:.6f}")
        if abs(elbo_after - prev_elbo) < tol:
            if verbose: print("Converged by ELBO change.")
            gamma, log_joint = gamma2, log_joint2
            break
        prev_elbo = elbo_after
        gamma, log_joint = gamma2, log_joint2
    return {'A':A_prior, 'F':F, 'B':B, 'sigma2':sigma2, 'positions':positions, 'gamma':gamma, 'log_joint':log_joint}

def multiple_restarts(images, Ah, Aw, restarts=n_restarts, hard=False):
    best=None; best_elbo=-np.inf
    for s in range(restarts):
        init = 'from_image' if s==0 else 'random'
        print(f"Restart {s+1}/{restarts}, init={init}, hard={hard}")
        res = EM(images, Ah, Aw, init_mode=init, hard=hard, verbose=True)
        elbo = compute_ELBO(res['gamma'], res['log_joint'])
        print(f"  final ELBO={elbo:.6f}")
        if elbo > best_elbo:
            best_elbo=elbo; best=res
    print("Best ELBO:", best_elbo)
    return best

# Влияние начального приближения 
def experiment_initialization_effect(images, Ah, Aw, restarts=3):
    results = []
    for init_mode in ['from_image', 'random']:
        res = EM(images, Ah, Aw, init_mode=init_mode, hard=False, verbose=False)
        elbo = compute_ELBO(res['gamma'], res['log_joint'])
        results.append((init_mode, res, elbo))
        print(f"Init mode={init_mode}, final ELBO={elbo:.6f}")
    return results

# Размер выборки и уровень шума 
def experiment_sample_size_and_noise(B_true, F_true, sample_sizes=[5,10,30], noise_levels=[0.02,0.05,0.1], Ah=A_h, Aw=A_w):
    experiments = []
    H,W = B_true.shape
    positions, _, _ = possible_positions(H,W,Ah,Aw)
    for K in sample_sizes:
        for sigma_noise in noise_levels:
            images=[]
            for k in range(K):
                pos = positions[np.random.randint(len(positions))]
                r,c = pos
                im = B_true.copy()
                im[r:r+Ah, c:c+Aw] = F_true
                im = im + np.random.normal(scale=sigma_noise, size=im.shape)
                im = np.clip(im,0,1)
                images.append(im)
            res = multiple_restarts(images, Ah, Aw, restarts=2, hard=False)
            elbo_norm = compute_ELBO(res['gamma'], res['log_joint']) / K
            experiments.append((K, sigma_noise, res, elbo_norm))
            print(f"K={K}, sigma={sigma_noise:.3f}, normalized ELBO={elbo_norm:.6f}")
    return experiments

# Применение EM к реальным данным 
def experiment_real_data(real_images, Ah, Aw, restarts=3, hard=False):
    results = []
    for K in [5, 10, len(real_images)]:
        images_subset = real_images[:K]
        res = multiple_restarts(images_subset, Ah, Aw, restarts=restarts, hard=hard)
        results.append((K, res))
        print(f"Real data subset size={K}, final ELBO={compute_ELBO(res['gamma'], res['log_joint']):.6f}")
    return results

# Пример модификации EM SS
def propose_modification(images, Ah, Aw, method='mini_batch', batch_size=5, n_iters=50):
    K = len(images)
    H,W = images[0].shape
    positions, R, C = possible_positions(H,W,Ah,Aw)
    P = len(positions)
    A_prior = np.ones((R,C))/(R*C)
    B = np.mean(np.stack(images, axis=0), axis=0)
    r0 = (H-Ah)//2; c0 = (W-Aw)//2
    F = images[0][r0:r0+Ah, c0:c0+Aw].copy()
    sigma2 = np.var(np.stack(images).ravel())
    prev_elbo = -np.inf
    
    for it in range(n_iters):
        batch_idx = np.random.choice(K, min(batch_size,K), replace=False)
        batch_images = [images[k] for k in batch_idx]
        gamma, log_joint = E_step(batch_images, B, F, sigma2, A_prior, positions, hard=False)
        A_prior, F, B, sigma2 = M_step(batch_images, gamma, positions, H, W, Ah, Aw)
        gamma2, log_joint2 = E_step(batch_images, B, F, sigma2, A_prior, positions, hard=False)
        elbo_after = compute_ELBO(gamma2, log_joint2)
        if it % 5 == 0:
            print(f"Iter {it}, ELBO (mini-batch)={elbo_after:.6f}")
        prev_elbo = elbo_after
    return {'A':A_prior, 'F':F, 'B':B, 'sigma2':sigma2, 'positions':positions, 'gamma':gamma2, 'log_joint':log_joint2}

def compare_runtime(images, Ah, Aw):
    print("\n 7")

    # --- Замер обычного EM ---
    start_em = time.time()
    res_em = EM(images, Ah, Aw, init_mode='from_image', hard=False, verbose=False)
    end_em = time.time()
    em_time = end_em - start_em
    elbo_em = compute_ELBO(res_em['gamma'], res_em['log_joint'])

    # --- Замер mini-batch EM ---
    start_mb = time.time()
    res_mb = propose_modification(images, Ah, Aw, method='mini_batch', batch_size=5, n_iters=50)
    end_mb = time.time()
    mb_time = end_mb - start_mb
    elbo_mb = compute_ELBO(res_mb['gamma'], res_mb['log_joint'])

    # --- Вывод результатов ---
    print(f"Обычный EM:     время = {em_time:.2f} сек, ELBO = {elbo_em:.2f}")
    print(f"Mini-batch EM:  время = {mb_time:.2f} сек, ELBO = {elbo_mb:.2f}")
    print(f"Ускорение Mini-batch: {em_time/mb_time:.2f}x")

    return {
        'time_em': em_time,
        'time_mb': mb_time,
        'elbo_em': elbo_em,
        'elbo_mb': elbo_mb,
        'speedup': em_time/mb_time
    }

def display_npy_results():
    npy_files = [
        "res_soft_A.npy", "res_soft_F.npy", "res_soft_B.npy", "res_soft_sigma2.npy",
        "res_hard_A.npy", "res_hard_F.npy", "res_hard_B.npy", "res_hard_sigma2.npy"
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i, file in enumerate(npy_files):
        if os.path.exists(file):
            data = np.load(file)
            
            # Для скаляров (sigma2)
            if np.isscalar(data) or (data.ndim == 1 and data.size == 1):
                value = data.item()  # получаем число из массива
                axes[i].text(0.5, 0.5, f"{value:.6f}", ha='center', va='center', fontsize=12)
                axes[i].set_title(f"{file}\n(sigma²)")
            
            # Для матриц A (heatmap)
            elif 'A.npy' in file:
                im = axes[i].imshow(data, cmap='hot')
                plt.colorbar(im, ax=axes[i])
                axes[i].set_title(f"{file}\nPrior A")
            
            # Для изображений F и B (grayscale)
            elif 'F.npy' in file or 'B.npy' in file:
                im = axes[i].imshow(data, cmap='gray')
                plt.colorbar(im, ax=axes[i])
                obj_type = "Template F" if 'F.npy' in file else "Background B"
                axes[i].set_title(f"{file}\n{obj_type}")
            
            axes[i].axis('off')
        else:
            axes[i].text(0.5, 0.5, "File not found", ha='center', va='center')
            axes[i].set_title(file)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    if not os.path.exists("photo.png"):
        raise FileNotFoundError("photo.png missing")
    if not os.path.exists("noisy_image.csv"):
        raise FileNotFoundError("noisy_image.csv missing")
    B_true = load_photo_gray("photo.png", min_h=A_h, min_w=A_w)
    noisy = load_noisy_csv("noisy_image.csv")
    # ensure noisy contains at least A_h x A_w; take central crop as template per ТЗ
    hn, wn = noisy.shape
    if hn < A_h or wn < A_w:
        noisy = np.pad(noisy, ((max(0,(A_h-hn))//2, max(0,(A_h-hn)) - max(0,(A_h-hn))//2),
                                (max(0,(A_w-wn))//2, max(0,(A_w-wn)) - max(0,(A_w-wn))//2)),
                       mode='constant', constant_values=0.0)
        hn, wn = noisy.shape
    sr = (hn - A_h)//2; sc = (wn - A_w)//2
    F_true = noisy[sr:sr+A_h, sc:sc+A_w].copy()
    H,W = B_true.shape
    positions, Rpos, Cpos = possible_positions(H,W,A_h,A_w)
    images=[]; true_pos=[]
    for k in range(K_images):
        pos = positions[np.random.randint(len(positions))]
        r,c = pos
        im = B_true.copy()
        im[r:r+A_h, c:c+A_w] = F_true
        im = im + np.random.normal(scale=sigma_noise_true, size=im.shape)
        im = np.clip(im,0,1)
        im = add_impulse_noise(im, p=0.02)
        images.append(im); true_pos.append(pos)
    print("Running soft-EM")
    res_soft = multiple_restarts(images, A_h, A_w, restarts=n_restarts, hard=False)
    print("Running hard-EM (Viterbi / Вага-EM)")
    res_hard = multiple_restarts(images, A_h, A_w, restarts=n_restarts, hard=True)
    np.save("res_soft_A.npy", res_soft['A'])
    np.save("res_soft_F.npy", res_soft['F'])
    np.save("res_soft_B.npy", res_soft['B'])
    np.save("res_soft_sigma2.npy", np.array([res_soft['sigma2']]))
    np.save("res_hard_A.npy", res_hard['A'])
    np.save("res_hard_F.npy", res_hard['F'])
    np.save("res_hard_B.npy", res_hard['B'])
    np.save("res_hard_sigma2.npy", np.array([res_hard['sigma2']]))
    plt.figure(figsize=(12,6))
    plt.subplot(2,4,1); plt.title("Background (photo.png)"); plt.imshow(B_true, cmap='gray'); plt.axis('off')
    plt.subplot(2,4,2); plt.title("Template F_true (from noisy_image.csv)"); plt.imshow(F_true, cmap='gray'); plt.axis('off')
    plt.subplot(2,4,3); plt.title("Example generated image"); plt.imshow(images[0], cmap='gray'); plt.axis('off')
    heat = np.zeros((Rpos, Cpos))
    for (r,c) in true_pos:
        heat[r,c]+=1
    plt.subplot(2,4,4); plt.title("True positions heat"); plt.imshow(heat, cmap='hot'); plt.colorbar()
    plt.subplot(2,4,5); plt.title("Estimated A (soft)"); plt.imshow(res_soft['A'], cmap='hot'); plt.axis('off')
    plt.subplot(2,4,6); plt.title("Estimated F (soft)"); plt.imshow(res_soft['F'], cmap='gray'); plt.axis('off')
    plt.subplot(2,4,7); plt.title("Estimated A (hard)"); plt.imshow(res_hard['A'], cmap='hot'); plt.axis('off')
    plt.subplot(2,4,8); plt.title("Estimated F (hard)"); plt.imshow(res_hard['F'], cmap='gray'); plt.axis('off')
    plt.tight_layout(); plt.savefig("em_results_overview.png", dpi=150); plt.show()

    # ---------------- Эксперименты для отчета ----------------
    print("\n 1 - Влияние начальной инициализации")
    results_init = experiment_initialization_effect(images, A_h, A_w)
    
    print("\n 2 - Размер выборки и уровень шума")
    results_noise = experiment_sample_size_and_noise(B_true, F_true)
    
    print("\n 6 - ример модификации EM (mini-batch)")
    res_mod = propose_modification(images, A_h, A_w)

    print("\n 7 - Сравнение времени работы EM и Mini-batch EM")
    runtime_results = compare_runtime(images, A_h, A_w)

if __name__ == "__main__":
    main()
    display_npy_results()