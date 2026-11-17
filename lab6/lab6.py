import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import time

# 1
amin, amax = 75, 90
bmin, bmax = 500, 600
p1, p2, p3 = 0.1, 0.01, 0.3
N = 50

a_mean = round((amin + amax) / 2)
b_mean = round((bmin + bmax) / 2)  

a_fixed = a_mean  
b_fixed = b_mean

# 2
def uniform_stats(xmin, xmax):
    mean = (xmin + xmax) / 2
    var = (xmax - xmin)**2 / 12
    return mean, var

E_a, Var_a = uniform_stats(amin, amax)
E_b, Var_b = uniform_stats(bmin, bmax)

E_c_n = a_fixed * p1 + b_fixed * p2
Var_c_n = a_fixed * p1 * (1 - p1) + b_fixed * p2 * (1 - p2) 

E_d_n = E_c_n * (1 + p3)

Var_d_n = (1 + p3)**2 * Var_c_n + p3 * (1 - p3) * E_c_n

print("1) Теоретические характеристики:")
print(f"E[a]={E_a:.2f}, Var[a]={Var_a:.2f}")
print(f"E[b]={E_b:.2f}, Var[b]={Var_b:.2f}")
print(f"E[c_n]={E_c_n:.2f}, Var[c_n]={Var_c_n:.2f}")
print(f"E[d_n]={E_d_n:.2f}, Var[d_n]={Var_d_n:.2f}\n")

# 2
def generate(N, a, b, params, model):

    if model != 4:
        raise ValueError("Функция реализована только для модели 4.")
    p1 = params['p1']
    p2 = params['p2']
    p3 = params['p3']

    rng = np.random.default_rng(42) 
    
    c_samples = rng.binomial(n=a, p=p1, size=N) + rng.binomial(n=b, p=p2, size=N)
    
    d_samples = c_samples + rng.binomial(n=c_samples, p=p3, size=N)

    return d_samples

# generator
params_model4 = {'amin': amin, 'amax': amax, 'bmin': bmin, 'bmax': bmax, 'p1': p1, 'p2': p2, 'p3': p3}
d_sample_model4 = generate(N, a_fixed, b_fixed, params_model4, 4)
c_sample_model4 = None 

df_model4 = pd.DataFrame({"n": np.arange(1, N+1), "c_n": [np.nan]*N, "d_n": d_sample_model4})
df_model4.to_csv("sample_model4.csv", index=False)
print("2) Выборка сгенерирована и сохранена (sample_model4.csv)")
print(df_model4.head())
print()

# 3
a_values = np.arange(amin, amax + 1)
b_values = np.arange(bmin, bmax + 1) 

prior_ab = np.ones((len(a_values), len(b_values)))
prior_ab /= prior_ab.sum()

def likelihood_ab(a_val, b_val, d, p1=p1, p2=p2, p3=p3):
    
    lambda_c = a_val * p1 + b_val * p2
    lambda_d = lambda_c * (1 + p3) 
  
    if lambda_d <= 0:
        return 0.0
   
    try:
        return stats.poisson.pmf(d, lambda_d)
    except FloatingPointError:
        
        return np.finfo(float).tiny

posterior_ab_unknown = prior_ab.copy()
posterior_means_unknown = []
posterior_vars_unknown = []

for k in range(1, N + 1):
    d_obs = d_sample_model4[k-1]
    
    lk_matrix = np.array([[likelihood_ab(a_val, b_val, d_obs) for b_val in b_values] for a_val in a_values])
    
    posterior_ab_unknown *= lk_matrix
    posterior_ab_unknown /= posterior_ab_unknown.sum() 
  
    posterior_b_unknown = posterior_ab_unknown.sum(axis=0) 
    posterior_b_unknown /= posterior_b_unknown.sum() 
   
    mean_b = np.sum(b_values * posterior_b_unknown)
    var_b = np.sum((b_values - mean_b)**2 * posterior_b_unknown)

    posterior_means_unknown.append(mean_b)
    posterior_vars_unknown.append(var_b)


posterior_b_known = np.ones_like(b_values, dtype=float)
posterior_b_known /= posterior_b_known.sum()
posterior_means_known = []
posterior_vars_known = []

for k in range(1, N + 1):
    d_obs = d_sample_model4[k-1]
    lk_array = np.array([likelihood_ab(a_fixed, b_val, d_obs) for b_val in b_values])
    posterior_b_known *= lk_array
    posterior_b_known /= posterior_b_known.sum()

    mean_b = np.sum(b_values * posterior_b_known)
    var_b = np.sum((b_values - mean_b)**2 * posterior_b_known)

    posterior_means_known.append(mean_b)
    posterior_vars_known.append(var_b)

plt.figure(figsize=(12, 10))

# 1
plt.subplot(3, 1, 1)
plt.plot(b_values, posterior_b_unknown, label='Posterior p(b|d1..dN), unknown a', alpha=0.7)
plt.axvline(b_fixed, color='red', linestyle='--', label='True b')
plt.title('Апостериорное распределение b (a неизвестно)')
plt.xlabel('b')
plt.ylabel('Вероятность')
plt.legend()
plt.grid(True)

# 2
plt.subplot(3, 1, 2)
plt.plot(range(1, N+1), posterior_means_unknown, label='Unknown a', alpha=0.7)
plt.plot(range(1, N+1), posterior_means_known, label='Known a', alpha=0.7, linestyle='-.')
plt.axhline(b_fixed, color='red', linestyle='--', label='True b')
plt.title('Мат. ожидание b | d1..dk')
plt.xlabel('Номер наблюдения k')
plt.ylabel('E[b]')
plt.legend()
plt.grid(True)

# 3
plt.subplot(3, 1, 3)
plt.plot(range(1, N+1), posterior_vars_unknown, label='Unknown a', alpha=0.7)
plt.plot(range(1, N+1), posterior_vars_known, label='Known a', alpha=0.7, linestyle='-.')
plt.title('Дисперсия b | d1..dk')
plt.xlabel('Номер наблюдения k')
plt.ylabel('Var[b]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("analysis_plots.png")
plt.show()

print("3) Графики построены и сохранены (analysis_plots.png).")
print(f"   Финальное апостериорное среднее b (a неизвестно): {posterior_means_unknown[-1]:.2f}")
print(f"   Финальная апостериорная дисперсия b (a неизвестно): {posterior_vars_unknown[-1]:.2f}")
print(f"   Финальное апостериорное среднее b (a известно): {posterior_means_known[-1]:.2f}")
print(f"   Финальная апостериорная дисперсия b (a известно): {posterior_vars_known[-1]:.2f}")
print()

# 4
start_time = time.time()
posterior_full_unknown = prior_ab.copy()
for d_obs in d_sample_model4:
    lk_matrix = np.array([[likelihood_ab(a_val, b_val, d_obs) for b_val in b_values] for a_val in a_values])
    posterior_full_unknown *= lk_matrix
    posterior_full_unknown /= posterior_full_unknown.sum()
posterior_b_final = posterior_full_unknown.sum(axis=0)
posterior_b_final /= posterior_b_final.sum()
end_time = time.time()
time_full_unknown = end_time - start_time
print(f"4) Время оценки p(b|d1..dN) (a неизвестно): {time_full_unknown:.3f} сек")

start_time = time.time()
posterior_full_known = np.ones_like(b_values, dtype=float)
posterior_full_known /= posterior_full_known.sum()
for d_obs in d_sample_model4:
    lk_array = np.array([likelihood_ab(a_fixed, b_val, d_obs) for b_val in b_values])
    posterior_full_known *= lk_array
    posterior_full_known /= posterior_full_known.sum()
end_time = time.time()
time_full_known = end_time - start_time
print(f"   Время оценки p(b|d1..dN) (a известно): {time_full_known:.3f} сек")

start_time = time.time()
for i in range(min(10, N)):
    c_test = int(E_c_n)
    d_test = int(E_d_n)

    p_c = stats.poisson.pmf(c_test, a_fixed*p1 + b_fixed*p2)
    
    p_d = stats.poisson.pmf(d_test, (a_fixed*p1 + b_fixed*p2)*(1+p3))
end_time = time.time()
time_single_eval = (end_time - start_time) / min(10, N) if N > 0 else 0
print(f"   Время оценки одной пары p(c_n), p(d_n): {time_single_eval:.6f} сек (на 10 точках)")
print(f"   Оценка времени для N оценок: {time_single_eval * N:.3f} сек")
print()


# 5
def generate_model3(N, a, b, params):

    p1 = params['p1']
    p2 = params['p2']
    p3 = params['p3']
    rng = np.random.default_rng(42)

    lambda_c = a * p1 + b * p2
    lambda_d = lambda_c * (1 + p3)
    d_samples = rng.poisson(lam=lambda_d, size=N)

    return d_samples

d_sample_model3 = generate_model3(N, a_fixed, b_fixed, params_model4)

mean_d_model4 = d_sample_model4.mean()
var_d_model4 = d_sample_model4.var() 
mean_d_model3 = d_sample_model3.mean()
var_d_model3 = d_sample_model3.var()

print("5) Сравнение моделей 3 и 4:")
print(f"   Mean(d_n) model4: {mean_d_model4:.2f}, Var(d_n) model4: {var_d_model4:.2f}")
print(f"   Mean(d_n) model3: {mean_d_model3:.2f}, Var(d_n) model3: {var_d_model3:.2f}")
