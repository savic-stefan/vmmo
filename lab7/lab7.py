import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from scipy.special import digamma, gammaln, logsumexp
import warnings
warnings.filterwarnings('ignore')

class DPMixture:
    def __init__(self, X, alpha, a, b, T=None, const_T=10):
        self.X = X.astype(int)
        self.N, self.D = self.X.shape
        self.alpha = float(alpha)
        self.a0 = float(a)
        self.b0 = float(b)

        if T is None:
            self.T = int(max(10, const_T * np.log(1 + self.N / max(1e-12, self.alpha))))
        else:
            self.T = int(T)

        self.gamma1 = np.ones(self.T)
        self.gamma2 = np.ones(self.T) * self.alpha

        self.a = np.ones((self.T, self.D)) * self.a0
        self.b = np.ones((self.T, self.D)) * self.b0
        
        self.resp = np.full((self.N, self.T), 1.0 / self.T)

        self.elbo_trace = []

    def _expect_log_sticks(self):
        e1 = digamma(self.gamma1) - digamma(self.gamma1 + self.gamma2)  # E[log v_k]
        e2 = digamma(self.gamma2) - digamma(self.gamma1 + self.gamma2)  # E[log(1-v_k)]
        E_log_pi = np.zeros(self.T)
        for k in range(self.T):
            if k == 0:
                E_log_pi[k] = e1[k]
            else:
                E_log_pi[k] = e1[k] + np.sum(e2[:k])
        return E_log_pi

    def _expect_log_theta(self):
        e_t = digamma(self.a) - digamma(self.a + self.b)  # E[log θ_ki]
        e_1t = digamma(self.b) - digamma(self.a + self.b)  # E[log(1-θ_ki)]
        return e_t, e_1t

    def _update_resp(self):
        E_log_pi = self._expect_log_sticks()
        e_t, e_1t = self._expect_log_theta()

        log_lik = self.X @ (e_t - e_1t).T + np.sum(e_1t, axis=1)
        log_joint = log_lik + E_log_pi[np.newaxis, :]

        self.resp = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))

    def _update_theta(self):
        Nk = np.sum(self.resp, axis=0)
        weighted_x = self.resp.T @ self.X
        self.a = self.a0 + weighted_x
        self.b = self.b0 + (Nk[:, None] - weighted_x)

    def _update_sticks(self):
        Nk = np.sum(self.resp, axis=0)
        suf = np.cumsum(Nk[::-1])[::-1]
        self.gamma1 = 1 + Nk
        self.gamma2 = self.alpha + (suf - Nk)

    def _compute_elbo(self):
        e_t, e_1t = self._expect_log_theta()
        E_log_pi = self._expect_log_sticks()

        term_X = np.sum(self.resp * (self.X @ e_t.T + (1 - self.X) @ e_1t.T))

        term_Z = np.sum(self.resp * E_log_pi)

        H_z = -np.sum(self.resp * np.log(self.resp + 1e-12))

        elbo = term_X + term_Z + H_z
        return elbo

    def fit_variational(self, num_starts=3, max_iter=200, tol=1e-5, display=True):
        best_elbo = -np.inf
        best_state = None

        for s in range(num_starts):
            self.resp = np.random.dirichlet(np.ones(self.T), size=self.N)
            prev_elbo = -np.inf
            self.elbo_trace = []

            for it in range(1, max_iter + 1):
                self._update_theta()
                self._update_sticks()
                self._update_resp()

                elbo = self._compute_elbo()
                self.elbo_trace.append(elbo)

                Nk = np.sum(self.resp, axis=0)
                active = np.sum(Nk > 1e-2)

                if display:
                    print(f"start {s+1} | iter {it} | ELBO={elbo:.3f} | active={active}")

                if it > 1 and abs(elbo - prev_elbo) / (abs(prev_elbo) + 1e-12) < tol:
                    if display:
                        print(f"  Сходимость достигнута на итерации {it}.")
                    break

                prev_elbo = elbo

            if elbo > best_elbo:
                best_elbo = elbo
                best_state = {
                    'resp': self.resp.copy(),
                    'gamma1': self.gamma1.copy(),
                    'gamma2': self.gamma2.copy(),
                    'a': self.a.copy(),
                    'b': self.b.copy(),
                    'elbo_trace': self.elbo_trace.copy()
                }
                if display:
                    print(f"  Новый лучший ELBO: {elbo:.3f}")

        if best_state:
            self.resp = best_state['resp']
            self.gamma1 = best_state['gamma1']
            self.gamma2 = best_state['gamma2']
            self.a = best_state['a']
            self.b = best_state['b']
            self.elbo_trace = best_state['elbo_trace']

        return self

    def show_clusters(self, min_weight=1.0, figsize=(10, 6)):
        Nk = np.sum(self.resp, axis=0)
        idx = np.where(Nk >= min_weight)[0]

        if idx.size == 0:
            print("Нет кластеров с весом >= min_weight")
            return

        centers = self.a / (self.a + self.b)

        cols = min(10, idx.size)
        rows = int(np.ceil(idx.size / cols))

        plt.figure(figsize=figsize)
        for i, k in enumerate(idx):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(centers[k].reshape(8, 8), cmap='gray', vmin=0, vmax=1)
            plt.title(f'k={k}\nNk={Nk[k]:.1f}')
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def add_sample(self, X_new):
        X_new = np.asarray(X_new, dtype=int)
        self.X = np.vstack([self.X, X_new])
        self.N = self.X.shape[0]

        new_resp = np.full((X_new.shape[0], self.T), 1.0 / self.T)
        self.resp = np.vstack([self.resp, new_resp])

        return self

    def predict(self, X_new):
        Xn = np.asarray(X_new, dtype=int)
        E_log_pi = self._expect_log_sticks()
        e_t, e_1t = self._expect_log_theta()

        log_lik = Xn @ (e_t - e_1t).T + np.sum(e_1t, axis=1)
        log_joint = log_lik + E_log_pi
        resp_new = np.exp(log_joint - logsumexp(log_joint, axis=1, keepdims=True))
        return np.argmax(resp_new, axis=1)

print("\n №1")
print("Модель DPMixture создана")

print("\n №2")
print("Выполнение вариационного вывода и расчёт ELBO")
digits = load_digits()
X_bin = (digits.data > 8).astype(int)
dp = DPMixture(X_bin[:300], alpha=1.0, a=1.0, b=1.0, T=30)
dp.fit_variational(num_starts=2, max_iter=20)

print("\n №3")
print("Визуализация кластеров")
dp.show_clusters(min_weight=5)

print("\n №4")
print("Запуск алгоритма на полной выборке и обучение классификатора")
X_full_bin = X_bin
dp_full = DPMixture(X_full_bin, alpha=1.0, a=1.0, b=1.0, T=40)
dp_full.fit_variational(num_starts=2, max_iter=30)
features = dp_full.resp

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

clf = LogisticRegression(max_iter=1000)
clf.fit(features, digits.target)
y_pred = clf.predict(features)
print("Accuracy:", accuracy_score(digits.target, y_pred))
print("Confusion matrix:\n", confusion_matrix(digits.target, y_pred))

print("\n № 5")
print("Добавление новых данных и обновление кластеров")
new_data = X_full_bin[1000:1100]
dp_full.add_sample(new_data)
dp_full.fit_variational(num_starts=1, max_iter=10, display=False)
dp_full.show_clusters(min_weight=10)

print("\n № 6")
print("Построение графика изменения ELBO")
plt.figure(figsize=(6, 4))
plt.plot(dp_full.elbo_trace)
plt.title("Изменение вариационной нижней оценки ELBO")
plt.xlabel("итерации")
plt.ylabel("ELBO")
plt.grid(True)
plt.show()
