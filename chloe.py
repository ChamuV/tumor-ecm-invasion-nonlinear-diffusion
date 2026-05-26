import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye, diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, interp1d
from scipy.optimize import root_scalar
from scipy.stats import linregress

class TumorECMModel1D_AB2AM2:
    def __init__(self, D=1.0, rho=1.0, K=1.0, k=1.0, m0 = 1, Mmax=1.0, perc=0.2, L=1000.0, N=2001, T=1000.0, dt=0.10):
        self.D = D
        self.rho = rho
        self.K = K
        self.k = k
        self.m0 = m0
        self.Mmax = Mmax

        self.perc = perc

        self.L = L
        self.N = N
        self.dx = L / (N - 1)
        self.x = np.linspace(0, L, N)

        self.T = T
        self.dt = dt
        self.Nt = int(T / dt)
        self.times = np.linspace(0, T, self.Nt)

        self.use_euler_for_M = True

        self.N_arr = np.zeros((self.Nt, self.N))  # Tumor density
        self.M_arr = np.zeros((self.Nt, self.N))  # ECM density

    def f(self, N):
        return self.rho * N * (1 - N / self.K)

    def g(self, N, M):
        return -self.k * M * N

    def update_laplacian(self, M):
        N = self.N
        lower = np.zeros(N)
        center = np.zeros(N)
        upper = np.zeros(N)
        for i in range(1, N - 1):
            a = 1 - M[i - 1] / self.Mmax
            c = 1 - M[i + 1] / self.Mmax
            b = - (a + c)
            lower[i] = a
            center[i] = b
            upper[i] = c
        c = 1 - M[1] / self.Mmax
        center[0] = -2 * c
        upper[0] = 2 * c
        a = 1 - M[-2] / self.Mmax
        center[-1] = -2 * a
        lower[-1] = 2 * a
        Lmat = diags([lower[1:], center, upper[:-1]], offsets=[-1, 0, 1], format='csr')
        return (self.D / self.dx**2) * Lmat

    def initial_condition(self):
        N0 = np.where(self.x < self.perc * self.L, 0.7, 0.0)
        M0 = self. m0 * self.Mmax * np.ones_like(self.x)
        return N0, M0

    def solve(self):
        N0, M0 = self.initial_condition()
        N_prev, M_prev = N0.copy(), M0.copy()
        f_prev = self.f(N_prev)
        g_prev = self.g(N_prev, M_prev)
        L_prev = self.update_laplacian(M_prev)
        I = eye(self.N, format='csr')

        N_curr = spsolve(I - self.dt * L_prev, N_prev + self.dt * f_prev)
        M_curr = M_prev + self.dt * g_prev

        self.N_arr[0], self.M_arr[0] = N_prev, M_prev
        self.N_arr[1], self.M_arr[1] = N_curr, M_curr

        for i in range(2, self.Nt):
            L_curr = self.update_laplacian(M_curr)
            f_curr = self.f(N_curr)
            g_curr = self.g(N_curr, M_curr)

            rhs_N = (I + 0.5 * self.dt * L_prev) @ N_curr + self.dt * (1.5 * f_curr - 0.5 * f_prev)
            N_next = spsolve(I - 0.5 * self.dt * L_curr, rhs_N)
            N_next[0], N_next[-1] = N_next[1], N_next[-2]  # Neumann BCs

            if self.use_euler_for_M:
                M_next = M_curr + self.dt * self.g(N_curr, M_curr)
            else:
                M_next = M_curr + self.dt * (1.5 * g_curr - 0.5 * g_prev)

            self.N_arr[i] = N_next
            self.M_arr[i] = M_next
            N_prev, N_curr = N_curr, N_next
            M_prev, M_curr = M_curr, M_next
            f_prev, g_prev = f_curr, g_curr
            L_prev = L_curr

    def plot_results(self, wave_skip=500, random_colors=True, legd=False):
        x = self.x
        t = self.times
        N_arr = self.N_arr
        M_arr = self.M_arr
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()
        time_indices = list(range(0, self.Nt, wave_skip))
        if time_indices[-1] != self.Nt - 1:
            time_indices.append(self.Nt - 1)
        if random_colors:
            np.random.seed(42)
            colors = np.random.rand(len(time_indices), 3)
        else:
            colors = plt.cm.tab10(np.linspace(0, 1, len(time_indices)))
        for i, idx in enumerate(time_indices):
            axs[0].plot(x, N_arr[idx], label=f't={t[idx]:.0f}', color=colors[i])
        axs[0].set_title('Tumor density N(x,t)')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('N')
        axs[0].grid(True)
        for i, idx in enumerate(time_indices):
            axs[1].plot(x, M_arr[idx], label=f't={t[idx]:.0f}', color=colors[i])
        axs[1].set_title('ECM density M(x,t)')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('M')
        axs[1].grid(True)
        if legd:
            handles, labels = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=min(5, len(labels)), frameon=False)
        im0 = axs[2].imshow(N_arr, aspect='auto', extent=[x[0], x[-1], t[-1], t[0]], cmap='viridis')
        axs[2].set_title('Tumor N(x,t) heatmap')
        axs[2].set_xlabel('x')
        axs[2].set_ylabel('t')
        fig.colorbar(im0, ax=axs[2])
        im1 = axs[3].imshow(M_arr, aspect='auto', extent=[x[0], x[-1], t[-1], t[0]], cmap='plasma')
        axs[3].set_title('ECM M(x,t) heatmap')
        axs[3].set_xlabel('x')
        axs[3].set_ylabel('t')
        fig.colorbar(im1, ax=axs[3])
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    def estimate_wave_speed(self, t_start=10.0, t_end=200.0, num_points=30, threshold=0.5,
                            spline_type='cubic', legend_loc='bottom', plot=True,
                            plot_front_vs_time=False, target='N'):
        x = self.x
        t_vec = self.times
        u_arr = self.N_arr if target.lower() == 'n' else self.M_arr
        label_var = 'Tumor N(x,t)' if target.lower() == 'n' else 'ECM M(x,t)'
        t_list = np.linspace(t_start, t_end, num_points)
        x_fronts = []
        t_fronts = []

        if plot:
            plt.figure(figsize=(12, 6))

        def get_spline(method, x, y):
            if method == 'cubic':
                return CubicSpline(x, y)
            elif method == 'pchip':
                return PchipInterpolator(x, y)
            elif method == 'akima':
                return Akima1DInterpolator(x, y)
            elif method == 'linear':
                return interp1d(x, y, kind='linear', fill_value="extrapolate")
            else:
                raise ValueError(f"Unsupported spline_type: {method}")

        for t_target in t_list:
            idx = np.argmin(np.abs(t_vec - t_target))
            u = u_arr[idx]
            if np.all(u > threshold) or np.all(u < threshold):
                print(f"❌ No crossing at t = {t_target:.2f}")
                continue
            spline = get_spline(spline_type, x, u)
            u_spline = spline(x)
            crossing_idx = np.where(np.sign(u_spline[:-1] - threshold) != np.sign(u_spline[1:] - threshold))[0]
            if len(crossing_idx) == 0:
                print(f"❌ No spline crossing at t = {t_target:.2f}")
                continue
            i = crossing_idx[0]
            x_left, x_right = x[i], x[i + 1]
            def root_func(x_val): return spline(x_val) - threshold
            try:
                sol = root_scalar(root_func, bracket=[x_left, x_right])
                if sol.converged:
                    x_star = sol.root
                    x_fronts.append(x_star)
                    t_fronts.append(t_target)
                    print(f"✅ t = {t_target:.2f} → x ≈ {x_star:.4f}")
                    if plot:
                        plt.plot(x, u, label=f't = {t_target:.1f}')
                        plt.plot(x_star, threshold, 'ro')
            except Exception as e:
                print(f"❌ Error at t = {t_target:.2f}: {e}")

        if plot:
            plt.axhline(threshold, color='gray', linestyle='--', label=f'u = {threshold}')
            plt.xlabel("x")
            plt.ylabel(f"{label_var}")
            plt.title(f"{label_var} Wavefront Tracking")
            if legend_loc == 'bottom':
                plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=5)
            else:
                plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        actual_speed = 2 * np.sqrt(self.rho * self.D * (1 - self.m0))

        if len(t_fronts) >= 2:
            slope, intercept, r_value, _, _ = linregress(t_fronts, x_fronts)

            print("\n--- Wave Speed Estimate ---")
            print(f"Estimated speed = {slope:.4f}")
            print(f"Intercept = {intercept:.4f}")
            print(f"R² = {r_value**2:.4f}")
            print(f"Analytical Speed = {actual_speed:.4f}")
            print(f"Error = {slope - actual_speed:.4f}")

            if plot and plot_front_vs_time:
                plt.figure(figsize=(8, 4))
                plt.plot(t_fronts, x_fronts, 'o', label="Fronts")
                plt.plot(t_fronts, slope * np.array(t_fronts) + intercept, 'r--',
                         label=f"x = {slope:.2f}t + {intercept:.1f}")
                plt.xlabel("Time t")
                plt.ylabel("Front position x(t)")
                plt.title("Estimated Wave Speed")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            return slope, intercept, r_value**2
        else:
            print("❌ Not enough valid front points.")
            return None, None, None
        
    def compute_mass(self, target='N', return_time_series=True, plot=True):
        if target.upper() == 'N':
            arr = self.N_arr
            label = "Tumor mass ∫N(x,t) dx"
        elif target.upper() == 'M':
            arr = self.M_arr
            label = "ECM mass ∫M(x,t) dx"
        else:
            raise ValueError("target must be 'N' or 'M'")
    
        dx = self.dx
        mass = np.trapz(arr, dx=dx, axis=1)  # shape (Nt,)
        t_vec = self.times
    
        if plot:
            plt.figure(figsize=(8, 4))
            plt.plot(t_vec, mass, label=label)
            plt.xlabel("Time t")
            plt.ylabel("Mass")
            plt.title(f"Mass of {target.upper()} over time")
            plt.grid(True)
            plt.tight_layout()
            plt.legend()
            plt.show()
    
        if return_time_series:
            return t_vec, mass