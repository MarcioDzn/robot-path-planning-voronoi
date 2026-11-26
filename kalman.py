"""EKF para robô diferencial observado por uma câmera fixa no teto.

Tudo em UMA classe (EKFCamera), UM arquivo, só depende de NumPy.

Estado interno : mu = [x, y, theta]   (metros, metros, RADIANOS)
Controle       : u  = (v, w)          (m/s, rad/s)  -> odometria / comando
Medida (câmera): z  = [x, y, theta]   ou [x, y]     (metros; theta em graus
                                                    se degrees=True)

Uso típico:
    ekf = EKFCamera(alpha=[0.05, 0.01, 0.01, 0.05],
                    sigma_x=0.02, sigma_y=0.02, sigma_theta=np.deg2rad(3))

    pose = ekf.step(u=(v, w), z=[x_cam, y_cam, th_cam_deg], dt=0.05)
    pose = ekf.step(u=(v, w), z=None, dt=0.05)    # frame sem detecção
    print(pose["x"], pose["y"], pose["th_deg"])
"""
import time
import numpy as np


def wrap_angle(a):
    """Normaliza um ângulo para [-pi, pi)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKFCamera:
    def __init__(self, alpha, sigma_x, sigma_y, sigma_theta=None,
                 degrees=True, homography=None, eps=1e-4):
        """
        alpha       : (a1, a2, a3, a4) ruído do modelo de movimento
        sigma_x/y   : ruído da câmera em metros (> 0)
        sigma_theta : ruído do ângulo da câmera em RADIANOS (> 0);
                      None se a câmera só fornece (x, y)
        degrees     : True se o theta que vem da câmera está em graus
        homography  : matriz 3x3 pixel -> metros (ex.: de cv2.findHomography),
                      opcional, só para usar pixel_to_world()
        eps         : se |w| < eps usa o modelo de linha reta (evita divisão por 0)
        """
        self.alpha = np.asarray(alpha, dtype=float)
        self.degrees = degrees
        self.eps = eps
        self.homography = None if homography is None else np.asarray(homography, float)

        self.use_heading = sigma_theta is not None
        if self.use_heading:
            self.H = np.eye(3)
            self.R = np.diag([sigma_x**2, sigma_y**2, sigma_theta**2])
        else:
            self.H = np.eye(3)[:2]
            self.R = np.diag([sigma_x**2, sigma_y**2])

        self.mu = None            # estado (None até inicializar)
        self.Sigma = None         # covariância
        self.likelihood = 1.0     # p(z) da última correção
        self._t_last = None

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def reset(self, x, y, theta, P0=None):
        """Inicializa o estado. theta em graus se degrees=True, senão rad."""
        th = np.deg2rad(theta) if self.degrees else theta
        self.mu = np.array([x, y, wrap_angle(th)], dtype=float)
        self.Sigma = (np.diag([0.05**2, 0.05**2, np.deg2rad(5)**2])
                      if P0 is None else np.asarray(P0, dtype=float))
        self._t_last = None

    @property
    def pose(self):
        x, y, th = self.mu
        return {"x": float(x), "y": float(y),
                "th_rad": float(th), "th_deg": float(np.degrees(th))}

    def pixel_to_world(self, px, py):
        """Converte pixel -> metros usando a homografia informada."""
        if self.homography is None:
            raise ValueError("Passe 'homography' no construtor para usar isto.")
        p = self.homography @ np.array([px, py, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def _parse_z(self, z):
        z = np.array(z, dtype=float)
        if self.use_heading:
            if self.degrees:
                z[2] = np.deg2rad(z[2])
            z[2] = wrap_angle(z[2])
        return z

    # ------------------------------------------------------------------
    # PREDIÇÃO (linhas 2-7 do algoritmo)
    # ------------------------------------------------------------------
    def predict(self, u, dt):
        v, w = u
        mu, Sigma = self.mu, self.Sigma
        theta = mu[2]
        a1, a2, a3, a4 = self.alpha

        if abs(w) > self.eps:
            s0, c0 = np.sin(theta), np.cos(theta)
            s1, c1 = np.sin(theta + w * dt), np.cos(theta + w * dt)
            r = v / w

            # linha 6: deslocamento esperado
            dx = -r * s0 + r * s1
            dy = r * c0 - r * c1

            # linha 4: Jacobiano em relação ao controle
            V = np.array([
                [(-s0 + s1) / w, v * (s0 - s1) / w**2 + v * c1 * dt / w],
                [(c0 - c1) / w, -v * (c0 - c1) / w**2 + v * s1 * dt / w],
                [0.0, dt],
            ])
        else:
            # limite w -> 0 (linha reta)
            dx = v * dt * np.cos(theta)
            dy = v * dt * np.sin(theta)
            V = np.array([
                [dt * np.cos(theta), -0.5 * v * dt**2 * np.sin(theta)],
                [dt * np.sin(theta), 0.5 * v * dt**2 * np.cos(theta)],
                [0.0, dt],
            ])

        # linha 3: Jacobiano em relação ao estado (G13 = -dy, G23 = dx)
        G = np.array([
            [1.0, 0.0, -dy],
            [0.0, 1.0, dx],
            [0.0, 0.0, 1.0],
        ])

        # linha 5: ruído no espaço de controle
        M = np.diag([a1 * v**2 + a2 * w**2,
                     a3 * v**2 + a4 * w**2])

        # linhas 6 e 7
        mu_bar = mu + np.array([dx, dy, w * dt])
        mu_bar[2] = wrap_angle(mu_bar[2])
        self.mu = mu_bar
        self.Sigma = G @ Sigma @ G.T + V @ M @ V.T

    # ------------------------------------------------------------------
    # CORREÇÃO com a medida da câmera (H constante)
    # ------------------------------------------------------------------
    def correct(self, z):
        H, R = self.H, self.R
        z = self._parse_z(z)

        innov = z - H @ self.mu
        if self.use_heading:
            innov[2] = wrap_angle(innov[2])

        S = H @ self.Sigma @ H.T + R
        S_inv = np.linalg.inv(S)
        K = self.Sigma @ H.T @ S_inv

        mu = self.mu + K @ innov
        mu[2] = wrap_angle(mu[2])
        self.mu = mu
        self.Sigma = (np.eye(3) - K @ H) @ self.Sigma

        self.likelihood = (np.linalg.det(2 * np.pi * S) ** -0.5
                           * np.exp(-0.5 * innov @ S_inv @ innov))

    # ------------------------------------------------------------------
    # Um ciclo completo: predição + correção (se houver medida)
    # ------------------------------------------------------------------
    def step(self, u, z=None, dt=None):
        """
        u  : (v, w)
        z  : medida da câmera ou None (frame sem detecção)
        dt : segundos desde o último step; se None, mede com o relógio
        Retorna a pose estimada (dict) ou None se ainda não inicializou.
        """
        now = time.perf_counter()
        if dt is None:
            dt = 0.0 if self._t_last is None else now - self._t_last
        self._t_last = now

        # primeira medida inicializa o filtro
        if self.mu is None:
            if z is None:
                return None
            z = np.asarray(z, dtype=float)
            self.reset(z[0], z[1], z[2] if self.use_heading else 0.0)
            return self.pose

        self.predict(u, dt)
        if z is not None:
            self.correct(z)
        else:
            self.likelihood = 1.0
        return self.pose