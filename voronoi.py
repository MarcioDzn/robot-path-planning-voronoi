import numpy as np
import cv2
from matplotlib.path import Path
from scipy.spatial import Voronoi
import heapq

# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------
PASSO_AMOSTRA = 10        # distância (px) entre sites ao longo do contorno


def construir_obstaculos_dilatados(mascara_obstaculos, mascara_arena, raio_px):
    raio = int(np.ceil(raio_px))
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2*raio + 1, 2*raio + 1)
    )

    # Obstáculos internos crescem
    obst = cv2.dilate(mascara_obstaculos, k)

    # Arena encolhe (o robô não encosta na parede)
    arena_livre = cv2.erode(mascara_arena, k)

    # Proibido = obstáculo dilatado OU fora da arena encolhida
    proibido = cv2.bitwise_or(obst, cv2.bitwise_not(arena_livre))
    return proibido

def extrair_contornos_sites(proibido):
    h, w = proibido.shape
    contornos, _ = cv2.findContours(
        proibido, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    resultado = []
    for c in contornos:
        pts = c.reshape(-1, 2)

        # descarta o contorno que roda pela borda da imagem
        na_borda = (
            (pts[:, 0] == 0) | (pts[:, 0] == w - 1) |
            (pts[:, 1] == 0) | (pts[:, 1] == h - 1)
        )
        if na_borda.any():
            continue

        if len(pts) >= 3:
            resultado.append(pts.astype(float))
    return resultado

def segmento_livre(a, b, proibido, passo=2.0):
    n = max(2, int(np.hypot(*(b - a)) / passo))
    for t in np.linspace(0, 1, n):
        if not livre(a + t*(b - a), proibido):
            return False
    return True


def livre(p, proibido):
    x, y = int(round(p[0])), int(round(p[1]))
    h, w = proibido.shape
    return 0 <= x < w and 0 <= y < h and proibido[y, x] == 0

def inside_bounds(p, limites):
    xmin, ymin, xmax, ymax = limites
    x, y = p

    return xmin <= x <= xmax and ymin <= y <= ymax



def amostrar_poligono(poly, passo=PASSO_AMOSTRA):
    # Gera pontos ao longo de TODAS as arestas do polígono (não só os vértices),
    # para o Voronoi aproximar bem a distância às arestas retas.
    pts = []

    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]

        n = max(int(np.linalg.norm(b - a) / passo), 1)

        for t in np.linspace(0, 1, n, endpoint=False):
            pts.append(a + t * (b - a))

    return np.array(pts)


def build_voronoi_roadmap(obstaculos, limites, proibido):
    if not obstaculos:
        return []

    amostras = [amostrar_poligono(poly) for poly in obstaculos]

    pontos = np.vstack(amostras)
    ids = np.concatenate([
        np.full(len(pts), i) for i, pts in enumerate(amostras)
    ])

    # pos, perimetros = [], []
    # for pts in amostras:
    #     seg = np.linalg.norm(np.diff(pts, axis=0, append=pts[:1]), axis=1)
    #     pos.append(np.concatenate([[0.0], np.cumsum(seg)[:-1]]))
    #     perimetros.append(seg.sum())
    # pos = np.concatenate(pos)
    # perimetros = np.array(perimetros)

    xmin, ymin, xmax, ymax = limites
    espelhado = []
    for x, y in pontos:
        espelhado += [(2*xmin - x, y), (2*xmax - x, y),
                      (x, 2*ymin - y), (x, 2*ymax - y)]

    ids_esp = np.full(len(espelhado), -1)

    vor = Voronoi(np.vstack([pontos, espelhado]))
    todos_ids = np.concatenate([ids, ids_esp])

    bordas = []

    for (s1, s2), (p1, p2) in zip(vor.ridge_points, vor.ridge_vertices):
        if p1 == -1 or p2 == -1:
            continue
        if todos_ids[s1] < 0 or todos_ids[s2] < 0:
            continue
        if todos_ids[s1] == todos_ids[s2]:
            continue
            # i = todos_ids[s1]
            # d_arco = abs(pos[s1] - pos[s2])
            # d_arco = min(d_arco, perimetros[i] - d_arco)   # contorno é fechado
            # d_euc = np.linalg.norm(pontos[s1] - pontos[s2])
            # if d_arco < FATOR_ARCO * d_euc:
            #     continue

        a, b = vor.vertices[p1], vor.vertices[p2]

        if not inside_bounds(a, limites) or not inside_bounds(b, limites):
            continue

        # Substitui os 4 testes antigos por um só
        if not segmento_livre(a, b, proibido):
            continue

        bordas.append((a, b))

    return bordas

def conectar_ao_roadmap(ponto, bordas, proibido, k=8):
    """Cria arestas do ponto até os k vértices do roadmap mais próximos
    que sejam alcançáveis em linha reta sem tocar região proibida."""
    if not bordas:
        return []

    ponto = np.asarray(ponto, float)
    vertices = np.unique(
        np.array([v for a, b in bordas for v in (tuple(a), tuple(b))]),
        axis=0
    )

    dist = np.linalg.norm(vertices - ponto, axis=1)
    novas = []
    for idx in np.argsort(dist)[:k]:
        v = vertices[idx]
        if segmento_livre(ponto, v, proibido):
            novas.append((ponto, v))
    return novas

def dijkstra_on_edges(edges, start, goal):
    graph = {}

    # Construir o grafo
    for a, b in edges:
        a, b = tuple(a), tuple(b)
        dist = np.linalg.norm(np.array(a) - np.array(b))

        graph.setdefault(a, []).append((b, dist))
        graph.setdefault(b, []).append((a, dist))

    if not graph:
        return []

    start = tuple(start)
    goal = tuple(goal)

    # Encontrar os nós mais próximos do início e do objetivo
    nos = list(graph.keys())

    no_start = min(nos, key=lambda p: np.linalg.norm(np.array(p) - start))
    no_goal = min(nos, key=lambda p: np.linalg.norm(np.array(p) - goal))

    # Conectar início e objetivo ao grafo
    graph.setdefault(start, []).append(
        (no_start, np.linalg.norm(np.array(start) - no_start))
    )
    graph.setdefault(no_start, []).append(
        (start, np.linalg.norm(np.array(start) - no_start))
    )

    graph.setdefault(goal, []).append(
        (no_goal, np.linalg.norm(np.array(goal) - no_goal))
    )
    graph.setdefault(no_goal, []).append(
        (goal, np.linalg.norm(np.array(goal) - no_goal))
    )

    # Dijkstra
    fila = [(0, start, [start])]
    visitados = set()

    while fila:
        custo, atual, caminho = heapq.heappop(fila)

        if atual in visitados:
            continue

        visitados.add(atual)

        if atual == goal:
            return caminho

        for vizinho, peso in graph.get(atual, []):
            if vizinho not in visitados:
                heapq.heappush(
                    fila,
                    (custo + peso, vizinho, caminho + [vizinho])
                )

    return []