import numpy as np
from matplotlib.path import Path
from scipy.spatial import Voronoi
import heapq

# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------
PASSO_AMOSTRA = 10        # distância (px) entre sites ao longo do contorno



def regular_polygon(center, n, radius, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation

    return np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])


def point_in_polygon(pt, poly):
    return Path(poly).contains_point(pt)


def point_in_any_polygon(pt, obstacles):
    return any(
        point_in_polygon(pt, poly)
        for poly in obstacles
    )


def dist_point_polygon(pt, poly):
    # Distância mínima do ponto ao polígono.
    # Retorna 0 se o ponto estiver dentro do polígono.
    if point_in_polygon(pt, poly):
        return 0.0

    dmin = np.inf
    n = len(poly)

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]

        d = dist_point_segment(pt, a, b)
        dmin = min(dmin, d)

    return dmin


def segment_inside_obstacle(a, b, obstaculos):
    meio = (np.asarray(a) + np.asarray(b)) / 2

    return point_in_any_polygon(meio, obstaculos)


def dist_point_segment(p, a, b):
    # Vetor que representa o segmento
    ab = b - a

    # Projeção do ponto sobre o segmento
    t = np.clip(
        np.dot(p - a, ab) /
        (np.dot(ab, ab) + 1e-9),
        0,
        1
    )

    proj = a + t * ab

    # Distância entre o ponto e sua projeção
    return np.linalg.norm(p - proj)


def inside_bounds(p, limites):
    xmin, ymin, xmax, ymax = limites
    x, y = p

    return xmin <= x <= xmax and ymin <= y <= ymax



def segment_intersects_obstacles(a, b, obstaculos):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    def orient(p, q, r):
        return (
            (q[0] - p[0]) * (r[1] - p[1])
            - (q[1] - p[1]) * (r[0] - p[0])
        )

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )

    def intersectam(a, b, c, d):
        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)

        if o1 * o2 < 0 and o3 * o4 < 0:
            return True

        if abs(o1) < 1e-9 and on_segment(a, c, b):
            return True
        if abs(o2) < 1e-9 and on_segment(a, d, b):
            return True
        if abs(o3) < 1e-9 and on_segment(c, a, d):
            return True
        if abs(o4) < 1e-9 and on_segment(c, b, d):
            return True

        return False

    for poligono in obstaculos:
        poligono = np.asarray(poligono, dtype=float)

        for i in range(len(poligono)):
            c = poligono[i]
            d = poligono[(i + 1) % len(poligono)]

            if intersectam(a, b, c, d):
                return True

    return False


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


def build_voronoi_roadmap(obstaculos, limites):
    if not obstaculos:
        return []

    # Sites = pontos amostrados no contorno de cada obstáculo,
    # guardando de qual obstáculo cada ponto veio.
    amostras = [amostrar_poligono(poly) for poly in obstaculos]

    pontos = np.vstack(amostras)
    ids = np.concatenate([
        np.full(len(pts), i) for i, pts in enumerate(amostras)
    ])

    xmin, ymin, xmax, ymax = limites
    espelhado = []

    for x, y in pontos:
        espelhado += [(2*xmin - x, y), (2*xmax - x, y),
                      (x, 2*ymin - y), (x, 2*ymax - y)]

    # Espelhos só servem para fechar as células; id -1 = "não é obstáculo real"
    ids_esp = np.full(len(espelhado), -1)

    vor = Voronoi(np.vstack([pontos, espelhado]))
    todos_ids = np.concatenate([ids, ids_esp])

    bordas = []

    # ridge_points informa quais dois sites geram cada aresta
    for (s1, s2), (p1, p2) in zip(vor.ridge_points, vor.ridge_vertices):
        if p1 == -1 or p2 == -1:
            continue

        # Descarta arestas que envolvem espelhos (ficam sobre a borda da imagem)
        if todos_ids[s1] < 0 or todos_ids[s2] < 0:
            continue

        # Descarta arestas entre pontos do MESMO obstáculo
        if todos_ids[s1] == todos_ids[s2]:
            continue

        a, b = vor.vertices[p1], vor.vertices[p2]

        # Verificar os limites da imagem
        if not inside_bounds(a, limites):
            continue

        if not inside_bounds(b, limites):
            continue

        # Verificar se os extremos estão dentro dos obstáculos
        if point_in_any_polygon(a, obstaculos):
            continue

        if point_in_any_polygon(b, obstaculos):
            continue

        # Verificar se o segmento cruza obstáculos
        if segment_intersects_obstacles(a, b, obstaculos):
            continue

        # Verificar o ponto médio
        if segment_inside_obstacle(a, b, obstaculos):
            continue

        bordas.append((a, b))

    return bordas


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