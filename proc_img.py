import cv2
import numpy as np
from matplotlib.path import Path
from scipy.spatial import Voronoi
import heapq
from aux_class import Objeto


# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------
MIN_AREA = 100
EPSILON_FACTOR = 0.002
LIMITE_PRETO = 95



# ----------------------------------------------------------------------
# Detecção e classificação de formas
# ----------------------------------------------------------------------
def classificarForma(cnt, approx):
    vertices = len(approx)

    area = cv2.contourArea(cnt)
    peri = cv2.arcLength(cnt, True)

    # Circularidade
    circ = (4 * np.pi * area) / (peri ** 2) if peri > 0 else 0

    if vertices >= 8 and circ > 0.85:
        return "Circulo", (0, 255, 255)

    elif vertices == 3:
        return "Triangulo", (255, 0, 0)

    elif vertices == 4:
        # minAreaRect é invariante à rotação (boundingRect não é)
        (_, _), (w, h), _ = cv2.minAreaRect(approx)
        prop = w / h if h > 0 else 0

        if 0.95 <= prop <= 1.05:
            return "Quadrado", (0, 255, 0)
        else:
            return "Retangulo", (0, 165, 255)

    elif vertices == 5:
        return "Pentagono", (255, 0, 255)

    elif vertices == 6:
        return "Hexagono", (0, 0, 255)

    elif vertices == 7:
        return "Heptagono", (128, 0, 128)

    elif vertices == 8:
        return "Octogono", (255, 255, 0)

    else:
        return f"Poligono ({vertices})", (128, 128, 128)


def detectarFormas(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # --------------------------------------------------
    # 1. Identifica a região branca da arena
    # --------------------------------------------------
    _, mascara_arena = cv2.threshold(
        blur, LIMITE_PRETO, 255, cv2.THRESH_BINARY
    )

    kernel = np.ones((5, 5), np.uint8)

    mascara_arena = cv2.morphologyEx(
        mascara_arena, cv2.MORPH_CLOSE, kernel
    )

    # Encontra as regiões brancas
    contornos_arena, _ = cv2.findContours(
        mascara_arena,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contornos_arena:
        return [], np.zeros_like(gray)

    # A maior região branca é considerada a arena
    contorno_arena = max(
        contornos_arena, key=cv2.contourArea
    )

    mascara_arena = np.zeros_like(gray)

    cv2.drawContours(
        mascara_arena,
        [contorno_arena],
        -1,
        255,
        cv2.FILLED
    )

    # --------------------------------------------------
    # Obstáculo externo: tudo fora da arena
    # --------------------------------------------------
    mascara_exterior = cv2.bitwise_not(mascara_arena)

    
    # --------------------------------------------------
    # 2. Detecta obstáculos pretos dentro da arena
    # --------------------------------------------------
    _, mascara_pretos = cv2.threshold(
        blur, LIMITE_PRETO, 255, cv2.THRESH_BINARY_INV
    )

    mascara_obstaculos = cv2.bitwise_and(
        mascara_pretos, mascara_arena
    )

    mascara_obstaculos = cv2.bitwise_or(
        mascara_obstaculos,
        mascara_exterior
    )

    # Remove ruídos pequenos
    kernel_ruido = np.ones((3, 3), np.uint8)

    mascara_obstaculos = cv2.morphologyEx(
        mascara_obstaculos, cv2.MORPH_OPEN, kernel_ruido
    )

    mascara_obstaculos = cv2.morphologyEx(
        mascara_obstaculos, cv2.MORPH_CLOSE, kernel_ruido
    )

    # --------------------------------------------------
    # 3. Extrai e classifica os obstáculos
    # --------------------------------------------------
    contours, _ = cv2.findContours(
        mascara_obstaculos,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    objetos = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < MIN_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(
            cnt, EPSILON_FACTOR * peri, True
        )

        if len(approx) < 3:
            continue

        nome, cor = classificarForma(cnt, approx)

        M = cv2.moments(cnt)

        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        obj = Objeto(
            nome, [cx, cy], cnt, approx,
            len(approx), area, cor
        )

        objetos.append(obj)
    
    
    contorno_exterior = contorno_arena

    # Obstáculo correspondente ao exterior da arena
    M = cv2.moments(contorno_exterior)

    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        area_exterior = (
            frame.shape[0] * frame.shape[1]
            - cv2.contourArea(contorno_exterior)
        )

        obj_exterior = Objeto(
            "Exterior",
            [cx, cy],
            contorno_exterior,
            contorno_exterior,
            len(contorno_exterior),
            area_exterior,
            (0, 0, 255)
        )

        # objetos.append(obj_exterior)
    
    return objetos, mascara_obstaculos