from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Calibracao:
    pixels_por_cm: float
    angulo_rad: float
    quadrados_detectados: int

    def pixel_para_cm(self, x, y, origem_pixels):
        origem = np.asarray(origem_pixels, dtype=float)
        ponto = np.array([x, y]) - origem

        eixo_x = np.array([np.cos(self.angulo_rad), np.sin(self.angulo_rad)])
        eixo_y = np.array([-np.sin(self.angulo_rad), np.cos(self.angulo_rad)])

        return (
            float(np.dot(ponto, eixo_x) / self.pixels_por_cm),
            float(np.dot(ponto, eixo_y) / self.pixels_por_cm),
        )

    def cm_para_pixel(self, x_cm, y_cm, origem_pixels):
        origem = np.asarray(origem_pixels, dtype=float)
        eixo_x = np.array([np.cos(self.angulo_rad), np.sin(self.angulo_rad)])
        eixo_y = np.array([-np.sin(self.angulo_rad), np.cos(self.angulo_rad)])
        ponto = origem + self.pixels_por_cm * (x_cm * eixo_x + y_cm * eixo_y)
        return float(ponto[0]), float(ponto[1])


def _maior_agrupamento(pontos, distancia_maxima):
    nao_visitados = set(range(len(pontos)))
    agrupamentos = []

    while nao_visitados:
        fila = [nao_visitados.pop()]
        agrupamento = []

        while fila:
            atual = fila.pop()
            agrupamento.append(atual)

            vizinhos = [
                indice for indice in nao_visitados
                if np.linalg.norm(pontos[atual] - pontos[indice]) < distancia_maxima
            ]
            for indice in vizinhos:
                nao_visitados.remove(indice)
                fila.append(indice)

        agrupamentos.append(agrupamento)

    return max(agrupamentos, key=len)


def calibrar_xadrez(caminho, lado_quadrado_cm):
    imagem = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    if imagem is None:
        raise FileNotFoundError(f"Não foi possível abrir a imagem de calibração: {caminho}")
    if lado_quadrado_cm <= 0:
        raise ValueError("O lado do quadrado deve ser maior que zero.")

    altura, largura = imagem.shape
    lado_minimo = min(altura, largura)
    area_imagem = altura * largura

    _, mascara = cv2.threshold(imagem, 50, 255, cv2.THRESH_BINARY_INV)
    _, _, estatisticas, centros = cv2.connectedComponentsWithStats(mascara)

    candidatos = []
    tamanhos = []
    for indice, (_, _, w, h, area) in enumerate(estatisticas[1:], start=1):
        lado = (w + h) / 2
        proporcao = w / h if h else 0
        if (
            0.005 * lado_minimo <= lado <= 0.04 * lado_minimo
            and 0.00007 * area_imagem <= area <= 0.001 * area_imagem
            and 0.65 <= proporcao <= 1.55
        ):
            candidatos.append(centros[indice])
            tamanhos.append(lado)

    if len(candidatos) < 8:
        raise ValueError(
            "Não foram detectados quadrados suficientes na imagem de calibração."
        )

    candidatos = np.asarray(candidatos, dtype=float)
    lado_px_estimado = float(np.median(tamanhos))
    agrupamento = _maior_agrupamento(candidatos, lado_px_estimado * 2.7)
    centros_xadrez = candidatos[agrupamento]

    if len(centros_xadrez) < 8:
        raise ValueError(
            "O agrupamento detectado não contém quadrados suficientes para calibrar."
        )

    covariancia = np.cov(centros_xadrez.T)
    _, vetores = np.linalg.eigh(covariancia)
    eixo_x = vetores[:, -1]
    if eixo_x[0] < 0:
        eixo_x = -eixo_x
    angulo_rad = float(np.arctan2(eixo_x[1], eixo_x[0]))

    escala = lado_px_estimado / lado_quadrado_cm
    return Calibracao(
        pixels_por_cm=escala,
        angulo_rad=angulo_rad,
        quadrados_detectados=len(centros_xadrez),
    )


def desenhar_referencial(frame, calibracao, origem_pixels):
    origem = np.asarray(origem_pixels, dtype=float)
    eixo_x = np.array([np.cos(calibracao.angulo_rad), np.sin(calibracao.angulo_rad)])
    eixo_y = np.array([
        -np.sin(calibracao.angulo_rad), np.cos(calibracao.angulo_rad)
    ])
    altura, largura = frame.shape[:2]
    comprimento = max(largura, altura)
    ponto_origem = tuple(np.round(origem).astype(int))

    for eixo, cor, rotulo in (
        (eixo_x, (255, 255, 0), "+X"),
        (eixo_y, (255, 0, 255), "+Y"),
    ):
        inicio = tuple(np.round(origem - comprimento * eixo).astype(int))
        fim = tuple(np.round(origem + comprimento * eixo).astype(int))
        cv2.line(frame, inicio, fim, cor, 1)
        ponta = tuple(np.round(origem + comprimento * 0.4 * eixo).astype(int))
        cv2.arrowedLine(frame, ponto_origem, ponta, cor, 2, tipLength=0.03)
        cv2.putText(
            frame, rotulo, (ponta[0] + 8, ponta[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2
        )

    cv2.drawMarker(
        frame, ponto_origem, (0, 0, 255), cv2.MARKER_CROSS, 18, 2
    )
    cv2.putText(
        frame, "Origem (0, 0) cm",
        (ponto_origem[0] + 8, ponto_origem[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
    )
