import cv2
import numpy as np
from aux_class import Objeto
from calibracao import Calibracao, calibrar_xadrez, desenhar_referencial
from proc_img import detectarFormas, detectarMarcadoresVermelhos
from voronoi import (
    build_voronoi_roadmap,
    dilatar_obstaculos,
    dijkstra_on_edges,
    point_in_any_polygon,
)


# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------
IMAGEM_CALIBRACAO = "novo/xadrez2.png"
IMAGEM_PROCESSAMENTO = "novo/mesa_ponto_vermelho3.png"
LADO_QUADRADO_CM = 2.35
ORIGEM_PIXELS = (210, 500)
INICIO_CM = (-30.0, 23.0)
OBJETIVO_CM = (30.0, 86.0)
RAIO_ROBO_CM = 12.5


def desenharResultados(
    frame, objetos: list[Objeto], calibracao: Calibracao
):
    for obj in objetos:
        cx, cy = obj.centroide

        cv2.drawContours(frame, [obj.contorno], -1, (128, 128, 128), 1)
        cv2.drawContours(frame, [obj.poligono], -1, obj.cor, 3)
        cv2.circle(frame, (cx, cy), 4, obj.cor, -1)

        if obj.area >= 1000:
            x_cm, y_cm = calibracao.pixel_para_cm(
                cx, cy, ORIGEM_PIXELS
            )
            rotulo = f"{obj.nome} ({x_cm:.1f}, {y_cm:.1f}) cm"
        else:
            rotulo = f"{obj.nome} ({obj.vertices}v)"
        cv2.putText(
            frame, rotulo, (cx - 30, cy - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

    return frame


def main():
    calibracao = calibrar_xadrez(IMAGEM_CALIBRACAO, LADO_QUADRADO_CM)
    print(
        f"Calibração: {calibracao.pixels_por_cm:.2f} px/cm; "
        f"ângulo dos eixos: {np.degrees(calibracao.angulo_rad):.2f}°; "
        f"quadrados detectados: {calibracao.quadrados_detectados}"
    )

    frame = cv2.imread(IMAGEM_PROCESSAMENTO)
    if frame is None:
        raise FileNotFoundError(
            f"Não foi possível abrir a imagem de processamento: {IMAGEM_PROCESSAMENTO}"
        )

    centros_vermelhos, mascara_vermelha = detectarMarcadoresVermelhos(frame)
    objetos, mascara = detectarFormas(frame, mascara_vermelha)
    obstaculos = [
        obj.poligono.reshape(-1, 2).astype(float)
        for obj in objetos
    ]

    altura, largura = frame.shape[:2]
    limites = (0, 0, largura - 1, altura - 1)
    raio_robo_px = RAIO_ROBO_CM * calibracao.pixels_por_cm
    obstaculos_dilatados = dilatar_obstaculos(
        obstaculos, raio_robo_px, (largura, altura)
    )
    bordas = build_voronoi_roadmap(obstaculos_dilatados, limites)

    for a, b in bordas:
        p1 = tuple(np.round(a).astype(int))
        p2 = tuple(np.round(b).astype(int))
        cv2.line(frame, p1, p2, (0, 255, 0), 2)

    for poligono in obstaculos_dilatados:
        contorno = np.round(poligono).astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(frame, [contorno], -1, (255, 0, 255), 1)

    if centros_vermelhos:
        inicio_px = centros_vermelhos[0]
        print(f"Início da trajetória definido pelo marcador: {inicio_px} px")
    else:
        inicio_px = calibracao.cm_para_pixel(*INICIO_CM, ORIGEM_PIXELS)
        print(f"Sem marcador; usando INICIO_CM como início: {INICIO_CM} cm")

    objetivo_px = calibracao.cm_para_pixel(*OBJETIVO_CM, ORIGEM_PIXELS)

    # if (
    #     point_in_any_polygon(inicio_px, obstaculos_dilatados)
    #     or point_in_any_polygon(objetivo_px, obstaculos_dilatados)
    # ):
    #     raise ValueError("Início ou objetivo está dentro da margem de segurança.")

    caminho_px = dijkstra_on_edges(bordas, inicio_px, objetivo_px)
    caminho_cm = np.array([
        calibracao.pixel_para_cm(x, y, ORIGEM_PIXELS)
        for x, y in caminho_px
    ], dtype=float).reshape(-1, 2)

    print(caminho_cm)

    for i in range(len(caminho_cm) - 1):
        p1 = tuple(np.round(
            calibracao.cm_para_pixel(*caminho_cm[i], ORIGEM_PIXELS)
        ).astype(int))
        p2 = tuple(np.round(
            calibracao.cm_para_pixel(*caminho_cm[i + 1], ORIGEM_PIXELS)
        ).astype(int))
        cv2.line(frame, p1, p2, (0, 0, 255), 3)

    for centro in centros_vermelhos:
        x_px, y_px = centro
        x_cm, y_cm = calibracao.pixel_para_cm(
            x_px, y_px, ORIGEM_PIXELS
        )
        centro_px = (int(round(x_px)), int(round(y_px)))
        cv2.circle(frame, centro_px, 6, (0, 0, 255), -1)
        cv2.putText(
            frame, f"Vermelho ({x_cm:.1f}, {y_cm:.1f}) cm",
            (centro_px[0] + 10, centro_px[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 0, 255), 2
        )
        print(f"Centro vermelho: ({x_cm:.1f}, {y_cm:.1f}) cm")

    desenhar_referencial(frame, calibracao, ORIGEM_PIXELS)
    frame = desenharResultados(frame, objetos, calibracao)
    cv2.imshow("mascara", mascara)
    cv2.imshow("Mapeamento pixel para cm", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()