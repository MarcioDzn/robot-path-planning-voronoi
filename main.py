import cv2
import numpy as np
from aux_class import Objeto
from calibracao import Calibracao, calibrar_xadrez, desenhar_referencial
from proc_img import detectarFormas
from voronoi import build_voronoi_roadmap, dijkstra_on_edges, point_in_any_polygon


# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------
CAM_INDEX = 0
IMAGEM_CALIBRACAO = "novo/xadrez.png"
IMAGEM_PROCESSAMENTO = "novo/mesa.png"
LADO_QUADRADO_CM = 2.35
ORIGEM_PIXELS = (860, 100)


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
    video = cv2.VideoCapture(CAM_INDEX)

    if not video.isOpened():
        print("Não foi possível abrir a câmera.")
        return

    calibracao = calibrar_xadrez(IMAGEM_CALIBRACAO, LADO_QUADRADO_CM)
    print(
        f"Calibração: {calibracao.pixels_por_cm:.2f} px/cm; "
        f"ângulo dos eixos: {np.degrees(calibracao.angulo_rad):.2f}°; "
        f"quadrados detectados: {calibracao.quadrados_detectados}"
    )

    frames = 0

    while True:
        check, _ = video.read()
        if not check:
            break

        frame = cv2.imread(IMAGEM_PROCESSAMENTO)
        if frame is None:
            raise FileNotFoundError(
                f"Não foi possível abrir a imagem de processamento: {IMAGEM_PROCESSAMENTO}"
            )

        frames += 1
        objetos, mascara = detectarFormas(frame)
        obstaculos = [
            obj.poligono.reshape(-1, 2).astype(float)
            for obj in objetos
        ]

        altura, largura = frame.shape[:2]
        limites = (0, 0, largura - 1, altura - 1)
        bordas = build_voronoi_roadmap(obstaculos, limites)

        for a, b in bordas:
            p1 = tuple(np.round(a).astype(int))
            p2 = tuple(np.round(b).astype(int))
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        inicio = (600, 390)
        objetivo = (600, 600)

        if point_in_any_polygon(inicio, obstaculos) or point_in_any_polygon(objetivo, obstaculos):
            print("Caminho inválido")

        caminho = dijkstra_on_edges(bordas, inicio, objetivo)

        for i in range(len(caminho) - 1):
            p1 = tuple(np.round(caminho[i]).astype(int))
            p2 = tuple(np.round(caminho[i + 1]).astype(int))
            cv2.line(frame, p1, p2, (0, 0, 255), 3)

        desenhar_referencial(frame, calibracao, ORIGEM_PIXELS)
        frame = desenharResultados(frame, objetos, calibracao)
        cv2.imshow("mascara", mascara)
        cv2.imshow("Mapeamento pixel para cm", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    print(f"Total de frames processados: {frames}")

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()