import cv2
import numpy as np
from aux_class import Objeto
from calibracao import Calibracao, calibrar_xadrez, desenhar_referencial, adicionar_borda_preta
from proc_img import detectarFormas, detectarMarcadoresVermelhos
from voronoi import (
    build_voronoi_roadmap,
    construir_obstaculos_dilatados,
    dijkstra_on_edges,
    livre,
    extrair_contornos_sites, conectar_ao_roadmap
)


# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------
IMAGEM_CALIBRACAO = "novo/xadrez2.png"
IMAGEM_PROCESSAMENTO = "novo/mesa_ponto_vermelho3.png"
LADO_QUADRADO_M = 2.35 / 100
ORIGEM_PIXELS = (210, 500)
INICIO_M = (-30.0 / 100, 23.0 / 100)
OBJETIVO_M = (-55.0 / 100, 0.0 / 100)
RAIO_ROBO_M = 12.5 / 100


def desenharResultados(
    frame, objetos: list[Objeto], calibracao: Calibracao
):
    for obj in objetos:
        cx, cy = obj.centroide

        cv2.drawContours(frame, [obj.contorno], -1, (128, 128, 128), 1)
        cv2.drawContours(frame, [obj.poligono], -1, obj.cor, 3)
        cv2.circle(frame, (cx, cy), 4, obj.cor, -1)

        if obj.area >= 1000:
            x_m, y_m = calibracao.pixel_para_m(
                cx, cy, ORIGEM_PIXELS
            )
            rotulo = f"{obj.nome} ({x_m:.1f}, {y_m:.1f}) m"
        else:
            rotulo = f"{obj.nome} ({obj.vertices}v)"
        cv2.putText(
            frame, rotulo, (cx - 30, cy - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
        )

    return frame


def selecionar_regiao(frame):
    janela = "Selecione a região para processar"
    ponto_inicial = None
    ponto_atual = None

    def _on_event(event, x, y, flags, param):
        nonlocal ponto_inicial, ponto_atual
        if event == cv2.EVENT_LBUTTONDOWN:
            ponto_inicial = (x, y)
            ponto_atual = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and ponto_inicial is not None:
            ponto_atual = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and ponto_inicial is not None:
            ponto_atual = (x, y)

    cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(janela, _on_event)

    while True:
        display = frame.copy()

        if ponto_inicial is not None and ponto_atual is not None:
            x0 = min(ponto_inicial[0], ponto_atual[0])
            x1 = max(ponto_inicial[0], ponto_atual[0])
            y0 = min(ponto_inicial[1], ponto_atual[1])
            y1 = max(ponto_inicial[1], ponto_atual[1])
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)

        cv2.imshow(janela, display)
        tecla = cv2.waitKey(1) & 0xFF

        if ponto_inicial is not None and ponto_atual is not None:
            x0 = min(ponto_inicial[0], ponto_atual[0])
            x1 = max(ponto_inicial[0], ponto_atual[0])
            y0 = min(ponto_inicial[1], ponto_atual[1])
            y1 = max(ponto_inicial[1], ponto_atual[1])

            if tecla == ord('c'):
                cv2.destroyWindow(janela)
                return frame[y0:y1, x0:x1]

        if tecla == ord('q'):
            cv2.destroyWindow(janela)
            return frame


def main():
    calibracao = calibrar_xadrez(IMAGEM_CALIBRACAO, LADO_QUADRADO_M)
    print(
        f"Calibração: {calibracao.pixels_por_m:.2f} px/m; "
        f"ângulo dos eixos: {np.degrees(calibracao.angulo_rad):.2f}°; "
        f"quadrados detectados: {calibracao.quadrados_detectados}"
    )

    frame = adicionar_borda_preta(cv2.imread(IMAGEM_PROCESSAMENTO))
    if frame is None:
        raise FileNotFoundError(
            f"Não foi possível abrir a imagem de processamento: {IMAGEM_PROCESSAMENTO}"
        )

    centros_vermelhos, mascara_vermelha = detectarMarcadoresVermelhos(frame)
    objetos, mascara, arena = detectarFormas(frame, mascara_vermelha)

    altura, largura = frame.shape[:2]
    limites = (0, 0, largura - 1, altura - 1)
    raio_robo_px = RAIO_ROBO_M * calibracao.pixels_por_m

    proibido = construir_obstaculos_dilatados(mascara, arena, raio_robo_px)
    obstaculos_para_sites = extrair_contornos_sites(proibido)
    

    bordas = build_voronoi_roadmap(obstaculos_para_sites, limites, proibido)

    for a, b in bordas:
        p1 = tuple(np.round(a).astype(int))
        p2 = tuple(np.round(b).astype(int))
        cv2.line(frame, p1, p2, (0, 255, 0), 2)

    # ---------- Debug visual: região proibida ----------
    contornos_proibido, _ = cv2.findContours(
        proibido, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(frame, contornos_proibido, -1, (255, 0, 255), 1)

    # ---------- Início e objetivo ----------
    if centros_vermelhos:
        inicio_px = centros_vermelhos[0]
        print(f"Início da trajetória definido pelo marcador: {inicio_px} px")
    else:
        inicio_px = calibracao.m_para_pixel(*INICIO_M, ORIGEM_PIXELS)
        print(f"Sem marcador; usando INICIO_M como início: {INICIO_M} m")

    objetivo_px = calibracao.m_para_pixel(*OBJETIVO_M, ORIGEM_PIXELS)

    # if not livre(inicio_px, proibido):
    #     print("Aviso: o início está na margem de segurança.")
    # if not livre(objetivo_px, proibido):
    #     raise ValueError("O objetivo está dentro da margem de segurança.")

    # ---------- Liga início/objetivo ao roadmap ----------
    arestas_inicio = conectar_ao_roadmap(inicio_px, bordas, proibido)
    arestas_objetivo = conectar_ao_roadmap(objetivo_px, bordas, proibido)

    # if not arestas_inicio or not arestas_objetivo:
    #     raise ValueError(
    #         "Início ou objetivo não consegue se conectar ao roadmap."
    #     )

    bordas_busca = bordas + arestas_inicio + arestas_objetivo

    # ---------- Caminho ----------
    caminho_px = dijkstra_on_edges(bordas_busca, inicio_px, objetivo_px)

    if caminho_px is None or len(caminho_px) < 2:
        print("Nenhum caminho encontrado.")
        caminho_m = np.empty((0, 2))
    else:
        caminho_m = np.array([
            calibracao.pixel_para_m(x, y, ORIGEM_PIXELS)
            for x, y in caminho_px
        ], dtype=float).reshape(-1, 2)
        print(caminho_m)

        for p, q in zip(caminho_px[:-1], caminho_px[1:]):
            cv2.line(
                frame,
                tuple(np.round(p).astype(int)),
                tuple(np.round(q).astype(int)),
                (0, 0, 255), 3
            )

    # ---------- Marcadores vermelhos ----------
    centro = centros_vermelhos[0]
    x_px, y_px = centro
    x_m, y_m = calibracao.pixel_para_m(x_px, y_px, ORIGEM_PIXELS)
    centro_px = (int(round(x_px)), int(round(y_px)))
    cv2.circle(frame, centro_px, 6, (0, 0, 255), -1)
    cv2.putText(
        frame, f"Vermelho ({x_m:.1f}, {y_m:.1f}) m",
        (centro_px[0] + 10, centro_px[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
    )
    print(f"Centro vermelho: ({x_m:.1f}, {y_m:.1f}) m")

    #----Exibir destino----
    x_px, y_px = objetivo_px
    x_m, y_m = calibracao.pixel_para_m(x_px, y_px, ORIGEM_PIXELS)
    centro_px = (int(round(x_px)), int(round(y_px)))
    cv2.circle(frame, centro_px, 6, (255, 0, 0), -1)
    cv2.putText(
        frame, f"Destino ({x_m:.1f}, {y_m:.1f}) m",
        (centro_px[0] + 10, centro_px[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
    )
    print(f"Centro destino: ({x_m:.1f}, {y_m:.1f}) m")


    # ---------- Exibição ----------
    desenhar_referencial(frame, calibracao, ORIGEM_PIXELS)
    frame = desenharResultados(frame, objetos, calibracao)

    cv2.imshow("mascara obstaculos", mascara)
    cv2.imshow("regiao proibida", proibido)
    cv2.imshow("Mapeamento pixel para m", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()