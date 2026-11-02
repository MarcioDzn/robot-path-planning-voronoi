import cv2
import numpy as np

from aux_class import Objeto
from calibracao import (
    Calibracao,
    adicionar_borda_preta,
    calibrar_xadrez,
    desenhar_referencial,
)
from controle_bt import connect, run_mission
from proc_img import detectarFormas, detectarMarcadoresVermelhos
from voronoi import (
    build_voronoi_roadmap,
    conectar_ao_roadmap,
    construir_obstaculos_dilatados,
    dijkstra_on_edges,
    extrair_contornos_sites,
)

# ----------------------------------------------------------------------
# Configurações
# ----------------------------------------------------------------------
CAM_INDEX = 0

IMAGEM_CALIBRACAO = "novo/xadrez2.png"
IMAGEM_PROCESSAMENTO = "novo/mesa_ponto_vermelho3.png"
LADO_QUADRADO_M = 2.35 / 100
ORIGEM_PIXELS = (210, 500)
INICIO_M = (-30.0 / 100, 23.0 / 100)
OBJETIVO_M = (-55.0 / 100, 0.0 / 100)
RAIO_ROBO_M = 12.5 / 100

# Variáveis de configuração da comunicação com o robô
ENVIAR_PARA_ROBO = False
PORTA_SERIAL = "COM11"
BAUD_RATE = 115200


def simplificar_caminho(caminho_px, max_pontos=12):
    """
    Simplifica o caminho mantendo as quinas principais e limitando
    o número de pontos ao máximo especificado (ex: 12).
    """
    if len(caminho_px) <= max_pontos:
        return caminho_px

    # Converte o array para o formato aceito pelo cv2.approxPolyDP
    pts = caminho_px.astype(np.float32).reshape(-1, 1, 2)
    
    # Começa com uma tolerância (epsilon) inicial e vai ajustando 
    # iterativamente até o número de pontos ficar menor ou igual a max_pontos
    epsilon = 1.0
    for _ in range(50):  # Limite de segurança para evitar loop infinito
        aprox = cv2.approxPolyDP(pts, epsilon, closed=False)
        if len(aprox) <= max_pontos:
            break
        epsilon += 1.05  # Aumenta a tolerância para remover mais pontos
        
    # Retorna no formato (N, 2)
    return aprox.reshape(-1, 2)


def formatar_waypoints(caminho_m):
    waypoints_formatados = []
    n_pts = len(caminho_m)
    for i in range(n_pts):
        x, y = caminho_m[i]
        if i < n_pts - 1:
            dx = caminho_m[i + 1][0] - x
            dy = caminho_m[i + 1][1] - y
            th = np.degrees(np.arctan2(dy, dx))
        else:
            th = waypoints_formatados[-1][2] if waypoints_formatados else 0.0
        waypoints_formatados.append((float(x), float(y), float(th)))
    return waypoints_formatados


def enviar_missao_robo(waypoints_formatados, enviar, porta, baud):
    if enviar:
        print(f"\n[Bluetooth] Tentando conectar em {porta}...")
        try:
            ser = connect(porta, baud)
            print("[Bluetooth] Conectado!")

            # Usa a função em lote do controle_bt.py (<W> + <R> + Telemetria)
            ok, msg = run_mission(ser, waypoints_formatados, poll_s=0.5, timeout_s=120.0)
            print(f"Resultado da missão: {msg}")

            ser.close()
            print("[Bluetooth] Porta fechada.")
        except Exception as e:
            print(f"Erro na comunicação serial com o robô: {e}")
    else:
        print("\n[Modo Simulação] Envio para o robô desativado (ENVIAR_PARA_ROBO = False).")


def desenharResultados(frame, objetos: list[Objeto], calibracao: Calibracao):
    for obj in objetos:
        cx, cy = obj.centroide

        cv2.drawContours(frame, [obj.contorno], -1, (128, 128, 128), 1)
        cv2.drawContours(frame, [obj.poligono], -1, obj.cor, 3)
        cv2.circle(frame, (cx, cy), 4, obj.cor, -1)

        if obj.area >= 1000:
            x_m, y_m = calibracao.pixel_para_m(cx, cy, ORIGEM_PIXELS)
            rotulo = f"{obj.nome} ({x_m:.1f}, {y_m:.1f}) m"
        else:
            rotulo = f"{obj.nome} ({obj.vertices}v)"
        cv2.putText(
            frame,
            rotulo,
            (cx - 30, cy - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    return frame


def abrir_camera():
    """Devolve um VideoCapture funcionando ou None se não der."""
    try:
        video = cv2.VideoCapture(CAM_INDEX)
        if not video.isOpened():
            raise RuntimeError("câmera não abriu")
        ok, _ = video.read()
        if not ok:
            raise RuntimeError("câmera abriu mas não entregou frame")
        return video
    except Exception as e:
        print(f"Câmera indisponível ({e}); usando imagens estáticas.")
        return None


def aguardar_calibracao(video):
    """Mostra o vídeo ao vivo até apertar N (devolve o frame) ou Q (devolve None)."""
    while True:
        ok, frame = video.read()
        if not ok:
            print("Falha ao ler frame da câmera.")
            return None

        preview = frame.copy()
        cv2.putText(
            preview,
            "N = calibrar com este frame | Q = sair",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("Aguardando calibracao", preview)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (ord("n"), ord("N")):
            cv2.destroyWindow("Aguardando calibracao")
            return frame
        if tecla in (ord("q"), ord("Q")):
            return None


def processar_frame(frame_bruto, calibracao):
    frame = adicionar_borda_preta(frame_bruto)
    if frame is None:
        raise FileNotFoundError("Não foi possível abrir a imagem de processamento")

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

    # ---------- Liga início/objetivo ao roadmap ----------
    arestas_inicio = conectar_ao_roadmap(inicio_px, bordas, proibido)
    arestas_objetivo = conectar_ao_roadmap(objetivo_px, bordas, proibido)

    bordas_busca = bordas + arestas_inicio + arestas_objetivo

    # ---------- Caminho ----------
    caminho_px = dijkstra_on_edges(bordas_busca, inicio_px, objetivo_px)

    if caminho_px is None or len(caminho_px) < 2:
        print("Nenhum caminho encontrado.")
        caminho_m = np.empty((0, 2))
    else:
        caminho_px = simplificar_caminho(np.array(caminho_px), max_pontos=12)
        
        caminho_m = np.array(
            [calibracao.pixel_para_m(x, y, ORIGEM_PIXELS) for x, y in caminho_px],
            dtype=float,
        ).reshape(-1, 2)
        print(caminho_m)

        waypoints_formatados = formatar_waypoints(caminho_m)

        enviar_missao_robo(waypoints_formatados, ENVIAR_PARA_ROBO, PORTA_SERIAL, BAUD_RATE)

        for p, q in zip(caminho_px[:-1], caminho_px[1:]):
            cv2.line(
                frame,
                tuple(np.round(p).astype(int)),
                tuple(np.round(q).astype(int)),
                (0, 0, 255),
                3,
            )

    # ---------- Marcadores vermelhos ----------
    centro = centros_vermelhos[0]
    x_px, y_px = centro
    x_m, y_m = calibracao.pixel_para_m(x_px, y_px, ORIGEM_PIXELS)
    centro_px = (int(round(x_px)), int(round(y_px)))
    cv2.circle(frame, centro_px, 6, (0, 0, 255), -1)
    cv2.putText(
        frame,
        f"Vermelho ({x_m:.1f}, {y_m:.1f}) m",
        (centro_px[0] + 10, centro_px[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    print(f"Centro vermelho: ({x_m:.1f}, {y_m:.1f}) m")

    # ----Exibir destino----
    x_px, y_px = objetivo_px
    x_m, y_m = calibracao.pixel_para_m(x_px, y_px, ORIGEM_PIXELS)
    centro_px = (int(round(x_px)), int(round(y_px)))
    cv2.circle(frame, centro_px, 6, (255, 0, 0), -1)
    cv2.putText(
        frame,
        f"Destino ({x_m:.1f}, {y_m:.1f}) m",
        (centro_px[0] + 10, centro_px[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    print(f"Centro destino: ({x_m:.1f}, {y_m:.1f}) m")

    # ---------- Exibição ----------
    desenhar_referencial(frame, calibracao, ORIGEM_PIXELS)
    frame = desenharResultados(frame, objetos, calibracao)

    return frame, mascara, proibido


def main():
    video = abrir_camera()

    # ---------------- Sem câmera: imagens estáticas ----------------
    if video is None:
        imagem_calib = cv2.imread(str(IMAGEM_CALIBRACAO), cv2.IMREAD_GRAYSCALE)
        if imagem_calib is None:
            raise FileNotFoundError(
                f"Não foi possível abrir a imagem de calibração: {IMAGEM_CALIBRACAO}"
            )
        if imagem_calib.ndim == 3:
            imagem_calib = cv2.cvtColor(imagem_calib, cv2.COLOR_BGR2GRAY)

        calibracao = calibrar_xadrez(
            adicionar_borda_preta(imagem_calib), LADO_QUADRADO_M
        )

        frame_bruto = cv2.imread(IMAGEM_PROCESSAMENTO)
        if frame_bruto is None:
            raise FileNotFoundError(
                f"Não foi possível abrir a imagem: {IMAGEM_PROCESSAMENTO}"
            )

        frame, mascara, proibido = processar_frame(frame_bruto, calibracao)
        cv2.imshow("mascara obstaculos", mascara)
        cv2.imshow("regiao proibida", proibido)
        cv2.imshow("Mapeamento pixel para m", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # ---------------- Com câmera ----------------
    try:
        frame_calib = aguardar_calibracao(video)
        if frame_calib is None:
            return
        if frame_calib.ndim == 3:
            frame_calib = cv2.cvtColor(frame_calib, cv2.COLOR_BGR2GRAY)

        calibracao = calibrar_xadrez(
            adicionar_borda_preta(frame_calib), LADO_QUADRADO_M
        )

        print(
            f"Calibração: {calibracao.pixels_por_m:.2f} px/m; "
            f"ângulo dos eixos: {np.degrees(calibracao.angulo_rad):.2f}°; "
            f"quadrados detectados: {calibracao.quadrados_detectados}"
        )

        while True:
            ok, frame_bruto = video.read()
            if not ok:
                print("Falha ao ler frame; encerrando.")
                break

            frame, mascara, proibido = processar_frame(frame_bruto, calibracao)

            cv2.imshow("mascara obstaculos", mascara)
            cv2.imshow("regiao proibida", proibido)
            cv2.imshow("Mapeamento pixel para m", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()