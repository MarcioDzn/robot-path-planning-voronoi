import math
import time
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from aux_class import Objeto
from calibracao import (
    Calibracao,
    adicionar_borda_preta,
    calibrar_xadrez,
    desenhar_referencial,
)
from controle_bt import connect, run_mission, set_pose, follow_waypoints, get_telemetry
from proc_img import detectarFormas, detectarMarcadoresVermelhos
from voronoi import (
    build_voronoi_roadmap,
    conectar_ao_roadmap,
    construir_obstaculos_dilatados,
    dijkstra_on_edges,
    extrair_contornos_sites,
)

CAM_INDEX = 0

IMAGEM_CALIBRACAO = "novo/xadrez4.png"
IMAGEM_PROCESSAMENTO = "novo/image2.png"
LADO_QUADRADO_M = 2.35 / 100
ORIGEM_PIXELS = (600, 400)
INICIO_M = (-30.0 / 100, 23.0 / 100)
OBJETIVO_M = (-0.7,-0.1) #(-55.0 / 100, 0.0 / 100)
RAIO_ROBO_M = 12.5 / 100

ENVIAR_PARA_ROBO = True
PORTA_SERIAL = "/dev/ttyUSB0"
BAUD_RATE = 57600
K_L = 0.75
K_THETA = 1.75
V_LIN_MAX = 0.722
V_ANG_MAX = 8.5
TOL_POSICAO_M = 0.02
PERIODO_CONTROLE_S = 0.05
TIMEOUT_MISSAO_S = 120.0

def simplificar_caminho(caminho_px, max_pontos=25):

    if len(caminho_px) <= max_pontos:
        return caminho_px

    pts = caminho_px.astype(np.float32).reshape(-1, 1, 2)
    
    epsilon = 1.0
    for _ in range(50): 
        aprox = cv2.approxPolyDP(pts, epsilon, closed=False)
        if len(aprox) <= max_pontos:
            break
        epsilon += 1.05  
        
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

        # robô não precisa saber da orientação
        waypoints_formatados.append((float(x), float(y), 0))
    
    return waypoints_formatados


def controle_referencia_variavel_posicao(
    pose_atual,
    objetivo,
    k_l,
    k_theta,
    v_lin_max,
    v_ang_max,
    tol,
):
    """Calcula (velocidade linear, omega) para a pose atual e o alvo."""
    x, y, theta = pose_atual
    x_d, y_d = objetivo
    valores = (x, y, theta, x_d, y_d, k_l, k_theta, v_lin_max, v_ang_max, tol)
    if not all(math.isfinite(valor) for valor in valores):
        raise ValueError("Pose, objetivo e parâmetros devem ser valores finitos.")
    if k_l < 0 or k_theta < 0 or v_lin_max <= 0 or v_ang_max <= 0 or tol <= 0:
        raise ValueError("Ganhos não podem ser negativos; limites e tolerância devem ser positivos.")

    delta_x = x_d - x
    delta_y = y_d - y
    delta_l = math.hypot(delta_x, delta_y)

    if delta_l < tol:
        return 0.0, 0.0, True

    theta_d = math.atan2(delta_y, delta_x)
    delta_theta = math.atan2(
        math.sin(theta_d - theta),
        math.cos(theta_d - theta),
    )
    delta_l_proj = delta_l * math.cos(delta_theta)
    v_lin = k_l * delta_l_proj
    v_ang = k_theta * delta_theta

    v_lin = max(-v_lin_max, min(v_lin, v_lin_max))
    v_ang = max(-v_ang_max, min(v_ang, v_ang_max))
    return v_lin, v_ang, False


def enviar_missao_robo(
    waypoints_formatados,
    enviar,
    pose,
    k_l=K_L,
    k_theta=K_THETA,
    v_lin_max=V_LIN_MAX,
    v_ang_max=V_ANG_MAX,
    tol=TOL_POSICAO_M,
    periodo_s=PERIODO_CONTROLE_S,
    timeout_s=TIMEOUT_MISSAO_S,
):
    """Lê a pose e chama run_mission com velocidade linear e angular."""
    if not enviar:
        print("\n[Modo Simulação] Envio para o robô desativado (ENVIAR_PARA_ROBO = False).")
        return

    if not waypoints_formatados:
        print("Missão não enviada: nenhum waypoint válido.")
        return
    if pose is None:
        raise RuntimeError(
            "Configure OBTER_POSE_ROBO antes de enviar a missão."
        )
    if (
        k_l < 0
        or k_theta < 0
        or v_lin_max <= 0
        or v_ang_max <= 0
        or tol <= 0
        or periodo_s <= 0
        or timeout_s <= 0
    ):
        raise ValueError("Ganhos não podem ser negativos; limites, tolerância e tempos devem ser positivos.")
    if not all(
        math.isfinite(valor)
        for valor in (k_l, k_theta, v_lin_max, v_ang_max, tol, periodo_s, timeout_s)
    ):
        raise ValueError("Parâmetros do controlador e do loop devem ser valores finitos.")


    inicio = time.monotonic()
    ser, rx_q, running, t = connect(port, args.baud, args.timeout)

    try:
        # define a pose inicial do robô
        # TODO: mudar pra a pose que a gente quer
        set_pose(ser, rx_q, pose[0], pose[1], pose[2])

        for waypoint in waypoints_formatados:
            objetivo = waypoint[:2]

            while True:
                if time.monotonic() - inicio >= timeout_s:
                    raise TimeoutError("Tempo limite da missão excedido.")

                tm = get_telemetry(ser, rx_q)
                if tm is None:
                    time.sleep(periodo_s)
                    continue

                pose_atual = (
                    tm["x"],
                    tm["y"],
                    math.radians(tm["th_deg"]),
                )

                v_lin, v_ang, concluido = controle_referencia_variavel_posicao(
                    pose_atual,
                    objetivo,
                    k_l,
                    k_theta,
                    v_lin_max,
                    v_ang_max,
                    tol,
                )

                if concluido:
                    send_v(ser, 0.0, 0.0)
                    break

                send_v(ser, v_lin, v_ang)
                time.sleep(periodo_s)

    finally:
        send_v(ser, 0.0, 0.0)
        close(ser, running, t)


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

    if centros_vermelhos:
        inicio_px = centros_vermelhos[0]
    else:
        inicio_px = calibracao.m_para_pixel(*INICIO_M, ORIGEM_PIXELS)

    objetivo_px = calibracao.m_para_pixel(*OBJETIVO_M, ORIGEM_PIXELS)


    objetos, mascara, arena, circulo = detectarFormas(frame, mascara_vermelha, inicio_px, calibracao.m_para_pixels_inteiro(RAIO_ROBO_M))

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

    contornos_proibido, _ = cv2.findContours(
        proibido, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(frame, contornos_proibido, -1, (255, 0, 255), 1)

    arestas_inicio = conectar_ao_roadmap(inicio_px, bordas, proibido)
    arestas_objetivo = conectar_ao_roadmap(objetivo_px, bordas, proibido)

    bordas_busca = bordas + arestas_inicio + arestas_objetivo

    caminho_px = dijkstra_on_edges(bordas_busca, inicio_px, objetivo_px)

    # nenhum caminho encontrado
    if caminho_px is None or len(caminho_px) < 2:
        caminho_m = np.empty((0, 2))
        waypoints_formatados = []
    else:
        caminho_px = simplificar_caminho(np.array(caminho_px), max_pontos=25)
        
        caminho_m = np.array(
            [calibracao.pixel_para_m(x, y, ORIGEM_PIXELS) for x, y in caminho_px],
            dtype=float,
        ).reshape(-1, 2)

        waypoints_formatados = formatar_waypoints(caminho_m)


        for p, q in zip(caminho_px[:-1], caminho_px[1:]):
            cv2.line(
                frame,
                tuple(np.round(p).astype(int)),
                tuple(np.round(q).astype(int)),
                (0, 0, 255),
                3,
            )

    # marcacao do centro vermelho
    x_px, y_px = inicio_px
    x_red_m, y_red_m = calibracao.pixel_para_m(x_px, y_px, ORIGEM_PIXELS)
    centro_px = (int(round(x_px)), int(round(y_px)))
    cv2.circle(frame, centro_px, 6, (0, 0, 255), -1)
    cv2.putText(
        frame,
        f"Vermelho ({x_red_m:.1f}, {y_red_m:.1f}) m",
        (centro_px[0] + 10, centro_px[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    # destino
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
    print(waypoints_formatados)
    tecla = cv2.waitKey(1) & 0xFF
    if tecla in (ord("s"), ord("S")):
        enviar_missao_robo(
            waypoints_formatados,
            ENVIAR_PARA_ROBO,
            pose=(x_red_m, y_red_m, 0.0),
        )

    elif tecla in (ord('t'), ord('T')):
        ser = connect(PORTA_SERIAL, BAUD_RATE)
        print("[Bluetooth] Conectado!")    
        set_pose(ser, x_red_m, y_red_m, 0)
        print(f"posição inicial: {x_red_m, y_red_m}")

    desenhar_referencial(frame, calibracao, ORIGEM_PIXELS)
    frame = desenharResultados(frame, objetos, calibracao)

    return frame, mascara, proibido, circulo


def main():
    video = abrir_camera()

    # caso não tenha camera
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

        frame, mascara, proibido, circulo = processar_frame(frame_bruto, calibracao)
        cv2.imshow("mascara obstaculos", mascara)
        cv2.imshow("regiao proibida", proibido)
        cv2.imshow("Exclusao robô", circulo)
        cv2.imshow("Mapeamento pixel para m", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return


    try:
        # frame_calib = aguardar_calibracao(video)
        # if frame_calib is None:
        #     return
        # if frame_calib.ndim == 3:
        #     frame_calib = cv2.cvtColor(frame_calib, cv2.COLOR_BGR2GRAY)

        # calibracao = calibrar_xadrez(
        #     adicionar_borda_preta(frame_calib), LADO_QUADRADO_M
        # )
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

            frame, mascara, proibido, circulo = processar_frame(frame_bruto, calibracao)

            cv2.imshow("mascara obstaculos", mascara)
            cv2.imshow("regiao proibida", proibido)
            cv2.imshow("Exclusao robô", circulo)
            cv2.imshow("Mapeamento pixel para m", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()