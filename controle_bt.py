#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controle remoto do Minibalance via Bluetooth (ZS-040 / HC-05 / HC-06).
Configuração padrão: COM11 @ 115200 baud (Windows).

Protocolo:
  <G,x,y,th>              alvo único  (firmware já liga Flag_Stop)
  <O,x,y,th>              redefine pose
  <R>                     liga + auto-eq 2s + navega waypoints
  <C> ou <CLR>            zera lista e para
  <A,x,y,th>              adiciona waypoint
  <S,idx,x,y,th>          substitui waypoint
  <D,idx>                 remove waypoint
  <L>                     lista waypoints
  <W,n,x0,y0,t0,...>      SUBSTITUI a lista por vetor
  <Q,n,x0,y0,t0,...>      ADICIONA vetor ao final
  <T>                     telemetria: <T,x,y,th,wp,st,v,w>
"""
import sys
import time
import argparse
import serial
import serial.tools.list_ports

DEFAULT_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 57600
TIMEOUT_S = 1.0
RESPONSE_TIMEOUT_S = 2.0
AUTO_EQ_S = 2.0
START_GRACE_S = 2.5

STATE_NAMES = {0: "GIRAR", 1: "ANDAR", 2: "DONE"}


# ---------------------------------------------------------------------------
# Conexão
# ---------------------------------------------------------------------------
def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("Nenhuma porta serial encontrada.")
        return
    print("Portas disponíveis:")
    for p in ports:
        print(f"  {p.device:20s}  {p.description}")


def connect(port=DEFAULT_PORT, baud=DEFAULT_BAUD):
    ser = serial.Serial(port, baud, timeout=TIMEOUT_S)
    time.sleep(1.5)
    ser.reset_input_buffer()
    return ser


# ---------------------------------------------------------------------------
# Envio e leitura
# ---------------------------------------------------------------------------
def send(ser, cmd):
    ser.write(f"<{cmd}>\n".encode("ascii"))
    ser.flush()


def read_response(ser, timeout=RESPONSE_TIMEOUT_S):
    t0 = time.time()
    buf = bytearray()
    while time.time() - t0 < timeout:
        b = ser.read(1)
        if not b:
            continue
        buf += b
        if b == b"\n":
            line = buf.decode("ascii", errors="replace").strip()
            if line.startswith("<") and line.endswith(">"):
                return line
            buf = bytearray()
    return None


def cmd_and_wait(ser, cmd, timeout=RESPONSE_TIMEOUT_S):
    send(ser, cmd)
    return read_response(ser, timeout)


# ---------------------------------------------------------------------------
# Waypoints
# ---------------------------------------------------------------------------
def add_waypoint(ser, x, y, th):
    return cmd_and_wait(ser, f"A,{x:.3f},{y:.3f},{th:.1f}")


def set_waypoints(ser, pts):
    if not pts:
        return cmd_and_wait(ser, "C")
    body = f"W,{len(pts)}"
    for (x, y, th) in pts:
        body += f",{x:.3f},{y:.3f},{th:.1f}"
    return cmd_and_wait(ser, body)


def append_waypoints(ser, pts):
    if not pts:
        return None
    body = f"Q,{len(pts)}"
    for (x, y, th) in pts:
        body += f",{x:.3f},{y:.3f},{th:.1f}"
    return cmd_and_wait(ser, body)


def list_waypoints(ser):
    resp = cmd_and_wait(ser, "L")
    if resp is None or not resp.startswith("<L,"):
        return None
    parts = resp[1:-1].split(",")[1:]
    n = int(parts[0])
    wps = []
    for i in range(n):
        x = float(parts[1 + 3 * i])
        y = float(parts[2 + 3 * i])
        th = float(parts[3 + 3 * i])
        wps.append((x, y, th))
    return wps


def delete_waypoint(ser, idx):
    return cmd_and_wait(ser, f"D,{idx}")


def replace_waypoint(ser, idx, x, y, th):
    return cmd_and_wait(ser, f"S,{idx},{x:.3f},{y:.3f},{th:.1f}")


# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------
def start(ser, wait_auto_eq=True):
    resp = cmd_and_wait(ser, "R")
    if resp is None:
        return False, None, "sem resposta (timeout)"
    if resp != "<OK>":
        if resp == "<ERR>":
            return False, resp, "lista vazia — envie waypoints antes de <R>"
        return False, resp, f"resposta inesperada: {resp}"
    if wait_auto_eq:
        time.sleep(START_GRACE_S)
    return True, resp, "navegação iniciada"


def stop(ser):
    return cmd_and_wait(ser, "C")


def goto(ser, x, y, th):
    return cmd_and_wait(ser, f"G,{x:.3f},{y:.3f},{th:.1f}")


def set_pose(ser, x, y, th):
    return cmd_and_wait(ser, f"O,{x:.3f},{y:.3f},{th:.1f}")


def status(ser):
    wps = list_waypoints(ser)
    return {
        "count": 0 if wps is None else len(wps),
        "waypoints": wps or [],
    }


# ---------------------------------------------------------------------------
# Telemetria
# ---------------------------------------------------------------------------
def get_telemetry(ser, timeout=RESPONSE_TIMEOUT_S):
    """
    Envia <T> e devolve dict com pose/estado, ou None em timeout/erro.

    Chaves: x, y, th_deg, wp, state, v, w_deg
    """
    resp = cmd_and_wait(ser, "T", timeout)
    if resp is None or not resp.startswith("<T,"):
        return None
    body = resp[3:-1]              # remove "<T," e ">"
    parts = body.split(",")
    if len(parts) != 7:
        return None
    try:
        return {
            "x":      float(parts[0]),
            "y":      float(parts[1]),
            "th_deg": float(parts[2]),
            "wp":     int(parts[3]),
            "state":  int(parts[4]),
            "v":      float(parts[5]),
            "w_deg":  float(parts[6]),
        }
    except ValueError:
        return None


def format_telemetry(t):
    if t is None:
        return "(sem telemetria)"
    st = STATE_NAMES.get(t["state"], "?")
    return (
        f"x={t['x']:+.3f}  y={t['y']:+.3f}  "
        f"th={t['th_deg']:+7.2f}°  "
        f"wp={t['wp']:2d}  st={st:5s}  "
        f"v={t['v']:+.3f} m/s  w={t['w_deg']:+6.2f} °/s"
    )


def print_telemetry(t):
    print(format_telemetry(t))


def wait_until_done_telemetry(ser, poll_s=0.5, timeout_s=120.0,
                              verbose=True, initial_delay_s=0.0):
    """
    Monitora via <T> até pos_state == DONE (2) ou timeout.
    `initial_delay_s` ignora leituras nos primeiros segundos (útil durante
    o auto-equilíbrio de 2 s, em que state ainda está em GIRAR/ANDAR).
    """
    if initial_delay_s > 0:
        time.sleep(initial_delay_s)

    t0 = time.time()
    last_wp = -1
    while time.time() - t0 < timeout_s:
        t = get_telemetry(ser)
        if t is None:
            time.sleep(poll_s)
            continue

        if verbose:
            if t["wp"] != last_wp:
                print(f"  [{time.time()-t0:5.1f}s] waypoint {t['wp']}  "
                      f"{format_telemetry(t)}")
                last_wp = t["wp"]
            else:
                print(f"  [{time.time()-t0:5.1f}s] {format_telemetry(t)}")

        if t["state"] == 2:                 # DONE
            return True
        time.sleep(poll_s)
    return False


def follow_waypoints(ser, waypoints,
                     poll_s=0.3,
                     timeout_per_wp=60.0,
                     position_tol=0.03,
                     angle_tol=3.0,
                     verbose=True):

    if not waypoints:
        return False, "Lista de waypoints vazia"

    for idx, (x_goal, y_goal, th_goal) in enumerate(waypoints):

        if verbose:
            print(f"\n[{idx+1}/{len(waypoints)}] "
                  f"Indo para x={x_goal:.3f}, "
                  f"y={y_goal:.3f}, th={th_goal:.1f}")

        # 1. Envia GOTO
        resp = goto(ser, x_goal, y_goal, th_goal)

        if resp != "<OK>":
            return False, f"Erro ao enviar GOTO: {resp}"

        # 2. Aguarda chegada ao waypoint
        t0 = time.time()
        arrived = False

        while time.time() - t0 < timeout_per_wp:

            t = get_telemetry(ser)

            if t is None:
                time.sleep(poll_s)
                continue

            # Erro de posição
            dx = x_goal - t["x"]
            dy = y_goal - t["y"]

            distance_error = (dx**2 + dy**2)**0.5

            # Erro angular normalizado para [-180, 180]
            angle_error = (
                (th_goal - t["th_deg"] + 180) % 360
            ) - 180

            if verbose:
                print(
                    f"  dist={distance_error:.3f} m | "
                    f"ang={angle_error:.2f}° | "
                    f"state={STATE_NAMES.get(t['state'], '?')}"
                )

            # 3. Confirma chegada
            if (distance_error <= position_tol and
                abs(angle_error) <= angle_tol and
                t["state"] == 2):

                arrived = True
                break

            time.sleep(poll_s)

        if not arrived:
            return False, (
                f"Timeout no waypoint {idx}: "
                f"posição não confirmada"
            )

        if verbose:
            print(f"  Waypoint {idx} alcançado!")

    return True, "Trajetória completamente executada"


def run_mission(ser, pts, poll_s=0.5, timeout_s=120.0):
    print(f"[1/3] Enviando {len(pts)} waypoints via <W>...")


    # pts = [
    #     (0.1, 0, 0),
    #     (0.1, 0.1, 0),
    #     (0.2, 0, 0),
    #     (0.2, -0.1, 0),
    #     (0, 0, 0)
    # ]

    print(f'Parando')
    stop(ser)

    # resp = goto(ser, pts[0], pts[1], 0)
    print(f'Iniciando pontos: {pts}')


    for pt in pts:
        print(f"Pontos individuais: {pt}")   
        resp = add_waypoint(ser, *pt)
        print(f"      resposta: {resp}")
        if resp is None or not resp.startswith("<OK"):
            return False, "falha ao enviar waypoints"

    print("[2/3] Disparando <R> (liga + auto-eq 2s + navega)...")
    ok, resp, msg = start(ser, wait_auto_eq=True)
    print(f"      resposta: {resp}  ({msg})")
    if not ok:
        return False, msg

    print(f"[3/3] Navegando. Monitorando <T> a cada {poll_s:.1f}s (timeout {timeout_s:.0f}s)...")
    done = wait_until_done_telemetry(ser, poll_s=poll_s, timeout_s=timeout_s,
                                     verbose=True, initial_delay_s=0.0)
    if done:
        return True, "missão concluída"
    return False, "timeout de monitoramento"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
HELP = """
Comandos:
  add X Y TH              adiciona waypoint ao final
  set X1 Y1 T1 [...]      SUBSTITUI lista por vetor
  app X1 Y1 T1 [...]      ADICIONA vetor ao final
  list                    lista waypoints
  status                  lista + resumo
  del IDX                 remove waypoint IDX
  sub IDX X Y TH          substitui waypoint IDX
  go X Y TH               alvo único
  pose X Y TH             redefine pose
  start                   liga + auto-eq 2s + navega
  stop                    para e ZERA a lista
  run                     start + monitora até DONE/timeout
  tm                      telemetria: uma amostra
  tm monitor              monitora telemetria a 2 Hz (Ctrl+C para sair)
  raw <texto>             envia texto cru dentro de <...>
  clear                   limpa a tela (não envia nada ao robô)
  help                    esta ajuda
  quit                    sai
"""


def parse_floats(args):
    toks = []
    for a in args:
        toks.extend(a.replace(",", " ").split())
    vals = [float(t) for t in toks]
    if not vals or len(vals) % 3 != 0:
        raise ValueError("Precisa ser múltiplo de 3 (x y th)")
    return [(vals[i], vals[i + 1], vals[i + 2]) for i in range(0, len(vals), 3)]


IGNORE_PREFIXES = ("aviso:", "#", "//", "robot>")


def should_ignore(line):
    low = line.strip().lower()
    if not low:
        return True
    return any(low.startswith(p) for p in IGNORE_PREFIXES)


def interactive(ser):
    print(HELP)
    while True:
        try:
            line = input("robot> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if should_ignore(line):
            continue

        parts = line.split()
        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd in ("quit", "exit", "q"):
                break

            elif cmd == "help":
                print(HELP)

            elif cmd == "clear":
                print("\033[H\033[J", end="")   # ANSI clear (Linux/macOS/Win10+)

            elif cmd == "raw":
                txt = line[len("raw"):].strip()
                resp = cmd_and_wait(ser, txt)
                print(resp if resp else "(sem resposta)")

            elif cmd == "add":
                pts = parse_floats(args)
                if len(pts) != 1:
                    print("add espera X Y TH")
                    continue
                x, y, th = pts[0]
                print(add_waypoint(ser, x, y, th))

            elif cmd == "set":
                print(set_waypoints(ser, parse_floats(args)))

            elif cmd == "app":
                print(append_waypoints(ser, parse_floats(args)))

            elif cmd == "list":
                wps = list_waypoints(ser)
                if wps is None:
                    print("(sem resposta ou erro)")
                else:
                    print(f"{len(wps)} waypoint(s):")
                    for i, (x, y, th) in enumerate(wps):
                        print(f"  [{i:2d}] x={x:.3f}  y={y:.3f}  th={th:.1f}")

            elif cmd == "status":
                st = status(ser)
                print(f"waypoints: {st['count']}")

            elif cmd == "del":
                print(delete_waypoint(ser, int(args[0])))

            elif cmd == "sub":
                idx = int(args[0])
                pts = parse_floats(args[1:])
                if len(pts) != 1:
                    print("sub espera IDX X Y TH")
                    continue
                x, y, th = pts[0]
                print(replace_waypoint(ser, idx, x, y, th))

            elif cmd == "go":
                pts = parse_floats(args)
                if len(pts) != 1:
                    print("go espera X Y TH")
                    continue
                x, y, th = pts[0]
                print(goto(ser, x, y, th))

            elif cmd == "pose":
                pts = parse_floats(args)
                if len(pts) != 1:
                    print("pose espera X Y TH")
                    continue
                x, y, th = pts[0]
                print(set_pose(ser, x, y, th))

            elif cmd == "start":
                ok, resp, msg = start(ser, wait_auto_eq=True)
                print(f"{resp}  ({msg})" if resp else msg)

            elif cmd == "stop":
                print(stop(ser))

            elif cmd == "run":
                wps = list_waypoints(ser)
                if not wps:
                    print("Lista vazia. Envie 'set'/'add' antes de 'run'.")
                    continue
                ok, resp, msg = start(ser, wait_auto_eq=True)
                print(f"{resp}  ({msg})" if resp else msg)
                if ok:
                    print("Monitorando <T>...")
                    wait_until_done_telemetry(ser, poll_s=0.5,
                                              timeout_s=120.0, verbose=True)

            elif cmd == "tm":
                # 'tm' ou 'tm monitor'
                if len(args) > 0 and args[0].lower() == "monitor":
                    print("Monitorando telemetria a 2 Hz (Ctrl+C para sair)...")
                    try:
                        while True:
                            print_telemetry(get_telemetry(ser))
                            time.sleep(0.5)
                    except KeyboardInterrupt:
                        print("\n(monitor encerrado)")
                else:
                    print_telemetry(get_telemetry(ser))

            else:
                print(f"Comando desconhecido: {cmd}. Digite 'help'.")

        except ValueError as e:
            print(f"Erro de argumento: {e}")
        except IndexError:
            print("Faltam argumentos.")
        except serial.SerialException as e:
            print(f"Erro serial: {e}")
            break
        except KeyboardInterrupt:
            print("\n(interrompido)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=f"Controle do Minibalance via Bluetooth "
                    f"(padrão: {DEFAULT_PORT} @ {DEFAULT_BAUD})"
    )
    ap.add_argument("--port", default=DEFAULT_PORT)
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    ap.add_argument("--list", action="store_true", help="lista portas seriais e sai")
    ap.add_argument("--send", help="envia um comando (sem <>) e sai")
    ap.add_argument("--csv", help="CSV com 'x,y,th' por linha; envia <W> + <R>")
    ap.add_argument("--tm", action="store_true", help="uma amostra de telemetria e sai")
    args = ap.parse_args()

    if args.list:
        list_ports()
        sys.exit(0)

    try:
        ser = connect(args.port, args.baud)
    except serial.SerialException as e:
        print(f"Não consegui abrir {args.port} @ {args.baud}: {e}")
        print("Verifique: HC-05 emparelhado, Monitor Serial fechado, porta correta.")
        sys.exit(1)

    print(f"Conectado em {args.port} @ {args.baud} baud.")

    try:
        if args.send:
            print(cmd_and_wait(ser, args.send))

        elif args.tm:
            print_telemetry(get_telemetry(ser))

        elif args.csv:
            pts = []
            with open(args.csv, "r", encoding="utf-8") as f:
                for ln, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    toks = line.replace(";", ",").split(",")
                    if len(toks) < 3:
                        print(f"[{ln}] linha inválida: {line}")
                        continue
                    pts.append((float(toks[0]), float(toks[1]), float(toks[2])))
            if not pts:
                print("CSV sem waypoints válidos.")
                sys.exit(1)
            ok, msg = run_mission(ser, pts)
            print(f"Resultado: {msg}")

        else:
            interactive(ser)
    finally:
        ser.close()
        print("Desconectado.")


if __name__ == "__main__":
    main()
