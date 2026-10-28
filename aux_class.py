

class Objeto():
    def __init__(self, nome, centroide, contorno, poligono, vertices, area, cor):            
        self.nome = nome
        self.centroide = centroide
        self.contorno = contorno
        self.poligono = poligono
        self.vertices = vertices
        self.area = area
        self.cor = cor


class Robo():
    def __init__(self):
        self.pos_x = 0
        self.pos_y = 0
        self.vel_lin = 0
        self.vel_ang = 0
        