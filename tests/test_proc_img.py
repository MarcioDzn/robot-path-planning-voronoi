import unittest

import cv2
import numpy as np

from proc_img import detectarFormas, detectarMarcadoresVermelhos


class TestDetectarFormas(unittest.TestCase):
    def test_ignora_fundo_branco_total(self):
        frame = np.ones((300, 300, 3), dtype=np.uint8) * 255

        objetos, _ = detectarFormas(frame)

        self.assertEqual(len(objetos), 0)

    def test_detecta_muro_externo_que_envolve_arena(self):
        frame = np.ones((300, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(frame, (0, 0), (299, 299), (0, 0, 0), 5)
        cv2.rectangle(frame, (60, 60), (240, 240), (0, 0, 0), 20)
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)

        objetos, _ = detectarFormas(frame)

        self.assertEqual(len(objetos), 1)
        self.assertGreater(objetos[0].area, 50000)

    def test_detecta_parede_com_furo(self):
        frame = np.ones((300, 300, 3), dtype=np.uint8) * 255
        cv2.rectangle(frame, (60, 60), (240, 240), (0, 0, 0), 20)
        cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)

        objetos, _ = detectarFormas(frame)

        self.assertEqual(len(objetos), 1)
        self.assertGreater(objetos[0].area, 1000)
        self.assertLess(objetos[0].area, 50000)

    def test_detecta_marcador_vermelho(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        cv2.circle(frame, (150, 150), 10, (0, 0, 255), -1)

        centros, _ = detectarMarcadoresVermelhos(frame)

        self.assertTrue(any(abs(x - 150) <= 5 and abs(y - 150) <= 5 for x, y in centros))


if __name__ == "__main__":
    unittest.main()
