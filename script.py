#!/usr/bin/env python
import os
import argparse
import cv2


class BrandDetector:

    def detect_all(self, img):
        return [((0, 100), (400, 200)), ((300, 100), (500, 500))]


parse = argparse.ArgumentParser()
parse.add_argument("-fp", "--filepath", help="Ruta a directorio con imagenes a evaluar")
args = parse.parse_args()
with os.scandir(args.filepath) as fs:
    imgs = [f.name for f in fs if f.is_file() and f.name.endswith(("jpg", "jpeg", "png"))]
for img in imgs:
    im_loaded = cv2.imread(args.filepath+img)
    bbox = BrandDetector().detect_all(im_loaded)  # HAY QUE INCLUIR EL METODO
    if not os.path.exists("./blurred"):
        os.mkdir("./blurred")
    for xy in bbox:
        x, y = xy[0][0], xy[0][1]
        w, h = xy[0][0] + xy[1][0], xy[0][1] + xy[1][1]
        ROI = im_loaded[y:y+h, x:x+w]
        blur = cv2.blur(ROI, (61, 61), 0)
        im_loaded[y:y+h, x:x+w] = blur
        cv2.imwrite("blurred/"+img, im_loaded)
