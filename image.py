import os
import cv2
import imutils
import numpy as np
from utils import Utils


class Image:

    def get_edges(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.medianBlur(image, ksize=3)
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.Canny(image, 30, 200)
        return image

    def prepare_cont(self, image, cont_image):
        contours = cv2.findContours(cont_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        screen_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            if len(approx) == 4:
                screen_cnt = approx
                break
        cv2.drawContours(image, [screen_cnt], -1, (255, 0, 0), 3)
        return screen_cnt

    def order_points(self, tab):    
        sq = np.zeros((4, 2), dtype="float32")

        tmp = tab.sum(axis=1)
        sq[0] = tab[np.argmin(tmp)]
        sq[2] = tab[np.argmax(tmp)]

        tmp = np.diff(tab, axis=1)
        sq[1] = tab[np.argmin(tmp)]
        sq[3] = tab[np.argmax(tmp)]

        return sq

    def max_dimension(self, tr, tl, br, bl):
        width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        return min(int(width_top), int(width_bottom))

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        max_width = self.max_dimension(tr, tl, br, bl)
        max_height = self.max_dimension(tl, bl, tr, br)

        dst = np.array([[0, 0], [max_width-1, 0], [max_width-1, max_height-1], [0, max_height-1]], dtype="float32")
        matrix = cv2.getPerspectiveTransform(rect, dst)
        cut_image = cv2.warpPerspective(image, matrix, (max_width, max_height))
        return cut_image

    def extract_paper(self, image, org_image):
        edges = self.get_edges(image)
        cont = self.prepare_cont(image, edges)
        image = self.four_point_transform(org_image, cont.reshape(4, 2) * Utils.calc_image_ratio(org_image, 1000))
        return image

    def get_paper(self, file):
        image = cv2.imread(file)
        org_image = image.copy()
        name = os.path.splitext(os.path.basename(file))[0]
        print(name)
        prepared_img = Utils.resize_image(image, 1000)
        extracted_paper = self.extract_paper(prepared_img, org_image)
        return name, extracted_paper
