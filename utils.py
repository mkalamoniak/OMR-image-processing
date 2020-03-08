from skimage.filters import threshold_local
from collections import Counter
import numpy as np
import cv2
import imutils


class Utils:
    @staticmethod
    def calc_image_ratio(image, new_height):
        return image.shape[0] / float(new_height)

    @staticmethod
    def get_image_copy(image):
        return image.copy()

    @staticmethod
    def resize_image(image, new_height):
        return imutils.resize(image, height=new_height)

    @staticmethod
    def show_image(title, image, height=1000):
        cv2.imshow(title, imutils.resize(image, height=height))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def bw_image(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = threshold_local(image, 21, offset=10, method="mean")
        image = (image > thresh).astype("uint8") * 255

        kernel = np.array([[1, 0, 0, 0, 1],
                           [0, 1, 0, 1, 0],
                           [0, 0, 1, 0, 0],
                           [0, 1, 0, 1, 0],
                           [1, 0, 0, 0, 1]]).astype("uint8")
        image = cv2.morphologyEx(image, cv2.MORPH_ELLIPSE, kernel)
        image = image.astype(np.uint8)
        return image

    @staticmethod
    def measure_staff(image):
        image = Utils.bw_image(image)
        cnt_pix = [sum(tmp == 0) for tmp in image]
        cnt_thresh = cnt_pix.copy()
        cnt_thresh.sort(reverse=True)
        min_cnt = cnt_thresh[60]
        sum_list = [(0, (cnt_pix[0] + cnt_pix[1]) // 2), (1, (cnt_pix[0] + cnt_pix[1] + cnt_pix[2]) // 3)]
        for i in range(2, len(cnt_pix) - 1):
            sum_list.append((i, sum_list[i - 1][1] - cnt_pix[i - 2] // 3 + cnt_pix[i + 1] // 3))
        final_list = [(i, j) for i, j in sum_list if j > min_cnt]
        distances = []
        zx, _ = final_list[0]
        for i in range(1, len(final_list)):
            x, _ = final_list[i]
            distances.append(abs(zx - x))
            zx = x
        distances.sort()
        distances = list(filter((1).__ne__, distances))
        cont = Counter(distances)
        cont, _ = cont.most_common(1)[0]
        return cont
