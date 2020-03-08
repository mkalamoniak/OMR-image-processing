from utils import Utils
import cv2
import numpy as np

class Staff:
    image = []
    image_copy = []
    index = 0
    x = 0
    y = 0
    h = 0
    w = 0
    staff_dist = 0
    first_line = 0
    notes_list = []

    def __init__(self, image, index, x, y, h, w, staff):
        self.image = image
        self.index = index
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.staff_dist = staff

    def get_first_line(self):
        return self.first_line

    def set_image_with_copy(self, image):
        self.image = image
        self.image_copy = image

    def set_notes_list(self, notes_list):
        self.notes_list = notes_list

    def get_notes_list(self):
        return self.notes_list

    def get_staff_list(self):
        return self.staff_dist

    def add_to_notes_list(self, tup):
        self.notes_list.append(tup)

    def clean_notes_list(self):
        self.notes_list = []

    def process_staff(self):
        img = self.image
        img_copy = Utils.bw_image(img)
        cnt_pix = [sum(tmp == 0) for tmp in img_copy]
        cnt_thresh = cnt_pix.copy()
        cnt_thresh.sort(reverse=True)
        cnt_thresh = cnt_thresh[:20]
        first_line = img.shape[1]
        for cnt in cnt_thresh:
            if cnt_pix.index(cnt) < first_line:
                first_line = cnt_pix.index(cnt)
        self.first_line = first_line

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)

        ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_TRUNC)
        ret, thresh = cv2.threshold(thresh, 150, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(thresh)
        thresh = cv2.medianBlur(thresh, 3)
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], np.uint8)

        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        self.set_image_with_copy(thresh)

    def find_notes(self):
        img = self.image
        _, thresh = cv2.threshold(img, 180, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, 1)
        areas = [cv2.contourArea(c) for c in contours]
        contours = [contours[a] for a in range(len(areas)) if (800 < areas[a] < 20000)]
        hierarchy = [hierarchy[:, a, :][0] for a in range(len(areas)) if (800 < areas[a] < 20000)]
        self.image_copy = cv2.cvtColor(self.image_copy, cv2.COLOR_GRAY2BGR)
        self.image_copy = cv2.drawContours(self.image_copy, contours, -1, (255, 0, 0), 3)
        for i in range(len(contours)):
            if hierarchy[i][3] == 0:
                x, y, w, h = cv2.boundingRect(contours[i])
                note = img[y:y + h, x:x + w]
                recognized_note, color, rev = self.recognize_note(note, x)
                if recognized_note == 'bass':
                    w = w + 30
                self.add_to_notes_list((recognized_note, x, y, h, w, color, rev))

    def recognize_note(self, note, x):
        image = note.copy()
        width = image.shape[1]
        height = image.shape[0]
        proportion = height / width
        cropped_image = image[5:height - 5, 5:width - 5]
        top = sum(cropped_image[0]) // 255
        bottom = sum(cropped_image[len(cropped_image) - 1]) // 255
        left = sum(cropped_image[:, 0]) // 255
        right = sum(cropped_image[:, len(cropped_image[0]) - 1]) // 255
        top_60_image = image[:int(0.6*height), :int(0.6*width)]
        tl_sum = len(top_60_image)*len(top_60_image[0]) - cv2.countNonZero(top_60_image)
        down_60_image = image[int(0.4 * height):height-1,int(0.4 * width):width-1]
        br_sum = len(down_60_image) * len(down_60_image[0]) - cv2.countNonZero(down_60_image)
        top_stripe = image[int(0.1 * height):int(0.25 * height)]
        b_prop = cv2.countNonZero(top_stripe)/ (len(top_stripe) * len(top_stripe[0]))
        down_stripe = image[int(0.75 * height):int(0.9 * height)]
        t_prop = cv2.countNonZero(down_stripe) / (len(down_stripe) * len(down_stripe[0]))
        lower_bound = 0.1
        upper_bound = 0.9
        bw_change_down = sum([int(0) if (image[int(lower_bound*height)][i] == image[int(lower_bound*height)][i+1]) else int(1) for i in range(len(image[int(lower_bound*height)])-1)])
        bw_change_top = sum([int(0) if (image[int(upper_bound*height)][i] == image[int(upper_bound*height)][i+1]) else int(1) for i in range(len(image[int(upper_bound*height)])-1)])
        eight_count = [i for i in range(len(image[int(upper_bound*height)])-1) if not (image[int(upper_bound*height)][i] == image[int(upper_bound*height)][i+1])]

        if height >= 0.8 * self.h:
            return 'violin', (0, 0, 255), 0
        elif height >= 0.3 * self.h and proportion >= 1.20 and proportion <= 1.8 and x < 200:
            return 'bass', (0, 255, 0), 0
        elif top < 0.7*(width-20) and  bottom < 0.7*(width-20) and left < 0.7*(height-20) and right < 0.7*(height-20) and 0.7 <= proportion <= 0.9:
            return 'full', (0, 255, 255), 0
        elif (tl_sum == 0 and t_prop < 0.3):
            return 'quarter', (255, 0, 0), 0
        elif (br_sum == 0 and b_prop < 0.3):
            return 'quarter', (255, 0, 0), 1
        elif (tl_sum == 0 and t_prop >= 0.3):
            return 'half', (255, 0, 255), 0
        elif (br_sum == 0 and b_prop >= 0.3):
            return 'half', (255, 0, 255), 1
        elif bw_change_down == 2 and bw_change_top == 2:
            if eight_count[1] < 25:
                return 'eight', (2, 61, 12), 1
            else:
                return 'eight', (2, 61, 12), 0
        else:
            return 'sharp', (0, 75, 150), 0
