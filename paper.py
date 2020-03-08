import cv2
from utils import Utils
from staff import Staff


class Paper:
    image = []
    name = ''
    org_image = None
    staff_imgs = []

    def set_new_image(self, name, image):
        self.name = name
        self.image = image.copy()
        self.org_image = image.copy()

    def add_staff(self, image, index, x, y, h, w, staff):
        self.staff_imgs.append(Staff(image, index, x, y, h, w, staff))

    def get_staff_imgs(self):
        return self.staff_imgs

    def remove_staffs(self):
        del self.staff_imgs[:]
        self.staff_imgs = []

    def cut_staff(self):
        staff = Utils.measure_staff(self.image)
        img = self.image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.remove_staffs()
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 0)
        contours, hierarchy = cv2.findContours(thresh, 2, 1)
        areas = [cv2.contourArea(c) for c in contours]
        width = img.shape[1]
        contours = [contours[a] for a in range(len(areas)) if (2 * staff * width < areas[a] < 6 * staff * width)]
        print('contours: ', len(contours))
        for i in range(len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
            new_img = self.image[y - 10:y + h + 10, x - 10:x + w + 10]
            self.add_staff(new_img, i, x-10, y-10, h, w, staff)

    def text_mark_note(self, line, clef, before):
        line = line + 0.5
        text = ''
        if clef == 'bass':
            if line < 0.75:
                text = 'B3'
            elif line < 1.25:
                text = 'A3'
            elif line < 1.75:
                text = 'G3'
            elif line < 2.25:
                text = 'F#3'
            elif line < 2.75:
                text = 'E3'
            elif line < 3.25:
                text = 'D3'
            elif line < 3.75:
                text = 'C3'
            elif line < 4.25:
                text = 'B2'
            elif line < 4.75:
                text = 'A2'
            elif line < 5.25:
                text = 'G2'
            elif line < 5.75:
                text = 'F#2'
        elif clef == 'violin':
            if line < 0.75:
                text = 'G5'
            elif line < 1.25:
                text = 'F#5'
            elif line < 1.75:
                text = 'E5'
            elif line < 2.25:
                text = 'D5'
            elif line < 2.75:
                text = 'C5'
            elif line < 3.25:
                text = 'B4'
            elif line < 3.75:
                text = 'A4'
            elif line < 4.25:
                text = 'G4'
            elif line < 4.75:
                text = 'F#4'
            elif line < 5.45:
                text = 'E4'
            elif line < 5.75:
                text = 'D4'
        if before == 'sharp':
            pass
        return text

    def process_all(self):
        for staff in self.staff_imgs:
            staff.clean_notes_list()
            staff.process_staff()
            staff.find_notes()
            first_line = staff.get_first_line()
            print(first_line)
            notes_list = staff.get_notes_list()
            notes_list.sort(key=lambda tup: tup[1])
            clef = notes_list[0][0]
            for i, (note, x, y, h, w, color, rev) in enumerate(notes_list):
                if rev == 1:
                    line = (y - first_line)/staff.staff_dist + 1
                else:
                    line = (y + h - first_line)/staff.staff_dist
                text = note
                if note not in ['bass', 'violin', 'sharp']:
                    text = self.text_mark_note(line, clef, notes_list[i-1][0]) + " " + note
                self.org_image = cv2.putText(self.org_image, text, (staff.x+x, staff.y+y), color=color,
                                              fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3, thickness=2)
                self.org_image = cv2.rectangle(self.org_image, (staff.x+x, staff.y+y),
                                                (staff.x+x + w, staff.y+y + h), color, thickness=2)
