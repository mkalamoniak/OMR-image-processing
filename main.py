from image import Image
from paper import Paper
from utils import Utils
from glob import glob
import os


def main():
    path = os.getcwd()
    files = glob(path + '/new_input/*.jp*')
    image = Image()
    paper = Paper()
    for file in files:
        name, extracted_paper = image.get_paper(file)
        paper.set_new_image(name, extracted_paper)
        paper.cut_staff()
        paper.process_all()
        Utils.show_image('Result', Paper.org_image)


if __name__ == '__main__':
    main()
