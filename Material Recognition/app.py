import os
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import cv2

from segment import segment, color_map
from retrieve import retrieve
from encode import encode

# Dimension of the image
WIDTH = 512
HEIGHT = 512
lasx = 0
lasy = 0
lines = []
image = None
image_np = None
image_segmented = None
image_segmented_np = None
real_area = None
crop_method = 'all'
cmap = color_map()

# Data directory
data_dir = './datasets/data'

# Directory for encoded dataset in the form of .pkl files
encoded_dir = './encoded'

num_query = 10

# Directory to save the list of similar materials
output_dir = './results'
output_path = os.path.join(output_dir, 'top{}.txt'.format(num_query))
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

pickle_path = './encoded/features.pkl'



img_size = 512

app = Tk()
app.title('Material Recognizer')
app.geometry('700x850')
title = Label(app, text='Choose area to recognize', font='arial 30 bold', fg='#068481')
title.pack()


# Open an image
def openAndPut():
    global image, image_np, real_area
    path = filedialog.askopenfilename()
    if path:
        image = Image.open(path)
        if '.png' in path:
            image = image.convert('RGB')
        image = image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        image_np = real_area = np.asarray(image)
        image = ImageTk.PhotoImage(image)
        image_area.create_image((0, 0), image=image, anchor='nw')


mask = np.ones((WIDTH, HEIGHT))


def get_x_and_y(event):
    global lasx, lasy
    lasx, lasy = event.x, event.y


def draw_smth(event):
    global lasx, lasy, lines
    lines.append(image_area.create_line((lasx, lasy, event.x, event.y), fill='red', width=2))
    lasx, lasy = event.x, event.y

    if img_size > lasx >= 0 and img_size > lasy >= 0:
        mask[lasy][lasx] = 0
        mask[lasy + 1][lasx + 1] = 0
        mask[lasy + 1][lasx - 1] = 0
        mask[lasy - 1][lasx + 1] = 0
        mask[lasy - 1][lasx - 1] = 0


def select_area_lasso():
    global lines, mask, crop_method
    crop_method = 'lasso'
    image_area.create_image((0, 0), image=image, anchor='nw')

    mask[:][:] = 1
    for line in lines:
        image_area.delete(line)
    image_area.bind("<Button-1>", get_x_and_y)
    image_area.bind("<B1-Motion>", draw_smth)


def draw_box(event):
    global lasx, lasy, lines
    for line in lines:
        image_area.delete(line)
    lines.append(image_area.create_rectangle(lasx, lasy, event.x, event.y, outline='red', width=2))


def crop_box(event):
    global image_np, real_area
    x, y = event.x, event.y
    if img_size > lasx >= 0 and img_size > x >= 0 and img_size > lasy >= 0 and img_size > y >= 0:
        real_area = image_np[min(y, lasy):max(y, lasy), min(x, lasx):max(x, lasx)]


def select_area_box():
    global lines, crop_method
    crop_method = 'box'
    image_area.create_image((0, 0), image=image, anchor='nw')

    for line in lines:
        image_area.delete(line)

    image_area.bind("<Button-1>", get_x_and_y)
    image_area.bind("<B1-Motion>", draw_box)
    image_area.bind("<ButtonRelease-1>", crop_box)


def crop_segment(event):
    global mask, image_segmented_np, image_np, real_area
    x, y = event.x, event.y
    selected_pixel = image_segmented_np[y, x]
    mask = np.where(image_segmented_np == selected_pixel, 1, 0).astype(np.uint8)
    mask_3d = np.repeat(mask[:, :, None], 3, axis=2)
    real_area = cv2.multiply(image_np, mask_3d)
    Image.fromarray(real_area).show()


def select_area_segment():
    global lines, mask, crop_method, image_segmented, image_segmented_np
    crop_method = 'segment'
    transparency = 0.5

    for line in lines:
        image_area.delete(line)

    image_segmented_np = segment(image_np, img_size)

    foreground = cv2.multiply(cmap[image_segmented_np].astype(float), np.full((img_size, img_size, 3), transparency))
    background = cv2.multiply(image_np.astype(float), np.full((img_size, img_size, 3), 1.0 - transparency))
    image_segmented = cv2.add(foreground, background).astype(np.uint8)
    image_segmented = np.clip(0, 255, image_segmented)

    image_segmented = ImageTk.PhotoImage(Image.fromarray(image_segmented))
    image_area.create_image((0, 0), image=image_segmented, anchor='nw')

    Image.fromarray(image_np).show()
    image_area.bind("<Button-1>", crop_segment)


def return_shape(image_in):
    new_img = image_in
    cv2.imshow("image in", image_in)
    cv2.waitKey(0)
    edged = cv2.Canny(image_in, 30, 200)
    cv2.imshow("edged", edged)
    cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(new_img, contours, -1, (0, 0, 0), 3)
    cv2.imshow("new img", new_img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    th, im_th = cv2.threshold(new_img, 200, 255, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask_floodfill = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask_floodfill, (0, 0), (255, 255, 255))
    im_floodfill = np.abs(im_floodfill - np.ones((WIDTH, HEIGHT)) * 255)
    return im_floodfill


def show_mask():
    global real_area

    if crop_method == 'lasso':
        mask_3_channels = np.ones((WIDTH, HEIGHT, 3))
        image_matte = (mask * 255).astype(np.uint8)
        the_real_mask = return_shape(image_matte)
        mask_3_channels[:, :, 0] = the_real_mask / 255
        mask_3_channels[:, :, 1] = the_real_mask / 255
        mask_3_channels[:, :, 2] = the_real_mask / 255

        real_area = image_np * mask_3_channels
        real_area = Image.fromarray(np.uint8(real_area)).convert('RGB')
        real_area.show()

    else:
        real_area = Image.fromarray(np.uint8(real_area)).convert('RGB')
        real_area.show()


def recognize():
    img = np.array(real_area)
    results = retrieve(img=img,
                       data_dir=data_dir,
                       output_dir=output_path,
                       pickle_path=pickle_path if os.path.exists(pickle_path) else None)

    for i, result in enumerate(results):
        cv2.imshow('result {}'.format(i), cv2.resize(cv2.imread(result[1]), (WIDTH, HEIGHT)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    with open(output_path, 'w') as f:
        for result in results:
            f.write("Name: {}, Score: {} \n".format(result[1].split('/')[-1], result[0]))



def dataset_prep():
    encode(data_dir, encoded_dir, save_pkl=True)


image_area = Canvas(app, width=WIDTH, height=HEIGHT, bg='#C8C8C8')
image_area.pack(pady=(10, 0))

open_image = Button(app, width=20, text='OPEN IMAGE', font='none 12', command=openAndPut)
open_image.pack(pady=(10, 5))

crop_area_lasso = Button(app, width=20, text='SELECT AREA - LASSO', font='none 12', command=select_area_lasso)
crop_area_lasso.pack(pady=(0, 5))

crop_area_box = Button(app, width=20, text='SELECT AREA - Box', font='none 12', command=select_area_box)
crop_area_box.pack(pady=(0, 5))

segment_area = Button(app, width=20, text='SELECT AREA - SEGMENT', font='none 12', command=select_area_segment)
segment_area.pack(pady=(0, 5))

show_area = Button(app, width=20, text='SHOW AREA', font='none 12', command=show_mask)
show_area.pack(pady=(0, 5))

recognize = Button(app, width=30, text='RECOGNIZE MATERIAL', font='none 12', command=recognize)
recognize.pack(pady=(0, 5))

prepare_database = Button(app, width=20, text='PREPARE DATASET', font='none 12', command=dataset_prep)
prepare_database.pack(pady=(0, 5))

app.mainloop()





