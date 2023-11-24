import os

files_dir = "data/Pockets, cushions, table - 2688 - B&W, rotated, mostly 9 ball/valid"

width = 244
height = 244

images_dir = files_dir + "/images"
labels_dir = files_dir + "/labels"

# sorting the images for consistency
# To get images, the extension of the filename is checked to be jpg
imgs = [image for image in sorted(os.listdir(images_dir)) if image[-4:]=='.jpg']

for i in range(len(imgs)):
    img_name = imgs[i]
    image_path = os.path.join(images_dir, img_name)

    # annotation file
    annot_filename = img_name[:-4] + '.txt'
    annot_file_path = os.path.join(labels_dir, annot_filename)

    boxes = []
    labels = []

    # box coordinates for xml files are extracted and corrected for image size given
    with open(annot_file_path) as f:
        for line in f:
            parsed = [float(x) for x in line.split(' ')]
            labels.append(parsed[0])
            x_center = parsed[1]
            y_center = parsed[2]
            box_wt = parsed[3]
            box_ht = parsed[4]

            xmin = x_center - box_wt/2
            xmax = x_center + box_wt/2
            ymin = y_center - box_ht/2
            ymax = y_center + box_ht/2
            
            xmin_corr = int(xmin*width)
            xmax_corr = int(xmax*width)
            ymin_corr = int(ymin*height)
            ymax_corr = int(ymax*height)
            
            if xmax_corr - xmin_corr == 0 or ymax_corr - ymin_corr == 0:
                print("YIKES", img_name)
                os.unlink(image_path)
                os.unlink(annot_file_path)