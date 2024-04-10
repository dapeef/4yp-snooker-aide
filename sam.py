import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import find_edges
import torch

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# Drawing functions
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



def create_mask(image_file, input_points, input_labels, output_file=None):
    sam = initialise_sam()

    create_mask_from_model(sam, image_file, input_points, input_labels, output_file)

def initialise_sam():
    # SAM setup
    print("Loading SAM...")
    # sam_checkpoint = "checkpoints\\sam_vit_b_01ec64.pth" # For base model
    # model_type = "vit_b" # For base model
    sam_checkpoint = "./checkpoints/sam_vit_h_4b8939.pth" # For huge model
    model_type = "vit_h" # For huge model
    device = "cpu" # "cuda" #if access to cuda -> torch.cuda.is_available()
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)

    print("SAM loaded!")

    return sam

def create_mask_from_model(sam, image_file, input_points, input_labels, output_file=None):
    # Load image
    print("Loading image...")
    # image_file = "images\\snooker1.png"
    # image_file = "images\\snooker2.jpg"
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Show image
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('on')
    # plt.show()

    print("Image loaded!")


    # SAM predictor
    print("Embedding image...")
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    print("Image embedded!")

    # Prompts
    # #snooker1.png
    # input_points = np.array([[600, 600], [1300, 600], [1625, 855]])
    # input_labels = np.array([1, 1, 0]) # 1=foreground, 0=background
    # #snooker2.jpg
    # input_points = np.array([[2000, 1500]]) 
    # input_labels = np.array([1]) # 1=foreground, 0=background

    print("Creating masks...")
    show_points(input_points, input_labels, plt.gca())
    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )

    print("Masks created!")

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_points, input_labels, plt.gca())
        plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('on')

        # plt.figure()
        # plt.imshow(mask)

        # file = open("temp\\" + str(time.time()) + ".json", "w")
        if output_file is None:
            output_file = "temp\\" + str(time.time())
        
        np.savetxt(output_file, mask, fmt='%.0f')

    print("SAM finished!")

    # plt.show()

    # # Auto mask generator
    # mask_generator = SamAutomaticMaskGenerator(sam)
    # masks = mask_generator.generate(image)

    # print(masks)

    # print("Masks generated!")

    # # Show masks
    # plt.figure(figsize=(20,20))
    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('on')

    # plt.show()


if __name__ == "__main__":
    image_file = "images\\snooker1.png"
    input_points = np.array([[600, 600], [1300, 600], [1625, 855]])
    input_labels = np.array([1, 1, 0]) # 1=foreground, 0=background
    expected_corners = np.array([[494, 321], [1448, 321], [1618, 946], [305, 944]])

    mask_file = "./temp/mask.txt"
    sam_evaluator = initialise_sam()
    create_mask_from_model(sam_evaluator, image_file, input_points, input_labels, mask_file)
    lines = find_edges.get_sam_lines(mask_file)
    lines = lines[0]
    corners = find_edges.get_rect_corners(lines)

    # Draw image
    img = cv2.imread(image_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.title(image_file)
    a = plt.gca()
    a.imshow(img)

    print(corners)
    plt.show()