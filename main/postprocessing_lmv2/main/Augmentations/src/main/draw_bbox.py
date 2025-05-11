import os
import cv2

path= "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Error_analysis_STP_generation/stp_verification_on_100_samples/certificate_of_origin"
image_name= "Packing_List_38_page_2.png"

def draw_bounding_box(img_path, labels_list):
    image = cv2.imread(img_path)
    # thresh = 255 - cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    for item in labels_list:
        label= item['label']
        x1= item['x1']
        y1= item['y1']
        x2= item['x2']
        y2= item['y2']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
        image,
        label,
        (int(x1), int(y1)),
        fontFace = cv2.FONT_HERSHEY_SIMPLEX,
        fontScale = 0.6,
        color = (255, 0, 0),
        thickness=2
    )

    return image

if __name__=="__main__":
    folder_path = "/home/ntlpt19/Downloads/MERGED_DATA/AIR_WAY/ROOT_AIR_WAY/AIR_WAY_AGUMENT"
    images_path = os.path.join(folder_path, "Images")
    labels_path = os.path.join(folder_path, "Labels")
    bounding_box_path= os.path.join(folder_path,"bounding_box")
    if not os.path.exists(bounding_box_path):
        os.mkdir(bounding_box_path)


    with open(os.path.join(folder_path, "label.txt"), "r") as f:
        classes = (f.read())
        classes = classes.split("\n")
    labelled_files = os.listdir(labels_path)
    labelled_files = [x.split(".txt")[0] for x in labelled_files]


    #using intersection percentage
    annotation_data = []
    thresh = 300
    for file in labelled_files:
        if os.path.exists(os.path.join(images_path, file + ".png")):
            print('yes')
            print(file)
            image = cv2.imread(os.path.join(images_path, file + ".png"))
            h, w, _ = image.shape
            with open(os.path.join(labels_path, file + ".txt"), "r") as f:
                label = (f.read())
            label = label.split("\n")
            labelled_data = []
            for l in label:
                l = l.split()
                if len(l) > 0:
                    l_class = classes[int(l[0])]
                    labelled_data.append({
                        "label": l_class,
                        "x1": int(l[1]),
                        "y1": int(l[2]),
                        "x2": int(l[3]),
                        "y2": int(l[4])
                    })

            print(labelled_data)
            img_path= os.path.join(images_path, file + ".png")
            processed_img= draw_bounding_box(img_path,labelled_data)
            bounding_box= os.path.join(bounding_box_path, file+ ".png")
            cv2.imwrite(bounding_box, processed_img)