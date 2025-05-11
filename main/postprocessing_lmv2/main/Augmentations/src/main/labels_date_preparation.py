import os
import cv2

def rotate90Deg(bndbox, img_width): # just passing width of image is enough for 90 degree rotation.
   x_min,y_min,x_max,y_max = bndbox
   new_xmin = y_min
   new_ymin = img_width-x_max
   new_xmax = y_max
   new_ymax = img_width-x_min
   return [new_xmin, new_ymin,new_xmax,new_ymax]


def rotate_90Deg( bndbox , image_width ):
    """
    image_width: Width of the image after clockwise rotation of 90 degrees
    """
    x_min,y_min,x_max,y_max = bndbox
    new_xmin = image_width - y_max # Reflection about center X-line
    new_ymin = x_min
    new_xmax = image_width - y_min # Reflection about center X-line
    new_ymax = x_max
    return [new_xmin, new_ymin,new_xmax,new_ymax]

if __name__=="__main__": 
   main_labels = '/media/tarun/D3/TradeFinance/final_delivery/lc_cancellation_test/Labels'
   labels_path = '/media/tarun/D3/TradeFinance/final_delivery/lc_cancellation_test/new_labels'
   root_path = '/media/tarun/D3/TradeFinance/final_delivery/lc_cancellation_test/OrigImages'
   label_files = os.listdir(main_labels)

   for i in label_files:
      img_files = i.split(".txt")[0]
      print("img files: ", img_files)
      image_path = os.path.join(root_path, f'{img_files}.png')
      print(image_path)

      image = cv2.imread(image_path)
      h, w, _ = image.shape
      with open(os.path.join(main_labels, i), "r") as f:
          label = (f.read())
      label = label.split("\n")

      for l in label:
          l = l.split()
          if len(l) > 0:
              l_class = int(l[0])
              x_center = float(l[1]) * w
              y_center = float(l[2]) * h
              width = float(l[3]) * w
              height = int(float(l[4]) * h)
              x0 = int(x_center - (width/2))
              x1 = int(x_center + (width/2))
              y0 = int(y_center - (height / 2))
              y1 = int(y_center + (height / 2))    


              new_x1, new_y1, new_x2, new_y2 = rotate_90Deg([x0, y0, x1, y1], h) #right

              normalized_values = [l_class, new_x1, new_y1, new_x2, new_y2] 
              with open(os.path.join(labels_path,img_files+'r'+'.txt'), "a") as file:
                  # Write the normalized values separated by spaces
                  file.write(" ".join(map(str, normalized_values)) + "\n")  


              new_x1, new_y1, new_x2, new_y2 = rotate90Deg([x0, y0, x1, y1], w) #left

              normalized_values = [l_class, new_x1, new_y1, new_x2, new_y2] 
              with open(os.path.join(labels_path,img_files+'l'+'.txt'), "a") as file:
                  # Write the normalized values separated by spaces
                  file.write(" ".join(map(str, normalized_values)) + "\n")              


              new_x1, new_y1, new_x2, new_y2 = rotate90Deg([x0, y0, x1, y1], w) #inverse
              new_x1, new_y1, new_x2, new_y2 = rotate90Deg([new_x1, new_y1, new_x2, new_y2], h)
              normalized_values = [l_class, new_x1, new_y1, new_x2, new_y2] 
              with open(os.path.join(labels_path,img_files+'i'+'.txt'), "a") as file:
                  # Write the normalized values separated by spaces
                  file.write(" ".join(map(str, normalized_values)) + "\n")              




              normalized_values = [l_class, x0, y0, x1, y1] #black and white
              with open(os.path.join(labels_path,img_files+'bw'+'.txt'), "a") as file:
                  # Write the normalized values separated by spaces
                  file.write(" ".join(map(str, normalized_values)) + "\n")  



              normalized_values = [l_class, x0, y0, x1, y1] #original
              with open(os.path.join(labels_path,img_files+'.txt'), "a") as file:
                  # Write the normalized values separated by spaces
                  file.write(" ".join(map(str, normalized_values)) + "\n")
                    
            
            
          
                     