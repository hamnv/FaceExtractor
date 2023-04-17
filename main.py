from facenet_pytorch import MTCNN
import torch
import os
from PIL import Image
import cv2
import torch.multiprocessing as mp
from tqdm import tqdm
import heapq
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

from transformers import pipeline

pipe_aesthetic = pipeline("image-classification", "cafeai/cafe_aesthetic")


#@markdown The directory of images to process.
image_dir = 'input' #@param {type:"string"}
#@markdown The directory images are saved to
output_dir = 'output' #@param {type:"string"}
#@markdown Minimum detected face size
min_face_size = 40 #@param {type:"integer"}

#@markdown Margin around face
margin = 60 #@param {type:"integer"}


#@markdown Choose crop type:
#@markdown - **(This one is probably good enough to use)** `square_crop` crops to smallest square containing face. It adds `margin` px around, and if that square is still smaller than `minimum_length`, it crops to a square of `minimum_length` px. (Some images will be thrown out if the crop goes up against the image's bounds.)
#@markdown - `square_crop_tight` crops to smallest square containing face. It adds `margin` px around, then only saves images whose dimensions are at least `minimum_length` px. This will throw out more images than `square_crop`.
#@markdown - `face_crop` crops to rectangle containing face. It adds `margin` px around, and if that rectangle is still smaller than `minimum_length`, it crops to a rectangle of at least `minimum_length` px. 
#@markdown - `face_crop_tight` crops to rectangle containing face. It adds `margin` px around, then only saves images whose dimensions are at least `minimum_length` px. This is the tightest crop and will throw out the most images.

crop_type = "square_crop" #@param ["square_crop", "square_crop_tight", "face_crop", "face_crop_tight"]

#@markdown If using `square_crop`, `top-center` centers the face horizontally and aligns it to the top (with `margin` px above the face), while `fully_centered` centers both vertically and horizontally
face_location = "top_center" #@param ["top_center", "fully_centered"]

#@markdown Minimum size of crop
minimum_length = 512 #@param {type:"integer"}
#@markdown Display images while running. Shows both the original and face-cropped version of the image.
display_images = False #@param {type:"boolean"}
#@markdown Draw boxes around detected faces when displaying images
draw_face_box = False #@param {type:"boolean"}
#@markdown Save cropped and/or original images of faces that meet the threshold to `output_dir`
save_cropped_images = True #@param {type:"boolean"}
save_original_images = False #@param {type:"boolean"}
#@markdown Score by `facenet face probability * (cafe aesthetic score)^2`. Slower (on the order of s/it instead of it/s), but should result in better quality. If not activated, then just score by `facenet face probability`.
score_with_aesthetic = False #@param {type:"boolean"}
#@markdown After going through all the images, save up to `n` faces with the highest score.
n = 200 #@param {type:"integer"}
#@markdown Show print statements
verbose = False #@param {type:"boolean"}

detected_num = 0
processed_num = 0
face_probs = []


def crop_face(img, face):
    x1, y1, x2, y2 = [int(val) for val in face]
    if crop_type == "face_crop" or crop_type == "face_crop_tight":
      # Add a margin to the face bounding box
      x_min = max(0, x1 - margin)
      y_min = max(0, y1 - margin)
      x_max = min(img.shape[1], x2 + margin)
      y_max = min(img.shape[0], y2 + margin)
      if crop_type == "face_crop" and (y_max - y_min < minimum_length or x_max - x_min < minimum_length):
          if x_max - x_min < y_max - y_min:
            new_width = minimum_length
            new_height = minimum_length * (y_max - y_min) // (x_max - x_min)
          else:   
            new_width = minimum_length * (x_max - x_min) // (y_max - y_min)
            new_height = minimum_length
          x_center = (x_min + x_max) // 2
          y_center = (y_min + y_max) // 2
          x = int(x_center - new_width/2)
          y = int(y_center - new_height/2)
          x_min = max(0, x)
          y_min = max(0, y)
          x_max = min(img.shape[1], x + new_width)
          y_max = min(img.shape[0], y + new_height)
    else:
      # Calculate the dimensions of the smallest square that contains the entire face
      height = y2 - y1
      width = x2 - x1
      x_center = (x1 + x2) // 2
      y_center = (y1 + y2) // 2
      face_size = max(width, height)
      x_min = max(0, x_center - face_size // 2 - margin)
      y_min = max(0, y_center - face_size // 2 - margin)
      x_max = min(img.shape[1], x_center + face_size // 2 + margin)
      y_max = min(img.shape[0], y_center + face_size // 2 + margin)
      if crop_type == "square_crop" and (y_max - y_min < minimum_length or x_max - x_min < minimum_length):
        x = int(x_center - minimum_length/2)
        y = int(y_center - minimum_length/2)
        x_min = max(0, x)
        y_min = max(0, y1 - margin) if face_location == "top_center" else max(0, y)
        x_max = min(img.shape[1], x + minimum_length)
        y_max = min(img.shape[0], y1 - margin + minimum_length) if face_location == "top_center" else min(img.shape[0], y + minimum_length)
      if y_max - y_min != x_max - x_min:
        size = min(y_max - y_min, x_max - x_min)
        x_center = x_min + (x_max - x_min) // 2
        y_center = y_min + (y_max - y_min) // 2
        x_min = max(0, x_center - size // 2)
        y_min = max(0, y_center - size // 2)
        x_max = min(img.shape[1], x_center + size // 2)
        y_max = min(img.shape[0], y_center + size // 2)
      cropped_face = img[y_min:y_max, x_min:x_max]
    return cropped_face

pipe_aesthetic = pipeline("image-classification", "cafeai/cafe_aesthetic")

def get_aesthetic_score(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image)
    data = pipe_aesthetic(pil_img)
    aesthetic_score = data[0]["score"]
    return aesthetic_score

def process_image(filename):
    global detected_num
    global processed_num
    global face_probs
    # Load image
    img = cv2.imread(os.path.join(image_dir, filename))
    detected_face = False
    if verbose: print(f'{filename} is being processed')

    # Detect face
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=True, 
                  select_largest=False, 
                  device=device,
                  min_face_size=min_face_size)
    boxes, probs = mtcnn.detect(img)
    # Draw rectangle around each face
    if draw_face_box and boxes is not None:
      for box in boxes:
          x1, y1, x2, y2 = box.astype(int)
          shownimg = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # Display original image with rectangles around each face
    if display_images:
        height, width = img.shape[:2]
        new_height = int(height * (400 / width))
        shownimg = cv2.resize(img, (400, new_height))
        cv2_imshow(shownimg)
        if verbose: print(f'Original image')

    # Filter out faces with low probability
    if boxes is None:
      if verbose: print(f'No face found in {filename}')
    else:
      for i, prob in enumerate(probs):
        detected_face = True
        # Crop face
        img_cropped = crop_face(img, boxes[i])
        
        # Display cropped image
        if display_images:
          cv2_imshow(img_cropped)

        if score_with_aesthetic:
          aesthetic_score = get_aesthetic_score(img_cropped)
          overall_score = pow(aesthetic_score, 2) * prob
        else:
          overall_score = prob
        if verbose: 
          print(f'Detected face {i+1} is cropped ({crop_type}) to dimensions {img_cropped.shape[1]}x{img_cropped.shape[0]}') 
          print(f'Detected face {i+1} has confidence: {prob:.5f}') 
          if score_with_aesthetic:
            print(f'Detected face {i+1} has aesthetic score {aesthetic_score:.5f}') 
            print(f'Detected face {i+1} has overall score {overall_score:.5f}')
        # Add filename, associated error (ie 1-overall_score, since heap sorts by smallest) and box to face_probs list
        heapq.heappush(face_probs, (1-overall_score, filename, boxes[i]))
    if detected_face: detected_num += 1
    processed_num += 1
    if verbose: print(f'Detected face in {detected_num} images out of {processed_num} images ({detected_num/processed_num:.2%})')

if __name__ == '__main__':
    filenames = [filename for filename in os.listdir(image_dir) if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png')]
    print(f'Processing {len(filenames)} images...')
    if verbose:
      for filename in filenames:
        process_image(filename)
    else:
      with tqdm(total=len(filenames)) as pbar:
        for filename in filenames:
          process_image(filename)
          pbar.update()
  
    print(f'Detected face in {detected_num} images out of {processed_num} images ({detected_num/processed_num:.2%})')

    if save_cropped_images or save_original_images:
      i = 0
      images_skipped = 0
      while face_probs and i < n:
        image_saved = False
        # here, error = 1-prob
        error, filename, box = heapq.heappop(face_probs)
        img = cv2.imread(os.path.join(image_dir, filename))
        if save_cropped_images:
          img_cropped = crop_face(img, box)
          if img_cropped.shape[0] >= minimum_length and img_cropped.shape[1] >= minimum_length:
            output_filename = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}-face{i+1}{os.path.splitext(filename)[1]}')
            cv2.imwrite(output_filename, img_cropped)
            image_saved = True
            if verbose: print(f'Cropped image of face {i+1} (score {1-error:.5f}) saved at {output_filename}')
          else: 
            if verbose: print(f'Detected face {i+1} is too small - {filename}, skipped saving image')
        if save_original_images:
            output_filename = os.path.join(output_dir, filename)
            cv2.imwrite(output_filename, img)
            image_saved = True
            if verbose: print(f'Original image saved at {output_filename}')
        # If no image is saved because it's skipped by crop_type, move to the next image in the heap
        if image_saved: 
          i += 1
        else:
          images_skipped += 1
      print(f'Saved {i} images with highest score ({images_skipped} images skipped)') 
