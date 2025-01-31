import torch
import requests
from PIL import Image
import os

from diffusers import StableDiffusionDepth2ImgPipeline
import sys
from ultralytics import YOLO
import cv2
ori_direct = './yolov8_face/examples/original'
ori_path = [(os.path.join(ori_direct, f)) for f in os.listdir(ori_direct)]
model = YOLO("yolov8s.pt")
ori_path = sorted(ori_path)

pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
   "stabilityai/stable-diffusion-2-depth",
   torch_dtype=torch.float16,
).to("cuda")


prompt = "animation, cartoon, beautiful woman/handsome man, (high detailed skin:1.2), pale skin, Intricate, High Detail, Sharp focus, dramatic, proportional body, beautiful/handsome eyes, toned body"
n_propmt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch,  drawing), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

for i in range(len(ori_path)):
    img_path = str(ori_path[i])
    results = model(img_path)

    img = cv2.imread(img_path)
    
    boxes = results[0].boxes
    pair = []

    for box in boxes:
        xyxy = box.xyxy.cpu().detach().numpy().tolist()  # box with xyxy format, (N, 4)
        label = box.cls.cpu().detach().numpy().tolist()    # cls, (N, 1)
        pair.append([label, xyxy[0]])
        
    human = []
    for k in pair:
        label = int(k[0][0])
        if label == 0:
            value = k[1]
            human.append(value)

    save = './part'
    for k in range(len(human)):
        coor = human[k]
        save_path = save  + '/part_' + str(k) +'.png'
        print(save_path)
        x_min = coor[0]
        y_min = coor[1]
        x_max = coor[2]
        y_max = coor[3]

        x_ran = x_max - x_min
        y_ran = y_max - y_min

        crop = img[int(y_min):int(y_min+y_ran), int(x_min):int(x_min+x_ran)]
        cv2.imwrite(save_path, crop)

    paths = [(os.path.join(save, f)) for f in os.listdir(save)]
    paths = sorted(paths)
#    print(paths)

    for i in range(len(paths)):
        init_image_ = Image.open(str(paths[i])).convert("RGB")
       #print(os.getcwd())
        image = pipe(prompt=prompt, image=init_image_ , negative_prompt=n_propmt).images[0]
        save_path = "./Af_stabel/" + "sd_result_"+ str(i) + ".png"
        image.save(save_path) 


    part_dir= './Af_stabel'
    paths = [(os.path.join(part_dir, f)) for f in os.listdir(part_dir)]
    paths = sorted(paths)

    for i in range(len(paths)):
        part = cv2.imread(str(paths[i]))
        coor = human[i]
        x_min = coor[0]
        y_min = coor[1]

        x,y,c = part.shape

        img[int(y_min):int(y_min + x), int(x_min):int(x_min+y)] = part

    yolo_path = './yolov8_face/examples/after/' + str(0) + '.png'
    cv2.imwrite(yolo_path, img)
