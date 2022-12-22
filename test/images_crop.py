import os
from PIL import Image

inputPath = "data/nerf_llff_data/apple/images"
imgs=[]
for i in os.listdir(inputPath):
    if (i.endswith('.jpg')):
        im=Image.open(inputPath+'/'+i)
        imgs.append(im)

outputDir=inputPath+'/'+'outputDir'
if(os.path.exists(outputDir)!=True):
    os.makedirs(outputDir)
    print(outputDir+"文件夹创建成功！")
else:
    print(outputDir+"文件夹存在！")

num=1
for img in imgs:
    print('第 '+str(num)+' 张图片 —> '+": size="+str(img.size))
    cropped = img.crop((0, 0, 4096, 3072))  # (left, upper, right, lower)
    cropped.save(outputDir+'/'+str(num)+'.jpg')
    num+=1
