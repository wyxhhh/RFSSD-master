import xml.etree.cElementTree as ET
import os
path = '/home/wangyixiong/pytorch/data/VOCdevkit/VOC2007/Annotations/'
dirs = os.listdir(path)
object_dic = {}
object_count = {}

iter = 0
for file in dirs:
    iter += 1
    tree = ET.parse('/home/wangyixiong/pytorch/data/VOCdevkit/VOC2007/Annotations/' + file)
    for obj in tree.findall('object'):
        name = obj.find('name').text
        for bndbox in obj.findall('bndbox'):
            area = (int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text)) * (int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text))
        if name not in object_count:
            object_count[name] = 1
            object_dic[name] = [area]
        else:
            object_count[name] += 1
            object_dic[name].append(area)
            

for name in object_dic:
    object_dic[name].sort()


iter = 0
for file in dirs:
# file = '000001.xml'
    iter += 1
    print(iter)
    tree = ET.parse('/home/wangyixiong/pytorch/data/VOCdevkit/VOC2007/Annotations/' + file)
    for obj in tree.findall('object'):
        name = obj.find('name').text
        for bndbox in obj.findall('bndbox'):
            area = (int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text)) * (int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text))

        position = object_dic[name].index(area) + 1
        position = position / object_count[name]
        if position < 0.1:
            type1 = 'XS'
        elif position < 0.3:
            type1 = 'S'
        elif position < 0.7:
            type1 = 'M'
        elif position < 0.9:
            type1 = 'L'
        else:
            type1 = 'XL'
        obj.set('type', type1)
        tree.write('/home/wangyixiong/pytorch/data/VOCdevkit/VOC2007/NEWAnnotations/' + file)