# RFSSD: Rich Feature Single Shot MultiBox Object Detector, in PyTorch
The model is trained on PASCAL VOC2007+ PASCAL VOC2012 with VGG-16. The input image is 300 \times 300. We reach an mAP of 78.9% on VOC2007 and 48.9% mAP (46.5% for SSD) for small object and 20.1% mAP (16.5% for SSD) for extra-small object.


## Evaluation
To test our model with VOC2007, simply run the command:

```shell
python evalour.py --size=x
```
where ```size``` can be selected from S(small), XS(extra-small), M(medium), L(large) and XL(extra-large). The default size parameter is to evaluate the general mAP of our model.

To compare SSD with our model, run the code below:

```shell
python evalssd.py --size=
```

## Our Work

### Parsing of Object Size Feature
We implemented parsing object size via ```process.py```, where we sort all the objects according to their absolute bbox size, and mark them as extra-small (XS: bottom 10%); small (S: next 20%);
medium (M: next 40%); large (L: next 20%); extra-large (XL: next 10%). Finally, we write this feature into the object attributes.

### Modification of SSD
We add DS_module1, DS_module2, DS_module3 for down-sampling operation, ConvBlock for typical 3 \times 3 conv. and IBN for batch norm operation. Furthermore, we combine the bottom-up scheme and top-down scheme with the original structure. The network structure are shown in our report. 

