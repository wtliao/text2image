# text2image
This repository includes the implementation for [Text to Image Generation with Semantic-Spatial Aware GAN](https://arxiv.org/abs/2104.00567)

This repo is not completely.

## Network Structure
![network_structure](./figures/framework.png)

The structure of the spatial-semantic aware convolutional network (SSACN) is shown as below

![ssacn](./figures/ssacn.png)

## Requirements
* python 3.6+
* pytorch 1.0+
* numpy
* matplotlib
* opencv

Or install full requirements by running:
```bash
pip install -r requirements.txt
```

## TODO
- [x] instruction to prepare dataset
- [ ] remove all unnecessary files
- [x] add link to download our pre-trained model
- [ ] clean code including comments
- [ ] instruction for training
- [ ] instruction for evaluation

## Prepare data
1. Download the preprocessed metadata for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`

## Pre-trained text encoder
1. Download the [pre-trained text encoder](https://drive.google.com/open?id=1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V) for CUB and save it to `DAMSMencoders/bird/inception/`
2. Download the [pre-trained text encoder](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ) for coco and save it to `DAMSMencoders/coco/inception/`

## Trained model
you can download our trained models from our [onedrive repo](https://1drv.ms/u/s!At2RxWvE6z1zgvEoglA1flTSnHZqHg?e=TtJNEW)

## Start training


See `opts.py` for the options.


## Evaluation



## Performance
You will get the scores close to below after training under xe loss for xxxxx epochs:

![results](./figures/results.png)

## Qualitative Results
Some qualitative results on coco and birds dataset from different methods are shown as follows:
![qualitative_results](./figures/qualitative.png)

The predicted mask maps on different stages are shown as as follows:
![mask](./figures/mask.png)

## Reference

If you find this repo helpful in your research, please consider citing our paper:

```
@article{liao2021text,
  title={Text to Image Generation with Semantic-Spatial Aware GAN},
  author={Liao, Wentong and Hu, Kai and Yang, Michael Ying and Rosenhahn, Bodo},
  journal={arXiv preprint arXiv:2104.00567},
  year={2021}
}
```
The code is released for academic research use only. For commercial use, please contact [Wentong Liao](http://www.tnt.uni-hannover.de/en/staff/liao/).

## Acknowledgements

This implementation borrows part of the code from [DF-GAN](https://github.com/tobran/DF-GAN).