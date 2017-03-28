# chainer-realisticGAN
Chainer implementation for realisticGAN --- generate realistic images via redrawing mask parts.
## [Learning a Discriminative Model for the Perception of Realism in Composite Images](https://people.eecs.berkeley.edu/~junyanz/projects/realism/index.html) Revisited
### Download pretrained caffe model
1. Download [pretrained caffe model](https://people.eecs.berkeley.edu/~junyanz/projects/realism/realismCNN_models.zip)
2. Run `python load_caffe_model.py` to transform pretrained caffe model into **Chainer** model

### Predict image's realism
#### Step by Step
1. Download dataset [Realism Prediction Data](https://people.eecs.berkeley.edu/~junyanz/projects/realism/human_evaluation.zip)
2. Run `python mat2list_human_eval.py` to obtain image list & ground truth
3. Run `python predict_realism.py` to obtain prediction results. **AUC** score will be printed out, prediction score for each images will be stored in plain txt file

### Image Editing towards generating more realistic composited images
#### Step by Step
1. Download dataset [Color Adjustment Data](https://people.eecs.berkeley.edu/~junyanz/projects/realism/color_adjustment.zip)
2. Run `python mat2list_image_editing.py` to obtain image list
3. Run `python image_editing.py` to obtain more realistic images. (cut_and_paste image, generated image) will be save in result folder, as well as a plain file recording (cut_and_paste loss, generated loss)

## [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/) Revisited
### Neural Style Transfer
#### Step by Step
Run `python neural_style.py`

## FAQ
1. Run `python [SCRIPT_NAME].py -h` for more options