# chainer-realismCNN
Chainer implementation for realismCNN proposed in [Learning a Discriminative Model for the Perception of Realism in Composite Images](https://people.eecs.berkeley.edu/~junyanz/projects/realism/index.html)

## Download pretrained caffe model
1. Download [pretrained caffe model](https://people.eecs.berkeley.edu/~junyanz/projects/realism/realismCNN_models.zip)
2. Run `python load_caffe_model.py` to transform pretrained caffe model into **Chainer** model

## Predict image's realism
### Step by Step
1. Download dataset [Realism Prediction Data](https://people.eecs.berkeley.edu/~junyanz/projects/realism/human_evaluation.zip)
2. Run `python mat2list_human_eval.py` to obtain image list & ground truth
3. Run `python predict_realism.py` to obtain prediction results. **AUC** score will be printed out, prediction score for each image will be stored in plain text file

## Image Editing towards generating more realistic composited images
### Step by Step
1. Download dataset [Color Adjustment Data](https://people.eecs.berkeley.edu/~junyanz/projects/realism/color_adjustment.zip)
2. Run `python mat2list_image_editing.py` to obtain image list
3. Run `python image_editing.py` to obtain more realistic images. (cut_and_paste image, generated image) will be saved in the `result` folder,and a plain file will be generated recording (cut_and_paste loss, generated loss) for each image.

## NOTE
Run `python [SCRIPT_NAME].py -h` for more options。
