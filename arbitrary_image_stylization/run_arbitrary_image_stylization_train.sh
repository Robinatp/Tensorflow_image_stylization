echo "input create or train :"
read a
echo "input is $a"

STYLE_IMAGES_PATHS=./tmp/dtd/dtd/images
#STYLE_IMAGES_PATHS=./tmp/dtd/dtd/images/pleated/*.jpg
RECORDIO_PATH=./tmp/tfrecord/style_images.tfrecord
if [ $a = create ] ; then
python image_stylization_create_dataset.py \
    --style_files=$STYLE_IMAGES_PATHS \
    --output_file=$RECORDIO_PATH \
    --compute_gram_matrices=False \
    --logtostderr
fi

if [ $a = train ] ; then
python arbitrary_image_stylization_train.py \
      --batch_size=8 \
      --imagenet_data_dir=./tmp/imagenet-2012-tfrecord \
      --vgg_checkpoint=./tmp/pretrain/vgg_16.ckpt \
      --inception_v3_checkpoint=./tmp/pretrain/inception_v3.ckpt\
      --style_dataset_file=$RECORDIO_PATH \
      --train_dir=./tmp/train_style_images_dir \
      --content_weights={\"vgg_16/conv3\":2.0} \
      --random_style_image_size=False \
      --augment_style_images=False \
      --center_crop=True \
      --logtostderr
fi