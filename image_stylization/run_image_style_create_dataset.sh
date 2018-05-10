python image_stylization_create_dataset.py \
      --vgg_checkpoint=pretrained/vgg_16.ckpt \
      --style_files=style/*.jpg \
      --output_file=./tmp/image_stylization/style_images.tfrecord