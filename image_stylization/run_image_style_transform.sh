set e

#monet
python image_stylization_transform.py \
      --num_styles=10 \
      --checkpoint=checkpoints/multistyle-pastiche-generator-monet.ckpt \
      --input_image=evaluation_images/test1.jpg \
      --which_styles="[0,1,2]" \
      --output_dir=./tmp/image_stylization/output \
      --output_basename="monet_stylized"


#varied
 python image_stylization_transform.py \
      --num_styles=32 \
      --checkpoint=checkpoints/multistyle-pastiche-generator-varied.ckpt \
      --input_image=evaluation_images/test2.jpg \
      --which_styles="[0,1,2]" \
      --output_dir=./tmp/image_stylization/output \
      --output_basename="varied_stylized"
      
 
 #mine train
 python image_stylization_transform.py \
      --num_styles=7 \
      --checkpoint=tmp/image_stylization/run1/train/ \
      --input_image=evaluation_images/test2.jpg \
      --which_styles="[0,1,2,3,4,5,6]" \
      --output_dir=./tmp/image_stylization/output \
      --output_basename="mine_stylized"