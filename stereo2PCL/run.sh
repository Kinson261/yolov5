

python stereo_depth.py \
    --calibration_file ./configs/stereo.yml \
    --left_source ./data/left.jpg \
    --right_source ./data/right.jpg \
    --pointcloud_dir ./output/


python stereo_depth_video.py \
    --calibration_file ./configs/stereo.yml \
    --left_source ../video/left_out.avi \
    --right_source ../data/right_out.avi \
    --pointcloud_dir ./output/
#   python stereo_depth_video.py --calibration_file ./configs/stereo.yml  --left_source ../video/left_out.avi  --right_source ../video/right_out.avi  --pointcloud_dir ./output/
