# Squash Player and Ball Detection

## Player Detection and Tracking
Use the scripts/track_players to start tracking players using the yolo v26n fine tuned model, and then use the calibrate.py to calibrate the court using the 4 different points on the court converting u, v to x, y.
After calibration use compute_coords to convert the 2-d pixel coordinates to real world coordinates and thus, this would help in tracking the player distance and the average speed of the player throughout the game. 

## Ball Detection and Tracking
Right now the ball tracking is a bit weird as it is tracking something else entierly but use the track_ball as it uses kalman filter and the bytetrack. use the script from Ball Detection file.

## Media & Weights
Use the following link to download specific model weights for the ball and the player tracking models: https://drive.google.com/drive/folders/1UL51cZjY7sn0XGnh5Rlu6NhJL1-vNSHC?usp=drive_link
