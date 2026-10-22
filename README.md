# Squash Player and Ball Detection

## Player Detection and Tracking
Use the scripts/track_players to start tracking players using the yolo v26n fine tuned model, and then use the calibrate.py to calibrate the court using the 4 different points on the court converting u, v to x, y.
After calibration use compute_coords to convert the 2-d pixel coordinates to real world coordinates and thus, this would help in tracking the player distance and the average speed of the player throughout the game.

runs has the storage of all the times the model was run. runs/coordinates is basically the storage area where the data after tracking would be stored. 

## Ball Detection and Tracking
Use ball tracking script in ball detection but it still hasn't been patched as the model keeps locking on some other id in the frame. I rather suggest to retrain the model with a better and bigger dataset for ball detection and tracking. Moreover, the occlusion problem will be even more prominent in the case of ball detection, because at many times the ball will get occluded because of the players, so a better tracker is needed. A physics model running in real time (with better calculation in built to predict where the ball would move instead of tracking it directly) would help solving the problem of occlusion, so I guess one of the solution would be calibrating the camera beforehand to convert the u,v axes into x,y and z using holography.  

## Media & Weights
Use the following link to download specific model weights for the ball and the player tracking models: https://drive.google.com/drive/folders/1UL51cZjY7sn0XGnh5Rlu6NhJL1-vNSHC?usp=drive_link
