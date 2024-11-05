#python drive.py
ffmpeg -framerate 10 -pattern_type glob -i 'frames/frame*.png' -c:v libx264 -pix_fmt yuv420p output.mp4 -y
rm frames/frame_*.*
vlc output.mp4