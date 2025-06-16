path = "/home/minhthan001/Projects/assignment_kaist_diffusion/image_diffusion_todo/results/diffusion-ddpm-06-15-211707"
import os
import re
import shutil

for file in os.listdir(path):
    # Define the pattern for files like {number}.png
    pattern = r'^\d+\.png$'

    # Create the samples directory if it doesn't exist
    samples_dir = os.path.join(path, 'samples')
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    # Check if the file matches the pattern and move it to the samples directory
    if re.match(pattern, file):
        src = os.path.join(path, file)
        dst = os.path.join(samples_dir, file)
        shutil.move(src, dst)