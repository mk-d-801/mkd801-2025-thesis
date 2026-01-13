import os
import subprocess

parent_input_dir = './dataset/MELD.Raw'
parent_output_dir = './dataset/MELD.Wav'

for subdir in os.listdir(parent_input_dir):
    input_dir = os.path.join(parent_input_dir, subdir)
    output_dir = os.path.join(parent_output_dir, subdir)
    
    if not os.path.isdir(input_dir):
        continue

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.mp4'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.wav')

            cmd = [
                'ffmpeg',
                '-y',  
                '-i', input_path,
                '-vn', 
                '-acodec', 'pcm_s16le',  
                '-ar', '16000',          
                '-ac', '1',              
                output_path
            ]

            try:
                subprocess.run(cmd, check=True)
                print(f"Converted {input_path} -> {output_path}")
            except subprocess.CalledProcessError:
                print(f"Error converting {input_path}")
