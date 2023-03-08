from pygame import mixer 
import os
import random
import time 

# loder le fichier 

files = [f for f in os.listdir('music/angry/') if f.endswith(".mp3")]        
random_file = random.choice(files)
file_path = os.path.join('music/angry/', random_file)
mixer.init()
mixer.music.load(file_path)
mixer.music.play()

#stoppper le fichier 
time.sleep(10)
mixer.music.stop()



