import numpy as np
from PIL import Image
from glob import glob
import os

def resizeImg(img):
	vsize = int(img.size[0]/4)
	hsize = int(img.size[1]/4)
	img2 = img.resize((vsize, hsize), Image.ANTIALIAS)
	return img2

if __name__ == "__main__":
	'''
	dire = "./train/004.Akita/Akita_00220.jpg"
	img = Image.open(dire)
	vsize = int(img.size[0]/4)
	hsize = int(img.size[1]/4)
	img = img.resize((vsize, hsize), Image.ANTIALIAS)
	img.save('resize.png')
	'''
	
	diref = "./train/"
	folder = sorted(glob(diref + "*/images/"))
	folders = folder[:]
	print(folders)
	#####
	num = 7
	#####

	for i in folders:
		print(i)
		newfolder = i[0:num] + '2' + i[num:-7]
		#print(newfolder)
		#input()
		if not os.path.exists(newfolder):
			os.makedirs(newfolder)
		files = sorted(glob(i + "*.png"))
		
		for file in files:
			print ("original=",file)
			#input()
			result=file.find('image')
			#print("index=",result)
			#input()
			try:
				img = Image.open(file)
				#img2 = resizeImg(img)
				#####
				newdir = file[0:num] + '2' + file[num:result-1] + file[result+6:-4] + '.png'
				#####
				print("new directory=",newdir)
				#input()
				img.save(newdir)
				print("done!!")
			except:
				continue


