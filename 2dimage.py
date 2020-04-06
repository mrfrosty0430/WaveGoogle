import os
import math
import csv
import json
from PIL import Image, ImageDraw

def distance(x1,y1,x2,y2):
	return math.sqrt((x2-x1)**2 + (y2-y1)**2)


def main():
	training = "wave"
	dir = "output/" + training
	origdir = "results/" + training
	filepath = "csv/" + training + ".csv"
	employee_file = open (filepath, mode='w',)
	csv_writer = csv.writer(employee_file)
	for filename in os.listdir(dir):
		try:
			orig = filename.split("_keypoints.json")[0]
			im = Image.open(origdir+"/"+orig+"_rendered.png")
			draw = ImageDraw.Draw(im)
			print(os.path.join(dir,filename))
			with open(os.path.join(dir,filename), 'r') as f:
				json_dict = json.load(f)

			data = json_dict["people"][0]
			hand_data = data["hand_right_keypoints_2d"]

		# print(len(hand_data))
			if (len(hand_data)//3 != 21):
				print("not enough features to classify")
				return

			baseX = hand_data[0]
			baseY = hand_data[1]
			maxX = -1
			maxY = -1
			maxDist = -1
			for i in range(1,21):
				currX = hand_data[i*3]
				currY = hand_data[i*3+1]
				dist = distance(baseX,baseY,currX,currY)
				if dist > maxDist:
					maxDist = dist
					maxX = currX
					maxY = currY

			print(maxDist)

			x1,y1 = baseX - 300, baseY - 300
			if (x1 < 0):
				 x1 = 0
			if (y1 < 0):
				 y1 = 0
			x4,y4 = baseX + 300, baseY + 300
			if (x4 < 0):
				 x4 = 0
			if (y4 < 0):
				 y4 = 0

			x2,y2 = baseX + 300, baseY - 300
			if (x2 < 0):
				 x2 = 0
			if (y2 < 0):
				 y2 = 0
			x3,y3 = baseX - 300, baseY + 300
			if (x3 < 0):
				 x3 = 0
			if (y3 < 0):
				 y3 = 0

			draw.polygon([(x1,y1),(x2,y2),(x4,y4),(x3,y3)])
			im.save(orig+"edited","png")


		except:
			print("exception found")

if __name__ == '__main__':
	main()
