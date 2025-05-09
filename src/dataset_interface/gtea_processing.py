import cv2
import json
import os
import shutil

if __name__ == '__main__':
	try:
		os.mkdir(f'/home/rsourave/data/datasets/GTEA/processed')
	except:
		print('`processed` directory already exists')

	try:
		os.mkdir(f'/home/rsourave/data/datasets/GTEA/processed/videos')
	except:
		print('`videos` directory already exists')
	
	try:
		os.mkdir(f'/home/rsourave/data/datasets/GTEA/processed/gaze')
	except:
		print('`gaze` directory already exists')

	allVideos = os.listdir(f'/home/rsourave/data/datasets/GTEA/raw/Videos')
	videoJson = {}

	for i, video in enumerate(allVideos):
		srcVideo = f'/home/rsourave/data/datasets/GTEA/raw/Videos/{video}'
		dstVideo = f'/home/rsourave/data/datasets/GTEA/processed/videos/{video}'

		shutil.copy(srcVideo, dstVideo)
		videoJson[i] = video
	
	with open(f'/home/rsourave/data/datasets/GTEA/processed/videos/video_order.json', 'w') as f:
		f.write(json.dumps(videoJson))

	gazeJson = {}
	for key, videoName in videoJson.items():
		viewerCount = 1
		cap = cv2.VideoCapture(f'/home/rsourave/data/datasets/GTEA/processed/videos/{videoName}')
		frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		gazeJson[key] = {}
		gazeJson[key][0] = {}
		videoFileName = videoName.split('.')[0]
		f = open(f'/home/rsourave/data/datasets/GTEA/raw/Gaze/{videoFileName}-Glasses-Data.tsv', 'r')
		contents = f.read()
		contents = contents.split('\n')
		contents = contents[22:]
		contents = [x.split('\t') for x in contents]
		f.close()
		
		videoDuration = frameCount / 30
		if contents[-1][0] == '':
			contents = contents[: -1]
		lastTimestamp = int(contents[-1][0])
		secondsPerTick = videoDuration / lastTimestamp

		j = 0

		for i in range(frameCount):
			gazeJson[key][0][i] = {}
			frameDuration = (i * 1 / 30, (i + 1) * 1 / 30)
			
			while int(contents[j][0]) * secondsPerTick < frameDuration[0]:
				j += 1
			
			sumX = 0
			sumY = 0
			cnt = 0

			while int(contents[j][0]) * secondsPerTick < frameDuration[1]:
				x = None
				y = None
				xFound = False
				yFound = False
				if(len(contents[j]) >= 2):
					x = contents[j][1]
					if x == '':
						x = 0
					else:
						x = int(x)
						xFound = True
				else:
					x = 0

				if len(contents[j]) >= 3:
					y = contents[j][2]
					if y == '':
						y = 0
					else:
						y = int(y)
						yFound = True
				else:
					y = 0
				
				if xFound == True and yFound == True:
					sumX += x
					sumY += y
					cnt += 1
					
				j += 1
			
			sumX /= cnt
			sumY /= cnt



			# print(f'videoName: {videoName}')
			# with 
				# print(contents
			# print(f'i: {i}')
			# print(f'contents[{2*i}]: {contents[2*i]}')
			# print(f'contents[{2*i + 1}]: {contents[2*i+1]}')	
			# x0 = int(contents[2*i].split('\t')[1])
			# y0 = int(contents[2*i].split('\t')[2])

			# x1 = int(contents[2*i + 1].split('\t')[1])
			# y1 = int(contents[2*i + 1].split('\t')[2])

			avgX = sumX
			avgY = sumY

			if avgX == 0 and avgY == 0:
				avgX = 360
				avgY = 240
				
			gazeJson[key][0][i]['x'] = avgX
			gazeJson[key][0][i]['y'] = avgY
	with open(f'/home/rsourave/data/datasets/GTEA/processed/gaze/gaze_order.json', 'w') as f:
		f.write(json.dumps(gazeJson))

