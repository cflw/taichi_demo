import taichi as ti
import time
from PIL import Image	#pillow
import numpy as np	#numpy
ti.init(arch = ti.gpu)
#画布
c宽度 = 800
c高度 = 800
c尺寸 = (c宽度, c高度)
v图像 = ti.Vector.field(3, dtype = float, shape = c尺寸)
#图片
def load1():	#逐像素转换
	v文件 = Image.open("太极/logo.png")
	v数据 = list(v文件.getdata())
	#对齐坐标填充数据
	for x in range(min(v文件.width, c宽度)):
		for y in range(min(v文件.height, c高度)):
			#y倒置
			y0 = c高度 - y - 1
			v图像[x, y0] = ti.Vector(v数据[y * v文件.width + x]) / 255
	v文件.close()
def load2():	#填入numpy数组再传给场
	v文件 = Image.open("太极/logo.png")
	v数据 = np.array(v文件.getdata(), dtype = float)	#二维数组,shape=(像素数,通道数)
	v数据 /= 255.0
	v填充 = np.zeros((c宽度, c高度, 3), float)
	#对齐坐标填充数据
	for x in range(min(v文件.width, c宽度)):
		for y in range(min(v文件.height, c高度)):
			#y倒置
			y0 = c高度 - y - 1
			v填充[x, y0] = v数据[y * v文件.width + x]
	v图像.from_numpy(v填充)
	v文件.close()
#窗口
v窗口 = ti.ui.Window("载入图片", res = c尺寸)
v画布 = v窗口.get_canvas()
v开始时间 = time.time()
# load1()
load2()
v结束时间 = time.time()
print("耗时:", v结束时间 - v开始时间)
while v窗口.running:
	v画布.set_image(v图像)
	v窗口.show()
