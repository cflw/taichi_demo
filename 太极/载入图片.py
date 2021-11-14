import taichi as ti
from PIL import Image	#pillow
ti.init(arch = ti.gpu)
#画布
c宽度 = 800
c高度 = 800
c尺寸 = (c宽度, c高度)
v图像 = ti.Vector.field(3, dtype = float, shape = c尺寸)
#图片
def init():
	v文件 = Image.open("太极/logo.png")
	v数据 = list(v文件.getdata())
	#对齐坐标填充数据
	for x in range(min(v文件.width, c宽度)):
		for y in range(min(v文件.height, c高度)):
			#y倒置
			y0 = c高度 - y - 1
			v像素 = v数据[y * v文件.width + x]
			for i in range(3):
				v图像[x, y0][i] = v像素[i]
	v文件.close()
#窗口
v窗口 = ti.ui.Window("载入图片", res = c尺寸)
v画布 = v窗口.get_canvas()
init()
while v窗口.running:
	v画布.set_image(v图像)
	v窗口.show()
