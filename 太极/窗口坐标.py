import taichi as ti
ti.init(arch = ti.gpu)
n = 800
pixels = ti.Vector.field(3, dtype = float, shape = (n, n))
@ti.kernel
def paint():
	for i, j in pixels:
		pixels[i, j] = ti.Vector([i / n, j / n, 0])
v窗口 = ti.GUI("color", res = (n, n))
paint()
while v窗口.running:
	v窗口.set_image(pixels)
	v窗口.show()
