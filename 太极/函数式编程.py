import taichi as ti
ti.init(arch = ti.cpu)
#全局调用函数(可以运行)
@ti.kernel
def f():
	h(1)
@ti.func
def g(a):
	print(a)
h = g
f()

#在类中保存成员函数并调用(可以运行)
@ti.data_oriented
class C:
	def __init__(self, f):
		self.f = f
	@ti.kernel
	def g(self):
		self.f(self, 2)
	@ti.func
	def h(self, a):
		print(a)
c = C(C.h)
c.g()

#传入函数(报错)
# @ti.func
# def ff(g_):
# 	g_(3)
# @ti.kernel
# def gg():
# 	ff(g)
# gg()