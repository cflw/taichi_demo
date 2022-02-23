import taichi as ti
ti.init(arch=ti.cpu)
@ti.data_oriented
class A:
	def __init__(self, a):
		self.m = a
	@ti.func
	def f(self):
		print(self.m)
@ti.kernel
def run(v: int):
	a = A(v)
	a.f()
run(123)
run(321)