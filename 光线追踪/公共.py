import taichi as ti
import enum
#常量
PI = 3.14159265
半PI = 1.5707963
二PI = 6.2831853
c最小值 = 0.0001
c最大值 = 10e5
#类型
t向量3 = ti.types.vector(3, float)
#===============================================================================
# 函数
#===============================================================================
@ti.func
def random_vector3():	#随机向量3
	return t向量3(ti.random(), ti.random(), ti.random())
@ti.func
def random_in_unit_sphere():
	θ = 2.0 * PI * ti.random()
	φ = ti.acos((2.0 * ti.random()) - 1.0)
	r = ti.pow(ti.random(), 1.0/3.0)
	return t向量3(r * ti.sin(φ) * ti.cos(θ), r * ti.sin(φ) * ti.sin(θ), r * ti.cos(φ))
	# p = 2.0 * random_vector3() - ti.Vector([1, 1, 1])
	# while p.norm() >= 1.0:
	# 	p = 2.0 * random_vector3() - ti.Vector([1, 1, 1])
	# return p
@ti.func
def random_unit_vector():
	return random_in_unit_sphere().normalized()
@ti.func
def reflect(v, n):	#反射
	return v - 2 * v.dot(n) * n
@ti.func
def refract(uv, n, etai_over_etat):	#折射
	cosθ = min(n.dot(-uv), 1.0)
	r_out_perp = etai_over_etat * (uv + cosθ * n)
	r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp))) * n
	return r_out_perp + r_out_parallel
@ti.func
def reflectance(cosine, ref_idx):	#反射比
	# 施力克近似（Schlick's approximation）求反射透明度
	r0 = (1 - ref_idx) / (1 + ref_idx)
	r0 = r0 * r0
	return r0 + (1 - r0) * pow((1 - cosine), 5)
#===============================================================================
# 几何体
#===============================================================================
@ti.data_oriented
class C球体:
	def __init__(self, a位置: t向量3, a半径: float, a颜色: t向量3, a材质):
		self.m位置 = a位置
		self.m半径 = a半径
		self.m颜色 = a颜色	#物体反射光
		self.m材质 = a材质
	@ti.func
	def intersect(self, a光线, a最小值, a最大值):	#求光线交于圆表面的距离,不相交则返回0
		v相对位置 = a光线.m位置 - self.m位置
		a = a光线.m方向.dot(a光线.m方向)
		b = 2 * v相对位置.dot(a光线.m方向)
		c = v相对位置.dot(v相对位置) - self.m半径 * self.m半径
		d = b * b - 4 * a * c
		x = 0.0
		if d > 0:
			sqrt_d = ti.sqrt(d)
			x = (-b - sqrt_d) / (2 * a)
			if x < a最小值 or x > a最大值:
				x = (-b + sqrt_d) / (2 * a)
				if x < a最小值 or x > a最大值:
					x = 0
		else:
			x = 0
		return x
#===============================================================================
# 渲染
#===============================================================================
class E材质(enum.IntEnum):
	e光源 = enum.auto()
	e漫反射 = enum.auto()
	e镜面反射 = enum.auto()
	e折射 = enum.auto()
	e模糊反射 = enum.auto()
@ti.data_oriented
class C光线:
	def __init__(self, a位置: t向量3, a方向: t向量3):
		self.m位置 = a位置
		self.m方向 = a方向
	@ti.func
	def at(self, t: float):
		return self.m位置 + self.m方向 * t
@ti.data_oriented
class C场景:	#存放物体
	def __init__(self):
		self.m物体 = []
	def add(self, obj):
		self.m物体.append(obj)
	@ti.func
	def hit(self, a光线):	#计算光与哪个物体相交,返回物体属性
		v距离 = 10e8
		v选择 = -1
		v碰撞 = False
		v交点 = t向量3(0, 0, 0)
		v交点法线 = t向量3(0, 0, 0)
		v前面 = False
		v颜色 = t向量3(0, 0, 0)
		v材质 = E材质.e漫反射
		for i in ti.static(range(len(self.m物体))):
			x = self.m物体[i].intersect(a光线, c最小值, v距离)
			if x > 0:
				#python似乎只把i当做整数,其它太极作用域内的变量都不是整数.直接写 self.m物体[v选择] 会报错
				v距离 = x
				v选择 = i
				#计算相关变量
				v碰撞 = True
				v交点 = a光线.at(v距离)
				v交点法线 = (v交点 - self.m物体[i].m位置).normalized()
				if a光线.m方向.dot(v交点法线) < 0:
					v前面 = True
				else:
					v交点法线 = -v交点法线
				v颜色 = self.m物体[i].m颜色
				v材质 = self.m物体[i].m材质
		#返回
		return v碰撞, v交点, v交点法线, v前面, v颜色, v材质
@ti.data_oriented
class C相机:	#生成光线
	def __init__(self, a位置: t向量3, a目标: t向量3, a上方: t向量3):
		self.m位置 = a位置
		self.m目标 = a目标
		self.m上方 = a上方
@ti.data_oriented
class C透视投影:
	def __init__(self, a宽高比: float = 1, a视角: float = PI / 3):
		self.m宽高比 = a宽高比
		self.m视角 = a视角
@ti.data_oriented
class C取景框:	#由相机和透视投影计算得到
	def __init__(self, a相机: C相机, a透视: C透视投影):
		self.m位置 = a相机.m位置
		v半高 = ti.tan(a透视.m视角 / 2)
		v半宽 = v半高 * a透视.m宽高比
		w = (a相机.m位置 - a相机.m目标).normalized()
		u = (a相机.m上方.cross(w)).normalized()
		v = w.cross(u)
		self.m左下角 = a相机.m位置 - v半宽 * u - v半高 * v - w
		self.m水平 = 2 * v半宽 * u
		self.m垂直 = 2 * v半高 * v
	@ti.func
	def get_ray(self, u: float, v: float):
		return C光线(self.m位置, self.m左下角 + u * self.m水平 + v * self.m垂直 - self.m位置)