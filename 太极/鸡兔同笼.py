import taichi as ti
ti.init(arch = ti.cpu)
# v头 = 35
# v腿 = 94
v头 = int(input("头数："))
v腿 = int(input("腿数："))
v建造 = ti.SparseMatrixBuilder(2, 2, dtype = int, max_num_triplets = 100)
v向量 = ti.field(dtype = int, shape = 2)
@ti.kernel
def init(a建造: ti.SparseMatrixBuilder()):
	a建造[0, 0] += 1
	a建造[0, 1] += 1
	a建造[1, 0] += 2
	a建造[1, 1] += 4
	v向量[0] = v头
	v向量[1] = v腿
init(v建造)
# v建造.print_triplets()
v矩阵 = v建造.build()
# print(v矩阵)
# print(v向量)

v解 = ti.SparseSolver(dtype = int, solver_type = "LU")
v解.analyze_pattern(v矩阵)
v解.factorize(v矩阵)
v答 = v解.solve(v向量)
# v成功 = v解.info()
print("鸡数：", int(v答[0]))
print("兔数：", int(v答[1]))
# print(v成功)