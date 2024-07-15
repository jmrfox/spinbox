from spinbox import *

isospin = False
ket = HilbertState(2, ketwise=True, isospin=isospin).randomize(100)
bra = HilbertState(2, ketwise=False, isospin=isospin).randomize(101)
print(bra * ket)

op_1 = HilbertOperator(2, isospin=isospin).apply_sigma(0, 0).apply_sigma(1, 1)  # x y
op_2 = HilbertOperator(2, isospin=isospin).apply_sigma(0, 1).apply_sigma(1, 1)  # y x
comm = op_1 * op_2 - op_2 * op_1
print(comm)