import torch

class Operator:
    def __init__(self, name, wires, params=None):
        self.name = name
        self.wires = tuple(wires)
        self.params = params

    def matrix(self):
        raise NotImplementedError

    def embed(self, n_qubits):
        U = self.matrix()
        wires = self.wires

        dim = 2 ** n_qubits
        result = torch.eye(dim, dtype=U.dtype)

        if len(wires) == 1:
            q = wires[0]
            return self._embed_single(U, q, n_qubits)

        if len(wires) == 2:
            return self._embed_two(U, wires, n_qubits)

        raise ValueError("Unsupported operator arity")

    def _embed_single(self, U, wire, n_qubits):
        I = torch.eye(2, dtype=U.dtype)

        ops = []
        for q in range(n_qubits):
            ops.append(U if q == wire else I)

        M = ops[0]
        for op in ops[1:]:
            M = torch.kron(M, op)

        return M


    def _embed_two(self, U4, wires, n_qubits):
        i, j = wires
        if i == j:
            raise ValueError("Two-qubit gate needs two distinct wiers")

        dim = 2 ** n_qubits
        M = torch.zeros((dim, dim), dtype=U4.dtype)

        #bit helpers: qubit 0 is most-significant in this convention
        def get_bit(x, q):
            return (x >> (n_qubits - 1 - q)) & 1

        def set_bit(x, q, b):
            mask = 1 << (n_qubits - 1 -q)
            return (x | mask) if b else (x & ~mask)

        for col in range(dim):
            bi = get_bit(col, i)
            bj = get_bit(col, j)

            local_in = (bi << 1) | bj

        # for each possible local out, write amplitude to correct row
            for local_out in range(4):
                bo_i = (local_out >> 1) & 1
                bo_j = local_out & 1

                row = col
                row = set_bit(row, i, bo_i)
                row = set_bit(row, j, bo_j)

                M[row, col] = U4[local_out, local_in]


        return M
            

    # def _embed_two(self, U, wires, n_qubits):
    #     #assume 2-qubit contiguous ordering

    #     q0, q1 = wires

    #     if q1 != q0 + 1:
    #         raise NotImplementedError("non-adjacent wires not implemented")

    #     I = torch.eye(2, dtype=U.dtype)

    #     ops = []
    #     skip = False
    #     for q in range(n_qubits):
    #         if skip:
    #             skip = False
    #             continue

    #         if q == q0:
    #             ops.append(U)
    #             skip = True
    #         else:
    #             ops.append(I)

    #     M = ops[0]
    #     for op in ops[1:]:
    #         M = torch.kron(M, op)

    #     return M