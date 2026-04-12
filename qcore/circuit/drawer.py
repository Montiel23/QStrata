def draw_ascii(circuit):

    n = circuit.n_qubits
    wires = [f"q{i}: " for i in range(n)]

    for op in circuit.ops:

        name = op.name

        if len(op.wires) == 1:

            w = op.wires[0]

            #standardize name to be centered in a 4-character block
            display_name = name.center(2)

            for i in range(n):
                if i == w:
                    wires[i] += f"─{display_name}─"
                else:
                    wires[i] += "────"

        elif len(op.wires) == 2:

            c, t = op.wires

            for i in range(n):
                if i == c:
                    wires[i] += "─●─"
                elif i == t:
                    wires[i] += "─X─"
                else:
                    wires[i] += "───"

    for w in wires:
        print(w)