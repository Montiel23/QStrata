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


def draw_cv_ascii(ansatz):
    n_modes = ansatz.n_modes
    manifest = ansatz.get_circuit_manifest()

    #initialize the lines for each mode
    mode_lines = [f"m{i}: --[D]--" for i in range(n_modes)]

    for layer in manifest:
        for i in range(n_modes):
            # add single mode gates from manifest
            gates = "".join([f"[{g}]" for g in layer["single_mode"]])
            mode_lines[i] += f"--{gates}--"


        if layer["two_mode"] == "BS":
            for i in range(n_modes):
                if i < n_modes - 1:
                    #connect current mode and the one below it
                    #we use a placeholder logic for 2-mode interactions
                    mode_lines[i] += "\u256c"
                    mode_lines[i+1] += "\u256c"

                elif i == n_modes -1 and n_modes % 2 != 0:
                    mode_lines[i] += "--"

        # # logic for beam splitter connections
        # if layer["two_mode"] == "BS":
        #     if i < n_modes - 1:
        #         mode_lines[i] += "\u256c"
        #         mode_lines[i+1] += "\u256c"

        #     else:
        #         mode_lines[i] += "--"

    for line in mode_lines:
        print(line + "--[Readout]")
    print("-------------------------------------------------------------------\n")


# def draw_cv_ascii(n_modes, depth):
#     print("\n---Proposed CV Photonic Circuit---")

#     #define the label for each line
#     modes = [f"m{i}:" for i in range(n_modes)]

#     # initial data encoding (displacemen)
#     for i in range(n_modes):
#         modes[i] += f"-[D(x{i})]-"


#     # variational layers
#     for d in range(depth):
#         for i in range(n_modes):
#             #single-mode gates
#             modes[i] += f"-[S_{d}]-[R_{d}]-"

#             # beam splitter (interaction)
#             # we draw a vertical connection between mode i and i+1
#             if i < n_modes - 1:
#                 modes[i] += "--\u256c--"
#                 modes[i+1] += "--\u256c--"


#     #final readout
#     for i in range(n_modes):
#         modes[i] += "-[M]-"
#         print(modes[i])


#     print("---------------------------------------------------\n")