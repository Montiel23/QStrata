from qcore.state.vacuum import vacuum_state
from qcore.measurement.probability import measure_probability

def forward(circuit, backend, measure_wire):

    U = backend.compile(circuit)

    state0 = vacuum_state(circuit.n_qubits)

    out = backend.run(U, state0)

    return measure_probability(out, measure_wire, circuit.n_qubits)