This dataset contains the optimized structures of the nitroxides, as XYZ files.
Each structure is the most stable among all tested possibilities.

They are organized as `{solvent}/{complexation_state}/mol_{n}/mol_{n}_{state}.xyz`, where:

- `{solvent}`: the implicit solvent used in optimization (`water`, `acetonitrile`),
- `{complexation_state}`: the complexation state of the nitroxide (`singles`, `pairs`, `ACs`),
- `{n}`: the index of the nitroxide, as defined in Fig. 7,
- `{state}`: redox state of the nitroxide (`ox`, `rad`, `red`).

The title of the XYZ file is the final SCF energy, as computed at the wB97X-D/6-311+G* level in Gaussian 16 (C01).

