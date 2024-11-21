


An atomic descriptor primarily consists of two parts. The first part comprises computational chemistry descriptors derived from **Conceptual Density Functional Theory (CDFT)** analysis. With the help of **Multiwfn**, these descriptors can be easily extracted from wavefunctions.

| **Chemical Significance**           | **Parameter**       | **Definition**                                                  |
| ----------------------------------- | ------------------- | --------------------------------------------------------------- |
| **Hirshfeld charges**               | $q_N^A$             | Atom charge in the neutral system (\(N\)) with \(N\) electrons. |
|                                     | $q_{N+1}^A$         | Atom charge in the anionic state with \(N+1\) electrons.        |
|                                     | $q_{N-1}^A$         | Atom charge in the cationic state with \(N-1\) electrons.       |
| **Electrophilic attack**            | $f_{A}^{-}$         | \(f_{A}^{-} = q_N^A - q_{N-1}^A\)                               |
| **Nucleophilic attack**             | $f_{A}^{+}$         | \(f_{A}^{+} = q_{N+1}^A - q_N^A\)                               |
| **Radical attack**                  | $f_{A}^{0}$         | \(f_{A}^{0} = \frac{q_{N-1}^A + q_{N+1}^A}{2}\)                 |
| **Reactivity descriptor**           | $\Delta f_A$        | \(\Delta f_A = f_A^{+} - f_A^{-}\)                              |
| **Electrophilicity**                | $\omega_{A}^{loc}$  | \(\omega_{A}^{loc} = \omega \times f_{A}^{+}\)                  |
| **Nucleophilicity**                 | $\N_{A}^{loc}$      | \(\N_{A}^{loc} = N_{\mathrm{Nu}} \times f_{A}^{-}\)             |
| **Local electrophilic softness**    | $s_A^{-}$           | \(s_A^{-} = S \times f_A^{-}\)                                  |
| **Local nucleophilic softness**     | $s_A^{+}$           | \(s_A^{+} = S \times f_A^{+}\)                                  |
| **Local radical softness**          | $s_A^0$             | \(s_A^0 = S \times f_A^0\)                                      |
| **Relative electrophilicity index** | $s_A^{+} / s_A^{-}$ |                                                                 |
| **Relative nucleophilicity index**  | $s_A^{-} / s_A^{+}$ |                                                                 |
| **Local hyper-softness**            | $s_A^{(2)}$         | \(s_A^{(2)} = S^2 \times \Delta f_A\)                           |

The **softness (\(S\))** of a molecule describes its ability to adjust or redistribute its electron density in response to external perturbations, such as during chemical reactions. It is the inverse of hardness (\(\eta\)) and is defined as \(S = \frac{1}{\eta}\). A higher softness value indicates that a molecule can more easily adapt its electron density, making it more chemically reactive. In contrast, harder molecules are less reactive.

The **electrophilicity index (\(\omega\))** quantifies a molecule's ability to accept electrons, indicating its tendency to act as an electrophile. A higher \(\omega\) value suggests a stronger electrophile. It is calculated using the chemical potential \(\mu\) and hardness \(\eta\), as \(\omega = \frac{\mu^2}{2\eta}\).

The **nucleophilicity index (\(N_{\mathrm{Nu}}\))** measures a molecule's electron-donating ability, representing its nucleophilic strength. It is calculated relative to a reference nucleophile, often **tetracyanoethylene (TCE)**, using the difference in HOMO energies: \(N_{\mathrm{Nu}} = E_{\text{HOMO}}(\text{Nu}) - E_{\text{HOMO}}(\text{TCE})\). Higher \(N_{\mathrm{Nu}}\) values indicate stronger nucleophiles. Together, these indices provide insight into the reactivity of molecules as electron acceptors or donors in chemical reactions.

The second part is the distance matrix, whose length varies depending on the type of reaction.

For the **Bergman cyclization**, the matrix length is 7, consisting of:
- the distance to the origin, and
- the distances to atoms indexed from 0 to 5.

Similarly, for the **Myers-Saito cyclization**, the matrix length is 8, consisting of:
- the distance to the origin, and
- the distances to atoms indexed from 0 to 6.

