# The pizzetti proofs, explained

**Supplement** ([supplement.pdf](supplement.pdf), source `supplement.tex`): material that is not part of
the paper itself: the error bounds for the truncation of the interior series by powers of p (mean-value
remainder and Abel bound), the construction, validation and benchmarks of the contact expansions about
first and second contact and of their certified pieces (summarised in Sect. 6.3 of the paper), and the
proof of the bound for the contact pieces. The paper keeps the interior series, its error bound
(Theorem 2) with its short proof, and the fixed-node limb rule.

Below, a slow, step-by-step companion to all proofs, for readers who want to
follow them in detail. The paper keeps the terse versions; these documents add explanations,
worked numerical examples and links to the validation notebooks (`validation/`), which check
every step against independent references.

| Part | Paper | Content | Background needed |
|---|---|---|---|
| [Part 1](part1_interior.pdf) | Sections 2–4.1, Appendix A | Interior series, Theorem 2 (error bound), the cut | First-year calculus |
| [Part 2](part2_meanvalue.pdf) | Section 4.2, supplement S1 (formerly Appendix B) | Mean-value expansion, Propositions 1–2 | Calculus, Gauss's divergence theorem |
| [Part 3](part3_contact.pdf) | Sections 5.1, 5.3, supplement S2–S4 | Contact expansions, Theorem 3 | Calculus, plus five tools from complex analysis explained in the text |

The LaTeX sources are in this folder (`pdflatex part1_interior.tex`, etc.).
