## Section 2: The Simple Recurrent Unit

- Also known as the Elman Unit
- Sequences
  - We are used to tabular data: $X \times D$ matrix (no sequence)
  - What if the data has sequence? What are the dimensions then?
    - $X(1) = (D,), X(2)=(D,), \cdots, X(T)=(D,) \rightarrow$ $T \times D$ matrix
    - Total: $N \times T \times D$
