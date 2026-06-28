# Perhitungan Manual Classification Report

## Confusion Matrix

| Actual \ Predicted | Anxiety Rendah | Anxiety Tinggi |
| ------------------ | -------------: | -------------: |
| Anxiety Rendah     |              9 |              6 |
| Anxiety Tinggi     |              6 |             12 |

Total data:

[9 + 6 + 6 + 12 = 33]

---

# 1. Accuracy

Rumus accuracy:

[text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}]

Dengan kelas **Anxiety Tinggi** sebagai positive class:

- TP = 12
- TN = 9
- FP = 6
- FN = 6

Substitusi:

[text{Accuracy} = \frac{12 + 9}{12 + 9 + 6 + 6}]

[= \frac{21}{33}]

[= 0.6364]

\[
= 63.64\%
\]

---

# 2. Precision

Rumus precision:

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

Substitusi:

\[
\text{Precision} = \frac{12}{12 + 6}
\]

\[
= \frac{12}{18}
\]

\[
= 0.6667
\]

\[
= 66.67\%
\]

---

# 3. Recall

Rumus recall:

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

Substitusi:

\[
\text{Recall} = \frac{12}{12 + 6}
\]

\[
= \frac{12}{18}
\]

\[
= 0.6667
\]

\[
= 66.67\%
\]

---

# 4. F1-Score

Rumus F1-score:

\[
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}
{\text{Precision} + \text{Recall}}
\]

Substitusi:

\[
F1 = 2 \times \frac{0.6667 \times 0.6667}
{0.6667 + 0.6667}
\]

\[
= 2 \times \frac{0.4444}{1.3334}
\]

\[
= 0.6667
\]

\[
= 66.67\%
\]

---

# Classification Report Manual

## Kelas: Anxiety Rendah

Untuk kelas ini:

- TP = 9
- FP = 6
- FN = 6

### Precision

\[
\frac{9}{9+6}
=
\frac{9}{15}
=
0.60
\]

### Recall

\[
\frac{9}{9+6}
=
\frac{9}{15}
=
0.60
\]

### F1-score

\[
2 \times \frac{0.60 \times 0.60}{0.60+0.60}
\]

# \[

0.60
\]

---

## Kelas: Anxiety Tinggi

Untuk kelas ini:

- TP = 12
- FP = 6
- FN = 6

### Precision

\[
\frac{12}{12+6}
=
\frac{12}{18}
=
0.6667
\]

### Recall

\[
\frac{12}{12+6}
=
\frac{12}{18}
=
0.6667
\]

### F1-score

\[
2 \times \frac{0.6667 \times 0.6667}{0.6667+0.6667}
\]

# \[

0.6667
\]

---

# Ringkasan Classification Report

| Class          | Precision | Recall | F1-score | Support |
| -------------- | --------: | -----: | -------: | ------: |
| Anxiety Rendah |      0.60 |   0.60 |     0.60 |      15 |
| Anxiety Tinggi |      0.67 |   0.67 |     0.67 |      18 |

---

# Accuracy

\[
\text{Accuracy} =
\frac{21}{33}
=
0.6364
=
63.64\%
\]

---

# Macro Average

## Precision

\[
\frac{0.60 + 0.6667}{2}
=
0.6333
\]

## Recall

\[
\frac{0.60 + 0.6667}{2}
=
0.6333
\]

## F1-score

\[
\frac{0.60 + 0.6667}{2}
=
0.6333
\]

---

# Weighted Average

Support:

- Anxiety Rendah = 15
- Anxiety Tinggi = 18

## Weighted Precision

\[
\frac{(0.60 \times 15) + (0.6667 \times 18)}{33}
\]

# \[

# \frac{9 + 12}{33}

0.6364
\]

## Weighted Recall

# \[

0.6364
\]

## Weighted F1-score

# \[

0.6364
\]

---
