# Task 6: Analysis

Fault detection on a three-phase transmission line (AI4003 Applied Machine Learning, Assignment 02).

---

## 6.1 Logistic regression (Set R) vs linear SVM

**What the results show:** On the test rows, logistic regression on Set R scored 72.6% and the linear SVM scored 73.6%. Both gave 0 false alarms, and both missed many faults (657 for logistic regression, 633 for the linear SVM).

**Question:** Both models draw a straight-line boundary. Why do they score almost the same, and what does that tell us about the data?

**Answer:**

---

## 6.2 Signs of the squared weights (Set Q)

**What the results show:** The weights on the current-squared terms are all positive (Ia² = 18.89, Ib² = 4.75, Ic² = 13.96). The weights on the voltage-squared terms are all negative (Va² = -6.03, Vb² = -6.46, Vc² = -6.14).

**Question:** What does a fault do to the size of the current and to the size of the voltage? How does that explain why the I² weights are positive and the V² weights are negative?

**Answer:**

---

## 6.3 Support vectors: linear SVM vs RBF SVM

**What the results show:** The best linear SVM uses 5835 support vectors. The best RBF SVM uses only 266. Both were trained on 7200 rows.

**Question:** What does it mean when most of the training points end up as support vectors?

**Answer:**

---

## 6.4 Effect of C on accuracy and support vectors

**What the results show:**

| Model | C goes from 0.1 to 100 | Accuracy | Support vectors |
|---|---|---|---|
| Linear | 0.1 to 100 | stays flat at about 0.726 | 5849 to 5835 |
| RBF, gamma = 0.1 | 0.1 to 100 | 0.950 to 0.995 | 1347 to 174 |
| RBF, gamma = 1 | 0.1 to 100 | 0.980 to 0.997 | 848 to 266 |

**Question:** What does the parameter C penalise, and what does a larger C allow the model to do?

**Answer:**

---

## 6.5 Is the test accuracy optimistic or pessimistic?

**What the results show:** Validation and test accuracy are very close. Set Q scored 0.9783 on validation and 0.9788 on test. The RBF SVM scored 0.9967 on validation and 0.9958 on test.

**Question:** The rows are consecutive time samples, but the split was random, so almost identical neighbouring rows ended up in both the training set and the test set. Is a test row with a near-identical twin in training easier or harder than a genuinely new fault? Is our test accuracy therefore optimistic or pessimistic for a real relay?

**Answer:**

---

## 6.6 Which model would I deploy?

**What the results show:**

| Model | Missed faults | False alarms |
|---|---|---|
| Logistic regression, Set R | 657 | 0 |
| Logistic regression, Set Q | 45 | 6 |
| Linear SVM | 633 | 0 |
| RBF SVM | 6 | 4 |

**Question:** For a protection relay, which kind of error is dangerous, and which model performs best on it? Which model would you deploy, and why?

**Answer:**
