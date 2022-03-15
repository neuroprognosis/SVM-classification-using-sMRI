"Longitudinal MRI Image Processing and
Classification of Rate of Brain Atrophy"

Abstract:

Alzheimer’s Disease (AD), characterized as a progressive neurological disorder, is
often identified by the associated decline in cognitive functions such as forgetfulness
or problems with language. Unfortunately, there is no cure; however, early detection
can help deal with the associated effects of the disease. Detection of the disease is
difficult due to overlapping features with aging; hence, precise detection and monitoring demand more sophisticated resources. This thesis work presents an automated
method to detect patients with AD based on anatomical magnetic resonance imaging
(MRI). Our method relies on using the segmented brain tissue, namely grey matter
(GM), from the T1-weighted MRI scans from patients. A well-established support
vector machines (SVMs) was applied to the segmented tissue to identify patterns that
characterize clinical groups of AD. A real-world dataset from the DZNE DELCODE cohort with known class labels was used to build and evaluate the classification models.
In particular, first, we build a baseline model that uses the cross-sectional MR data.
Later, we model the structural changes in the brain over time using longitudinal MRI
data. The longitudinal MRI scans were taken annually at three time points and rate
of change images were obtained from these scans. We start by building a classifier to
characterize subjects as healthy or cognitive decliners (subjective cognitive decline
(SCD) or mild cognitive impairment (MCI)). Due to limited data over time, we have
built a classifier to distinguish healthy subjects from MCI. The baseline model to classify unseen data as healthy and MCI obtained an accuracy of 75%, an F1 score of 74%,
and an AUC score of 81%. While the longitudinal model attained an accuracy of 82%,
an F1 score of 81%, and an AUC score of 82%. Additionally, the baseline model for
classification of cognitive normal (CN), SCD, and MCI acquired an accuracy of 48%,
an F1 score of 47%, and an AUC score of 63%. On the other hand, the longitudinal
model outperformed the baseline model with an accuracy of 50%, an F1 score of
47%, and an AUC score of 65%. The overall performance of the SVM models based on
longitudinal MRI data outperformed those built on cross-sectional MRI data.
