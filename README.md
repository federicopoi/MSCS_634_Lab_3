# MSCS 634 – Lab 3

## Purpose
Two clustering algorithms, K-Means and K-Medoids, run on the Wine dataset from scikit-learn with `k = 3` and compared. K-Means builds each cluster around a computed average position. K-Medoids has to use a real sample as the center. I scored both with Silhouette Score and Adjusted Rand Index, plotted the clusters side by side, and worked out which one actually fits this data.

## Dataset
178 wine samples, 13 chemical measurements, three cultivars at 59, 71, and 48 samples. No missing values, no duplicate rows. The cultivar labels are held back from the clustering entirely and only come out at the end to compute ARI.

The raw feature scales are the first real problem. `proline` has a standard deviation around 315 and tops out near 1,680, while `nonflavanoid_phenols` varies by about 0.12. Clustering is all distance, so on raw numbers `proline` would decide everything and the other twelve features would barely register. Z-score normalization fixes that: every feature ends up at mean 0 and standard deviation 1, so all 13 pull their own weight.

## Key insights
K-Means got a Silhouette Score of 0.2849 and an ARI of 0.8975. K-Medoids got 0.2676 and 0.7411. K-Means wins on both, but the interesting part is how differently the two gaps behave.

The silhouette scores are almost the same. That metric only looks at the geometry, how tight each cluster is against its neighbors, and by that standard both algorithms found partitions of roughly equal quality. The ARI gap is the wide one, and ARI is the metric that checks the answer against the real cultivars. So both found reasonable-looking clusters, and only one found the clusters that match the actual chemistry.

Counting misplaced samples makes it concrete: K-Means puts 6 of 178 wines in the wrong cultivar group, K-Medoids puts 16 in the wrong group, and the two disagree with each other on 14.

The shapes differ in a way that traces straight back to the definitions. K-Means centroids land in the visual middle of each group because a centroid is an average and never has to be a real wine. The K-Medoids markers sit noticeably off center, since none of the 178 actual wines happens to occupy the exact middle of its cluster. That constraint also shifts the boundary. K-Medoids inflated one cluster to 74 samples, past any true class size, and the per-sample silhouette plot shows its negative-scoring points piling up in exactly that cluster.

Both algorithms fail in the same region, along the seam where `class_1` overlaps its two neighbors in the PCA projection. That is a property of the data, not of either method.

On when to use which: K-Means is right here and generally when features are standardized, clusters are roughly round and comparable in size, and outliers are not a worry. It is also cheaper, since it only needs distances from points to `k` centers while PAM builds the full 178 by 178 distance matrix and rescans every cluster member each iteration.

K-Medoids pays off when outliers really are a threat, because a medoid cannot get dragged toward an extreme value the way a mean can. It also handles the case where all you have is a distance or dissimilarity matrix and no sensible average exists, which covers categorical data, Gower distance, and edit distance on strings, none of which K-Means can touch. The other nice property is that every center is a real record you can go read. Here that gave three actual wines, one per cultivar, which is more useful to look at than three averaged coordinate vectors.

For this dataset the standardization had already removed the scale problem that would have hurt K-Means, and the outliers left in `magnesium` and `proanthocyanins` were too mild to cost it anything. K-Medoids traded away accuracy without getting much robustness in exchange, which is why its ARI came out lower.

## Challenges and decisions
`scikit-learn` has no K-Medoids. The usual add-on, `scikit-learn-extra`, has no prebuilt wheels for current Python versions, so pip falls back to compiling it and the build dies on a missing C++ header (`fatal error: 'ios' file not found`). Its last release was 2023. Rather than depend on something that unreliable, I implemented PAM directly in about 25 lines, so the notebook runs with nothing beyond numpy, pandas, scipy, matplotlib, and scikit-learn.

Writing the algorithm myself raised an obvious question: is it actually right? Section 3.2 answers it. The `kmedoids` package on PyPI implements FasterPAM in Rust and does ship working wheels, so I ran it on the same distance matrix and compared. Both pick medoids 35, 106, and 148, both score 0.2676 silhouette and 0.7411 ARI, and the ARI between the two labelings is exactly 1.0000, meaning an identical partition. The check is wrapped in a try/except so the notebook still runs end to end if that package is missing.

For the medoid initialization I used k-means++ style seeding rather than picking three random samples. A plain random draw can easily land two medoids inside the same natural cluster, which gives PAM a bad starting point it may not escape. With the spread-out seeding the three medoids came from three different cultivars on the first try.

PCA is in the notebook only to make the scatter plots possible, since 13 dimensions cannot be drawn. Both algorithms still cluster on all 13 standardized features. The two components shown carry about 55% of the variance, which is enough to see the structure but worth stating explicitly so the plots are not mistaken for the full picture.

I added the misplaced-sample plot and the per-cluster silhouette profile because the two headline numbers do not explain themselves. A single silhouette average hides which cluster is weak, and an ARI value does not say which wines went where. Both plots turned out to point at the same cluster, which is what tied the metric gap to the off-center medoid.

## Files
- `MSCS_634_Lab_3.ipynb`: completed notebook with all outputs
- `screenshots/`: exported plots (class distribution, raw and standardized feature ranges, true cultivars, side-by-side clusters, metric comparison, silhouette profiles, misplaced samples)
