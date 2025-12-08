# Metrics & Confusion Matrices (one memory)

## AURC

| Model | RaVAEn-Landslides | RaVAEn-Wildfires | RaVAEn-Hurricanes | RaVAEn-Floods | STTORM-Floods |
|---|---|---|---|---|---|
| RaVAEn (small) | 84.34 | 89.10 | 77.55 | 77.09 | 47.00 |
| RaVAEn (medium) | 85.56 | 90.12 | 76.45 | 76.09 | 50.19 |
| RaVAEn (large) | 83.57 | 89.95 | 76.94 | 74.54 | 47.81 |
| STTORM-CD (small, variable) | 74.97 | 85.73 | 65.23 | 86.77 | 81.71 |
| STTORM-CD (medium, variable) | 79.90 | 78.82 | 69.41 | 85.85 | 82.72 |
| STTORM-CD (large, variable) | 79.87 | 85.31 | 71.05 | 85.17 | 79.87 |
| STTORM-CD (small, fixed) | 62.94 | 80.74 | 60.69 | 88.09 | 79.39 |
| STTORM-CD (medium, fixed) | 74.33 | 76.75 | 67.13 | 86.48 | 80.42 |
| STTORM-CD (large, fixed) | 60.83 | 70.53 | 62.99 | 85.08 | 79.40 |
| Index | 89.03 | 95.75 | 88.00 | 70.32 | 77.67 |
| Cosine baseline | 77.44 | 90.40 | 65.39 | 72.32 | 62.96 |


## RDP

| Model | RaVAEn-Landslides | RaVAEn-Wildfires | RaVAEn-Hurricanes | RaVAEn-Floods | STTORM-Floods |
|---|---|---|---|---|---|
| RaVAEn (small) | 50.32 | 99.82 | 99.29 | 96.33 | 80.70 |
| RaVAEn (medium) | 51.28 | 99.80 | 97.43 | 95.17 | 83.36 |
| RaVAEn (large) | 52.88 | 99.74 | 98.90 | 94.63 | 80.59 |
| STTORM-CD (small, variable) | 56.55 | 99.98 | 96.77 | 89.02 | 72.58 |
| STTORM-CD (medium, variable) | 49.36 | 99.99 | 97.54 | 90.99 | 85.91 |
| STTORM-CD (large, variable) | 54.95 | 99.99 | 95.93 | 95.50 | 75.89 |
| STTORM-CD (small, fixed) | 61.02 | 99.95 | 99.82 | 94.07 | 75.71 |
| STTORM-CD (medium, fixed) | 60.22 | 99.99 | 94.61 | 91.22 | 76.98 |
| STTORM-CD (large, fixed) | 53.19 | 99.99 | 99.13 | 91.75 | 79.99 |
| Index | 96.33 | 100.00 | 99.99 | 99.99 | 99.73 |
| Cosine baseline | 64.70 | 99.99 | 98.40 | 98.28 | 81.70 |


## AUPRC

| Model | RaVAEn-Landslides | RaVAEn-Wildfires | RaVAEn-Hurricanes | RaVAEn-Floods | STTORM-Floods |
|---|---|---|---|---|---|
| RaVAEn (small) | 77.40 | 89.27 | 72.16 | 70.32 | 39.21 |
| RaVAEn (medium) | 79.10 | 90.36 | 70.24 | 68.70 | 40.74 |
| RaVAEn (large) | 76.67 | 90.23 | 69.03 | 67.15 | 39.10 |
| STTORM-CD (small, variable) | 70.67 | 86.37 | 57.70 | 81.85 | 72.45 |
| STTORM-CD (medium, variable) | 73.29 | 79.66 | 64.35 | 78.79 | 73.45 |
| STTORM-CD (large, variable) | 73.71 | 85.48 | 66.10 | 80.22 | 69.05 |
| STTORM-CD (small, fixed) | 59.44 | 81.33 | 50.93 | 81.55 | 66.51 |
| STTORM-CD (medium, fixed) | 67.92 | 77.59 | 61.58 | 81.35 | 69.03 |
| STTORM-CD (large, fixed) | 61.48 | 72.76 | 54.67 | 81.26 | 69.08 |
| Index | 81.86 | 95.82 | 84.45 | 56.18 | 60.28 |
| Cosine baseline | 68.40 | 89.93 | 64.63 | 62.65 | 48.04 |


## F1

| Model | RaVAEn-Landslides | RaVAEn-Wildfires | RaVAEn-Hurricanes | RaVAEn-Floods | STTORM-Floods |
|---|---|---|---|---|---|
| RaVAEn (small) | 69.60 | 84.23 | 63.07 | 66.48 | 46.90 |
| RaVAEn (medium) | 70.94 | 85.53 | 61.66 | 64.77 | 47.77 |
| RaVAEn (large) | 68.63 | 85.48 | 60.22 | 64.22 | 47.52 |
| STTORM-CD (small, variable) | 68.73 | 84.33 | 52.36 | 72.15 | 66.42 |
| STTORM-CD (medium, variable) | 69.46 | 81.80 | 58.52 | 72.17 | 69.00 |
| STTORM-CD (large, variable) | 65.56 | 82.38 | 60.26 | 72.54 | 63.18 |
| STTORM-CD (small, fixed) | 63.37 | 82.38 | 44.88 | 72.19 | 60.91 |
| STTORM-CD (medium, fixed) | 65.01 | 80.63 | 56.49 | 72.87 | 62.00 |
| STTORM-CD (large, fixed) | 66.25 | 79.48 | 48.12 | 73.86 | 61.56 |
| Index | 77.61 | 89.21 | 77.79 | 54.53 | 59.24 |
| Cosine baseline | 65.51 | 83.67 | 66.96 | 69.56 | 52.97 |


## Precision

| Model | RaVAEn-Landslides | RaVAEn-Wildfires | RaVAEn-Hurricanes | RaVAEn-Floods | STTORM-Floods |
|---|---|---|---|---|---|
| RaVAEn (small) | 67.38 | 82.38 | 59.30 | 65.01 | 34.66 |
| RaVAEn (medium) | 70.68 | 85.92 | 61.99 | 64.82 | 36.56 |
| RaVAEn (large) | 66.91 | 84.54 | 66.04 | 59.67 | 36.01 |
| STTORM-CD (small, variable) | 62.89 | 81.35 | 37.91 | 70.62 | 63.61 |
| STTORM-CD (medium, variable) | 77.57 | 74.45 | 45.53 | 71.53 | 63.94 |
| STTORM-CD (large, variable) | 51.52 | 75.77 | 47.49 | 76.79 | 61.70 |
| STTORM-CD (small, fixed) | 56.14 | 75.13 | 36.72 | 73.91 | 53.50 |
| STTORM-CD (medium, fixed) | 51.08 | 75.35 | 42.84 | 75.88 | 62.02 |
| STTORM-CD (large, fixed) | 56.76 | 70.98 | 38.15 | 80.14 | 59.11 |
| Index | 76.47 | 88.04 | 82.89 | 55.50 | 55.56 |
| Cosine baseline | 53.05 | 84.64 | 54.49 | 66.92 | 44.78 |


## Recall

| Model | RaVAEn-Landslides | RaVAEn-Wildfires | RaVAEn-Hurricanes | RaVAEn-Floods | STTORM-Floods |
|---|---|---|---|---|---|
| RaVAEn (small) | 71.97 | 86.17 | 67.35 | 68.02 | 72.52 |
| RaVAEn (medium) | 71.21 | 85.13 | 61.33 | 64.72 | 68.89 |
| RaVAEn (large) | 70.45 | 86.43 | 55.34 | 69.52 | 69.84 |
| STTORM-CD (small, variable) | 75.76 | 87.53 | 84.62 | 73.75 | 69.49 |
| STTORM-CD (medium, variable) | 62.88 | 90.75 | 81.88 | 72.82 | 74.94 |
| STTORM-CD (large, variable) | 90.15 | 90.26 | 82.45 | 68.75 | 64.74 |
| STTORM-CD (small, fixed) | 72.73 | 91.17 | 57.71 | 70.55 | 70.70 |
| STTORM-CD (medium, fixed) | 89.39 | 86.72 | 82.92 | 70.09 | 61.97 |
| STTORM-CD (large, fixed) | 79.55 | 90.28 | 65.14 | 68.49 | 64.22 |
| Index | 78.79 | 90.40 | 73.27 | 53.58 | 63.44 |
| Cosine baseline | 85.61 | 82.72 | 86.83 | 72.41 | 64.82 |



---

## Confusion Matrices

### RaVAEn-Landslides

| Model | True Positive | False Positive | False Negative | True Negative |
|---|---|---|---|---|
| RaVAEn (small) | 95 | 46 | 37 | 448 |
| RaVAEn (medium) | 94 | 39 | 38 | 455 |
| RaVAEn (large) | 93 | 46 | 39 | 448 |
| STTORM-CD (small, variable) | 100 | 59 | 32 | 435 |
| STTORM-CD (medium, variable) | 83 | 24 | 49 | 470 |
| STTORM-CD (large, variable) | 119 | 112 | 13 | 382 |
| STTORM-CD (small, fixed) | 96 | 75 | 36 | 419 |
| STTORM-CD (medium, fixed) | 118 | 113 | 14 | 381 |
| STTORM-CD (large, fixed) | 105 | 80 | 27 | 414 |
| Index | 104 | 32 | 28 | 462 |
| Cosine baseline | 113 | 100 | 19 | 394 |


### RaVAEn-Wildfires

| Model | True Positive | False Positive | False Negative | True Negative |
|---|---|---|---|---|
| RaVAEn (small) | 13791 | 2950 | 2213 | 8911 |
| RaVAEn (medium) | 13625 | 2232 | 2379 | 9629 |
| RaVAEn (large) | 13832 | 2529 | 2172 | 9332 |
| STTORM-CD (small, variable) | 14009 | 3211 | 1995 | 8650 |
| STTORM-CD (medium, variable) | 14524 | 4984 | 1480 | 6877 |
| STTORM-CD (large, variable) | 14445 | 4619 | 1559 | 7242 |
| STTORM-CD (small, fixed) | 14591 | 4830 | 1413 | 7031 |
| STTORM-CD (medium, fixed) | 13878 | 4540 | 2126 | 7321 |
| STTORM-CD (large, fixed) | 14449 | 5908 | 1555 | 5953 |
| Index | 14468 | 1965 | 1536 | 9896 |
| Cosine baseline | 13239 | 2402 | 2765 | 9459 |


### RaVAEn-Hurricanes

| Model | True Positive | False Positive | False Negative | True Negative |
|---|---|---|---|---|
| RaVAEn (small) | 2137 | 1467 | 1036 | 7133 |
| RaVAEn (medium) | 1946 | 1193 | 1227 | 7407 |
| RaVAEn (large) | 1756 | 903 | 1417 | 7697 |
| STTORM-CD (small, variable) | 2685 | 4397 | 488 | 4203 |
| STTORM-CD (medium, variable) | 2598 | 3108 | 575 | 5492 |
| STTORM-CD (large, variable) | 2616 | 2893 | 557 | 5707 |
| STTORM-CD (small, fixed) | 1831 | 3155 | 1342 | 5445 |
| STTORM-CD (medium, fixed) | 2631 | 3511 | 542 | 5089 |
| STTORM-CD (large, fixed) | 2067 | 3351 | 1106 | 5249 |
| Index | 2325 | 480 | 848 | 8120 |
| Cosine baseline | 2755 | 2301 | 418 | 6299 |


### RaVAEn-Floods

| Model | True Positive | False Positive | False Negative | True Negative |
|---|---|---|---|---|
| RaVAEn (small) | 1319 | 710 | 620 | 8604 |
| RaVAEn (medium) | 1255 | 681 | 684 | 8633 |
| RaVAEn (large) | 1348 | 911 | 591 | 8403 |
| STTORM-CD (small, variable) | 1430 | 595 | 509 | 8719 |
| STTORM-CD (medium, variable) | 1412 | 562 | 527 | 8752 |
| STTORM-CD (large, variable) | 1333 | 403 | 606 | 8911 |
| STTORM-CD (small, fixed) | 1368 | 483 | 571 | 8831 |
| STTORM-CD (medium, fixed) | 1359 | 432 | 580 | 8882 |
| STTORM-CD (large, fixed) | 1328 | 329 | 611 | 8985 |
| Index | 1039 | 833 | 900 | 8481 |
| Cosine baseline | 1404 | 694 | 535 | 8620 |


### STTORM-Floods

| Model | True Positive | False Positive | False Negative | True Negative |
|---|---|---|---|---|
| RaVAEn (small) | 839 | 1582 | 318 | 3939 |
| RaVAEn (medium) | 797 | 1383 | 360 | 4138 |
| RaVAEn (large) | 808 | 1436 | 349 | 4085 |
| STTORM-CD (small, variable) | 804 | 460 | 353 | 5061 |
| STTORM-CD (medium, variable) | 867 | 489 | 290 | 5032 |
| STTORM-CD (large, variable) | 749 | 465 | 408 | 5056 |
| STTORM-CD (small, fixed) | 818 | 711 | 339 | 4810 |
| STTORM-CD (medium, fixed) | 717 | 439 | 440 | 5082 |
| STTORM-CD (large, fixed) | 743 | 514 | 414 | 5007 |
| Index | 734 | 587 | 423 | 4934 |
| Cosine baseline | 750 | 925 | 407 | 4596 |

