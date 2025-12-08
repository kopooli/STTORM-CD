# Metrics & Confusion Matrices (avg memory)

## AURC

| Model | RaVAEn-Landslides | RaVAEn-Wildfires | RaVAEn-Hurricanes | RaVAEn-Floods | STTORM-Floods |
|---|---|---|---|---|---|
| RaVAEn (small) | 86.71 | 87.88 | 80.05 | 77.83 | 50.00 |
| RaVAEn (medium) | 88.21 | 88.67 | 78.99 | 76.60 | 50.50 |
| RaVAEn (large) | 87.24 | 89.15 | 79.10 | 74.70 | 48.54 |
| STTORM-CD (small, variable) | 68.27 | 84.52 | 68.06 | 87.01 | 86.11 |
| STTORM-CD (medium, variable) | 74.81 | 82.55 | 71.96 | 86.61 | 86.54 |
| STTORM-CD (large, variable) | 80.86 | 87.47 | 73.32 | 85.87 | 83.99 |
| STTORM-CD (small, fixed) | 63.88 | 83.67 | 62.85 | 88.63 | 85.31 |
| STTORM-CD (medium, fixed) | 82.64 | 82.69 | 69.58 | 86.69 | 85.20 |
| STTORM-CD (large, fixed) | 62.67 | 78.46 | 64.95 | 84.79 | 83.94 |
| Index | 88.08 | 96.02 | 87.28 | 69.28 | 77.25 |
| Cosine baseline | 71.03 | 88.26 | 67.75 | 72.16 | 73.43 |


## RDP

| Model | RaVAEn-Landslides | RaVAEn-Wildfires | RaVAEn-Hurricanes | RaVAEn-Floods | STTORM-Floods |
|---|---|---|---|---|---|
| RaVAEn (small) | 67.89 | 99.82 | 99.13 | 98.36 | 81.87 |
| RaVAEn (medium) | 70.77 | 99.54 | 98.87 | 98.52 | 80.92 |
| RaVAEn (large) | 63.74 | 99.57 | 99.73 | 99.16 | 84.10 |
| STTORM-CD (small, variable) | 82.27 | 99.94 | 99.66 | 96.21 | 80.44 |
| STTORM-CD (medium, variable) | 71.88 | 99.81 | 98.14 | 91.12 | 88.26 |
| STTORM-CD (large, variable) | 74.12 | 99.95 | 98.89 | 92.89 | 76.43 |
| STTORM-CD (small, fixed) | 79.39 | 99.98 | 99.99 | 97.07 | 73.64 |
| STTORM-CD (medium, fixed) | 74.92 | 99.48 | 97.71 | 96.44 | 73.30 |
| STTORM-CD (large, fixed) | 92.65 | 99.96 | 99.62 | 94.56 | 77.48 |
| Index | 97.12 | 99.86 | 99.99 | 99.99 | 99.36 |
| Cosine baseline | 81.95 | 99.99 | 99.23 | 98.19 | 83.12 |


## AUPRC

| Model | RaVAEn-Landslides | RaVAEn-Wildfires | RaVAEn-Hurricanes | RaVAEn-Floods | STTORM-Floods |
|---|---|---|---|---|---|
| RaVAEn (small) | 81.05 | 88.74 | 74.00 | 71.72 | 39.62 |
| RaVAEn (medium) | 83.33 | 89.60 | 71.95 | 70.27 | 39.60 |
| RaVAEn (large) | 81.19 | 90.10 | 70.36 | 68.38 | 38.32 |
| STTORM-CD (small, variable) | 62.57 | 86.14 | 58.82 | 82.22 | 75.70 |
| STTORM-CD (medium, variable) | 67.89 | 83.90 | 65.45 | 79.50 | 76.12 |
| STTORM-CD (large, variable) | 73.25 | 88.52 | 66.52 | 80.91 | 72.26 |
| STTORM-CD (small, fixed) | 60.43 | 85.03 | 50.68 | 81.94 | 72.96 |
| STTORM-CD (medium, fixed) | 75.80 | 83.64 | 63.10 | 82.08 | 73.77 |
| STTORM-CD (large, fixed) | 64.74 | 80.55 | 54.87 | 81.14 | 73.57 |
| Index | 77.85 | 95.83 | 83.53 | 54.73 | 59.86 |
| Cosine baseline | 58.16 | 88.60 | 64.21 | 61.34 | 57.99 |


## F1

| Model | RaVAEn-Landslides | RaVAEn-Wildfires | RaVAEn-Hurricanes | RaVAEn-Floods | STTORM-Floods |
|---|---|---|---|---|---|
| RaVAEn (small) | 76.19 | 87.66 | 65.52 | 67.21 | 48.15 |
| RaVAEn (medium) | 77.22 | 88.39 | 63.94 | 65.79 | 48.60 |
| RaVAEn (large) | 73.39 | 88.66 | 62.26 | 64.67 | 47.03 |
| STTORM-CD (small, variable) | 67.83 | 87.54 | 50.79 | 72.24 | 69.35 |
| STTORM-CD (medium, variable) | 68.68 | 85.16 | 57.42 | 72.37 | 71.15 |
| STTORM-CD (large, variable) | 72.22 | 87.55 | 58.24 | 72.67 | 66.67 |
| STTORM-CD (small, fixed) | 67.80 | 85.94 | 44.30 | 72.48 | 66.96 |
| STTORM-CD (medium, fixed) | 74.29 | 83.22 | 55.82 | 73.24 | 67.05 |
| STTORM-CD (large, fixed) | 74.27 | 83.89 | 47.73 | 72.94 | 67.03 |
| Index | 73.39 | 89.30 | 76.60 | 53.28 | 59.96 |
| Cosine baseline | 54.88 | 85.55 | 61.71 | 65.20 | 55.74 |


## Precision

| Model | RaVAEn-Landslides | RaVAEn-Wildfires | RaVAEn-Hurricanes | RaVAEn-Floods | STTORM-Floods |
|---|---|---|---|---|---|
| RaVAEn (small) | 80.00 | 85.87 | 63.40 | 68.85 | 39.69 |
| RaVAEn (medium) | 78.74 | 86.86 | 63.02 | 61.24 | 40.85 |
| RaVAEn (large) | 78.45 | 87.34 | 70.73 | 62.68 | 39.01 |
| STTORM-CD (small, variable) | 62.99 | 84.21 | 45.69 | 70.74 | 66.02 |
| STTORM-CD (medium, variable) | 68.42 | 83.57 | 49.67 | 70.24 | 66.64 |
| STTORM-CD (large, variable) | 75.83 | 86.27 | 51.03 | 73.29 | 61.57 |
| STTORM-CD (small, fixed) | 61.35 | 83.13 | 44.22 | 75.59 | 62.19 |
| STTORM-CD (medium, fixed) | 70.27 | 81.73 | 50.91 | 85.69 | 63.54 |
| STTORM-CD (large, fixed) | 65.14 | 79.76 | 46.54 | 82.12 | 65.75 |
| Index | 93.02 | 93.12 | 81.02 | 54.86 | 56.50 |
| Cosine baseline | 45.92 | 82.82 | 51.57 | 61.54 | 54.50 |


## Recall

| Model | RaVAEn-Landslides | RaVAEn-Wildfires | RaVAEn-Hurricanes | RaVAEn-Floods | STTORM-Floods |
|---|---|---|---|---|---|
| RaVAEn (small) | 72.73 | 89.52 | 67.79 | 65.65 | 61.19 |
| RaVAEn (medium) | 75.76 | 89.97 | 64.89 | 71.07 | 59.98 |
| RaVAEn (large) | 68.94 | 90.02 | 55.59 | 66.79 | 59.20 |
| STTORM-CD (small, variable) | 73.48 | 91.15 | 57.17 | 73.80 | 73.03 |
| STTORM-CD (medium, variable) | 68.94 | 86.80 | 68.04 | 74.63 | 76.32 |
| STTORM-CD (large, variable) | 68.94 | 88.88 | 67.82 | 72.05 | 72.69 |
| STTORM-CD (small, fixed) | 75.76 | 88.93 | 44.37 | 69.62 | 72.52 |
| STTORM-CD (medium, fixed) | 78.79 | 84.77 | 61.77 | 63.95 | 70.96 |
| STTORM-CD (large, fixed) | 86.36 | 88.47 | 48.98 | 65.60 | 68.37 |
| Index | 60.61 | 85.78 | 72.64 | 51.78 | 63.87 |
| Cosine baseline | 68.18 | 88.47 | 76.84 | 69.31 | 57.04 |



---

## Confusion Matrices

### RaVAEn-Landslides

| Model | True Positive | False Positive | False Negative | True Negative |
|---|---|---|---|---|
| RaVAEn (small) | 96 | 24 | 36 | 470 |
| RaVAEn (medium) | 100 | 27 | 32 | 467 |
| RaVAEn (large) | 91 | 25 | 41 | 469 |
| STTORM-CD (small, variable) | 97 | 57 | 35 | 437 |
| STTORM-CD (medium, variable) | 91 | 42 | 41 | 452 |
| STTORM-CD (large, variable) | 91 | 29 | 41 | 465 |
| STTORM-CD (small, fixed) | 100 | 63 | 32 | 431 |
| STTORM-CD (medium, fixed) | 104 | 44 | 28 | 450 |
| STTORM-CD (large, fixed) | 114 | 61 | 18 | 433 |
| Index | 80 | 6 | 52 | 488 |
| Cosine baseline | 90 | 106 | 42 | 388 |


### RaVAEn-Wildfires

| Model | True Positive | False Positive | False Negative | True Negative |
|---|---|---|---|---|
| RaVAEn (small) | 14327 | 2357 | 1677 | 9504 |
| RaVAEn (medium) | 14398 | 2178 | 1606 | 9683 |
| RaVAEn (large) | 14407 | 2089 | 1597 | 9772 |
| STTORM-CD (small, variable) | 14587 | 2735 | 1417 | 9126 |
| STTORM-CD (medium, variable) | 13891 | 2730 | 2113 | 9131 |
| STTORM-CD (large, variable) | 14224 | 2264 | 1780 | 9597 |
| STTORM-CD (small, fixed) | 14233 | 2888 | 1771 | 8973 |
| STTORM-CD (medium, fixed) | 13567 | 3033 | 2437 | 8828 |
| STTORM-CD (large, fixed) | 14159 | 3593 | 1845 | 8268 |
| Index | 13729 | 1015 | 2275 | 10846 |
| Cosine baseline | 14158 | 2936 | 1846 | 8925 |


### RaVAEn-Hurricanes

| Model | True Positive | False Positive | False Negative | True Negative |
|---|---|---|---|---|
| RaVAEn (small) | 2151 | 1242 | 1022 | 7358 |
| RaVAEn (medium) | 2059 | 1208 | 1114 | 7392 |
| RaVAEn (large) | 1764 | 730 | 1409 | 7870 |
| STTORM-CD (small, variable) | 1814 | 2156 | 1359 | 6444 |
| STTORM-CD (medium, variable) | 2159 | 2188 | 1014 | 6412 |
| STTORM-CD (large, variable) | 2152 | 2065 | 1021 | 6535 |
| STTORM-CD (small, fixed) | 1408 | 1776 | 1765 | 6824 |
| STTORM-CD (medium, fixed) | 1960 | 1890 | 1213 | 6710 |
| STTORM-CD (large, fixed) | 1554 | 1785 | 1619 | 6815 |
| Index | 2305 | 540 | 868 | 8060 |
| Cosine baseline | 2438 | 2290 | 735 | 6310 |


### RaVAEn-Floods

| Model | True Positive | False Positive | False Negative | True Negative |
|---|---|---|---|---|
| RaVAEn (small) | 1273 | 576 | 666 | 8738 |
| RaVAEn (medium) | 1378 | 872 | 561 | 8442 |
| RaVAEn (large) | 1295 | 771 | 644 | 8543 |
| STTORM-CD (small, variable) | 1431 | 592 | 508 | 8722 |
| STTORM-CD (medium, variable) | 1447 | 613 | 492 | 8701 |
| STTORM-CD (large, variable) | 1397 | 509 | 542 | 8805 |
| STTORM-CD (small, fixed) | 1350 | 436 | 589 | 8878 |
| STTORM-CD (medium, fixed) | 1240 | 207 | 699 | 9107 |
| STTORM-CD (large, fixed) | 1272 | 277 | 667 | 9037 |
| Index | 1004 | 826 | 935 | 8488 |
| Cosine baseline | 1344 | 840 | 595 | 8474 |


### STTORM-Floods

| Model | True Positive | False Positive | False Negative | True Negative |
|---|---|---|---|---|
| RaVAEn (small) | 708 | 1076 | 449 | 4445 |
| RaVAEn (medium) | 694 | 1005 | 463 | 4516 |
| RaVAEn (large) | 685 | 1071 | 472 | 4450 |
| STTORM-CD (small, variable) | 845 | 435 | 312 | 5086 |
| STTORM-CD (medium, variable) | 883 | 442 | 274 | 5079 |
| STTORM-CD (large, variable) | 841 | 525 | 316 | 4996 |
| STTORM-CD (small, fixed) | 839 | 510 | 318 | 5011 |
| STTORM-CD (medium, fixed) | 821 | 471 | 336 | 5050 |
| STTORM-CD (large, fixed) | 791 | 412 | 366 | 5109 |
| Index | 739 | 569 | 418 | 4952 |
| Cosine baseline | 660 | 551 | 497 | 4970 |

