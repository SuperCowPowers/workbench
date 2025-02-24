# Fun with Endpoints

This is a blog about timing, combining, and choosing AWS® endpoints.

## Endpoints

| Endpoint                | Instance      | Price/Hour |
|-------------------------|--------------|-------|
| tautomerize-v0-rt      | ml.t2.medium | $0.06 |
| smiles-to-md-v0-rt     | ml.t2.medium | $0.06 |
| aqsol-mol-class-rt     | ml.t2.medium | $0.06 |
| pipeline-model         | ml.t2.medium | $0.06 |
| pipeline-model-fast    | ml.c7i.xlarge | $0.21 |

## Timing results (fast_inference direct)

```
Timing for 10 rows
Individual endpoints Total: 2.083378314971924 seconds
Taut: 1.0616083145141602 seconds, MD: 0.5406620502471924 seconds, Model: 0.48110198974609375 seconds
Pipeline endpoint: 1.445024013519287 seconds
Pipeline Fast endpoint: 1.2368178367614746 seconds


Timing for 100 rows
Individual endpoints Total: 3.918509006500244 seconds
Taut: 2.0423460006713867 seconds, MD: 1.1638188362121582 seconds, Model: 0.7123398780822754 seconds
Pipeline endpoint: 2.8939127922058105 seconds
Pipeline Fast endpoint: 1.852799892425537 seconds


Timing for 500 rows
Individual endpoints Total: 14.36154294013977 seconds
Taut: 9.30370306968689 seconds, MD: 4.105616092681885 seconds, Model: 0.9522147178649902 seconds
Pipeline endpoint: 12.265265226364136 seconds
Pipeline Fast endpoint: 8.951914072036743 seconds


Timing for 1000 rows
Individual endpoints Total: 23.543747663497925 seconds
Taut: 15.042346715927124 seconds, MD: 6.840778112411499 seconds, Model: 1.6606106758117676 seconds
Pipeline endpoint: 21.389296770095825 seconds
Pipeline Fast endpoint: 12.815913915634155 seconds


Timing for 10000 rows
Individual endpoints Total: 189.780255317688 seconds
Taut: 120.399178981781 seconds, MD: 59.34339618682861 seconds, Model: 10.037672281265259 seconds
Pipeline endpoint: 190.11978220939636 seconds
Pipeline Fast endpoint: 110.5272912979126 seconds

```

# Storage


```
Timing for 10 rows
Individual endpoints Total: 1.5638809204101562 seconds
Taut: 0.4830169677734375 seconds, MD: 0.5586769580841064 seconds, Model: 0.5221822261810303 seconds
Pipeline endpoint: 0.9330408573150635 seconds
Pipeline Fast endpoint: 1.0244662761688232 seconds

Timing for 100 rows
Individual endpoints Total: 3.009629011154175 seconds
Taut: 1.0144999027252197 seconds, MD: 1.3848729133605957 seconds, Model: 0.6102499961853027 seconds
Pipeline endpoint: 2.384669065475464 seconds
Pipeline Fast endpoint: 1.4272871017456055 seconds

Timing for 500 rows
Individual endpoints Total: 12.45196795463562 seconds
Taut: 7.562067985534668 seconds, MD: 3.9801008701324463 seconds, Model: 0.9097919464111328 seconds
Pipeline endpoint: 10.858699083328247 seconds
Pipeline Fast endpoint: 6.549439907073975 seconds

```
## Timing results (without instantiation)

```
Timing for 10 rows
Individual endpoints Total: 1.4224412441253662 seconds
Taut: 0.46330809593200684 seconds, MD: 0.47082972526550293 seconds, Model: 0.4882962703704834 seconds
Pipeline endpoint: 0.9333341121673584 seconds
Pipeline Fast endpoint: 0.7804160118103027 seconds

Timing for 100 rows
Individual endpoints Total: 4.0739476680755615 seconds
Taut: 2.0824239253997803 seconds, MD: 1.3490509986877441 seconds, Model: 0.6424648761749268 seconds
Pipeline endpoint: 2.9797439575195312 seconds
Pipeline Fast endpoint: 1.944951057434082 seconds

Timing for 500 rows
Individual endpoints Total: 12.776091814041138 seconds
Taut: 8.085582971572876 seconds, MD: 3.8499021530151367 seconds, Model: 0.8406000137329102 seconds
Pipeline endpoint: 10.98793077468872 seconds
Pipeline Fast endpoint: 6.76329493522644 seconds

```

## Timing results (with instantiation)

```
Timing for 10 rows
Individual endpoints Total: 6.6 seconds
Taut: 2.05 seconds, MD: 2.32 seconds, Model: 2.27 seconds
Pipeline endpoint: 2.347784996032715 seconds
Pipeline Fast endpoint: 1.9402740001678467 seconds

Timing for 100 rows
Individual endpoints Total: 7.686334848403931 seconds
Taut: 2.3493008613586426 seconds, MD: 2.758458137512207 seconds, Model: 2.578571081161499 seconds
Pipeline endpoint: 3.0636699199676514 seconds
Pipeline Fast endpoint: 2.204558849334717 seconds

Timing for 500 rows
Individual endpoints Total: 14.027569770812988 seconds
Taut: 6.019566774368286 seconds, MD: 5.3194098472595215 seconds, Model: 2.6885838508605957 seconds
Pipeline endpoint: 8.654812097549438 seconds
Pipeline Fast endpoint: 6.5653462409973145 seconds



® Amazon Web Services (AWS) is a trademark of Amazon.com, Inc. or its affiliates.