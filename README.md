# NLP-Named-Entity-Recognition

> This is my first time writing the NER, so there might be many mistakes, sorry!!

## Testing

 * using `conlleval.py` to evaluate the performance of the model
 * detail link: https://github.com/sighsmile/conlleval
 
 ### My Best Dev Result
 ```
 accuracy:  29.27%; (non-O)
 accuracy:  94.37%; precision:  45.62%; recall:  30.05%; FB1:  36.23
           company: precision:  60.00%; recall:  30.77%; FB1:  40.68  20
          facility: precision:  37.04%; recall:  26.32%; FB1:  30.77  27
           geo-loc: precision:  45.60%; recall:  49.14%; FB1:  47.30  125
             movie: precision:   0.00%; recall:   0.00%; FB1:   0.00  4
       musicartist: precision:   0.00%; recall:   0.00%; FB1:   0.00  3
             other: precision:   5.56%; recall:   1.53%; FB1:   2.40  36
            person: precision:  60.40%; recall:  52.94%; FB1:  56.43  149
           product: precision:  18.75%; recall:  16.22%; FB1:  17.39  32
        sportsteam: precision:  58.33%; recall:  30.00%; FB1:  39.62  36
            tvshow: precision:   0.00%; recall:   0.00%; FB1:   0.00  2
 ```
 
 ### Testing Score
 
  | Precision | Recall |  F1 |
  |:---------:|:------:|:---:|
  |   47.92   |  31.82 |38.24|

## Model Architecture

  1. Word Embedding
  2. BiLSTM
  3. DropOut
  4. Linear ( hidden to tag )
  5. CRF
  
 ## HyperParemeter
  
  ``` python
  HIDDEN_SIZE = 1024
  EMBEDDING_DIM = 100
  BATCH_SIZE = 4
  LSTM_LAYER = 1
  DROPOUT = 0.5
  EPOCH = 5
  ```
 
 
