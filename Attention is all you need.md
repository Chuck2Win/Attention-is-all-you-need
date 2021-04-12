# Attention is all you need

## 구현

![Transformer (Attention Is All You Need) 구현하기 (1/3) | Reinforce NLP](https://paul-hyun.github.io/assets/2019-12-19/transformer-model-architecture.png)

직접 짜보면서 깨닫게 됬는데, Decoder 부분에서 Masked Multi-Head Attention , Encoder-Decoder Attention, FFN이 N번 반복된다. (즉, Encoder - Decoder Attention이 1번만 하는 줄 착각?하고 있었던 것 같음..)

- 결국 Encoder Output은 여러번 참조된다고 할 수 있겠군

:ballot_box_with_check: 또한 Encoder-Decoder Attention에서 Query는 Decoder, Key, Value는 Encoder

:heavy_check_mark: 직접 짜면서 배운점

1. Mask 에 대해서 기존에 pytorch것을 가져다 쓰면서 조금 헷갈렸는데(거기에선 padding mask는 additive(before Softmax)(-inf)  그리고 subsequent mask는 0로 했었음) 하지만 단순하게 둘이 더해서 동일하게 취급하면 됨(Softmax 취하기 전에 해당 위치에 -inf를 취하면 됨)
2. Encoder Decoder Attention 역시도 여러번 계산된다. 즉 위 그림을 고대로 생각하면 됨
3. `lambda`의 편리함을 다시끔 깨달음
4. layer connection :muscle:



## 참조

https://paul-hyun.github.io/transformer-01/

https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/attention/single.py

감사합니다!