# mix-blink
A library for mixed bi-encoding entity linking models

## Usage

### Instllation
```
git clone git@github.com:YuSawan/mix-blink.git
cd mix-blink
python -m venv .venv
source .venv/bin/activate
pip install .
```

### Dataset preparation
#### Dataset
```
{
  "id": "doc-001",
  "examples": [
    {
      "paragraph-id": "doc-001-P1",
      "text": "She graduated from NAIST.",
      "entities": [
        {
          "start": 19,
          "end": 24,
          "label": ["000011"],
          "hard_negatives": ["000012", "000013", "000014"] # Need if training with hard negatives
        }
      ],
    }
  ]
}
```

#### Dictionary
```
{
  "id": "000011",
  "name": "NAIST",
  "description": "NAIST is located in Ikoma."
}
```

### Finetuning

#### Inbatch Training
```
python mix_blink/cli/train.py \
    --config_file configs/config.yaml \
    --dictionary_file datasets/dictionary.jsonl \
    --train_file datasets/train.jsonl \
    --output_dir ./initial_output/
```

#### Inbatch+Hard Negatives Training
```
python mix_blink/cli/train.py \
    --model_path ./initial_model/ # Optional
    --config_file configs/config.yaml \
    --dictionary_file datasets/dictionary.jsonl \
    --train_file datasets_with_candidates/train.jsonl \
    --output_dir ./second_output/ \
    --hard_negative
```

### Build Index
```
mkdir ./retriever/
python mix_blink/cli/build_index.py \
    --model_path ./initial_model/ # Optional
    --config_file configs/config.yaml \
    --dictionary_file datasets/dictionary.jsonl \
    --output_dir ./retriever/
```

### Retrieve Hard Negatives/Candidates
```
mkdir ./datasets_with_candidates/
python mix_blink/cli/get_candidates.py \
    --model_path ./initial_model \ # Optional
    --config_file configs/config.yaml \
    --dictionary_path ./datasets/dictionary.jsonl
    --input_file ./datasets/train.jsonl \
    --output_dir datasets_with_candidates \
    --retriever_path ./retriever/ \
```

### Evaluation/Prediction
```
python mix_blink/cli/eval.py \
    --model_path ./initial_model/ \ # Optional
    --config_file configs/config.yaml \
    --dictionary_path ./datasets/dictionary.jsonl \
    --retriever_path ./retriever/
    --input_file ./datasets/test.jsonl \
    --output_dir ./initial_model/
```
