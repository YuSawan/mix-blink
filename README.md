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
          "label": ["000011"]
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
    --config_file configs/config_inbatch.yaml \
    --output_dir ./initial_output/ \
    --negative False
```

#### Inbatch+Hard Negatives Training
```
python mix_blink/cli/train.py \
    --config_file configs/config_hard.yaml \
    --output_dir ./second_output/ \
    --prev_path ./initial_output/ \
    --negative True \
```

### Build Index
```
python mix_blink/cli/build_index.py \
    --config_file configs/config_hard.yaml \
    --output_dir ./retriever/ \
    --prev_path ./initial_output/
```

### Retrieve Hard Negatives/Candidates
```
python mix_blink/cli/get_candidates.py \
    --input_file ./datasets/dataset.jsonl \
    --dictionary_path ./datasets/dictionary.jsonl \
    --output_dir ./dataset_candidates/ \
    --model_path ./initial_output/ \
    --retriever_dir ./retriever/ \
```

### Evaluation/Prediction
```
python mix_blink/cli/eval.py \
    --input_file ./datasets/dataset.jsonl \
    --dictionary_path ./datasets/dictionary.jsonl \
    --config_file configs/config.yaml \
    --prev_path PATH_TO_YOUR_MODEL \
    --output_dir PATH_TO_YOUR_MODEL
```
