import json
import os
from argparse import ArgumentParser, Namespace
from typing import Any, Iterable, Union

TEST_DATASET_NAME = [
    "aida-b",
    "tweeki",
    "reddit-comments",
    "reddit-posts",
    "wned-wiki",
    "cweb",
    "shadowlinks-top",
    "shadowlinks-tail",
    "shadowlinks-shadow"
]


def read_conll(
        file: Union[str, bytes, os.PathLike],
        delimiter: str = ' ',
        word_column: int = 0,
        tag_column: int = 1,
        link_column: int = 2
    ) -> Iterable[tuple[str, list[dict[str, Any]]]]:
    sentences: list[dict[str, Any]] = []
    words: list[str] = []
    labels: list[str] = []
    links: list[str] = []
    id = ""

    with open(file, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("-DOCSTART-"):
                if sentences:
                    yield id, sentences
                    sentences = []
            elif line.startswith("# "):
                id = line[2:].strip().split('\t')[0]
            elif not line:
                if words:
                    sentences.append(_conll_to_example(words, labels, links))
                    words = []
                    labels = []
                    links = []
            else:
                cols = line.split(delimiter)
                words.append(cols[word_column])
                labels.append(cols[tag_column])
                links.append(cols[link_column])

    if sentences:
        yield id, sentences


def _conll_to_example(words: list[str], tags: list[str], links: list[str]) -> dict[str, Any]:
    text, positions = _conll_words_to_text(words)
    entities = [
        {"start": positions[start][0], "end": positions[end - 1][1], "label": [label], "title": [title], 'text': text[positions[start][0]: positions[end - 1][1]]}
        for start, end, label, title in _conll_tags_to_spans(tags, links)
    ]
    return {"text": text, "entities": entities}


def _conll_words_to_text(words: Iterable[str]) -> tuple[str, list[tuple[int, int]]]:
    text = ""
    positions = []
    offset = 0
    for word in words:
        if text:
            text += " "
            offset += 1
        text += word
        n = len(word)
        positions.append((offset, offset + n))
        offset += n
    return text, positions


def _conll_tags_to_spans(tags: Iterable[str], links: Iterable[str]) -> Iterable[tuple[int, int, str, str]]:
    # NOTE: assume BIO scheme
    start, label = -1, None
    for i, (tag, link) in enumerate(zip(list(tags) + ["O"], list(links) + ["O"])):
        if tag == "O":
            if start >= 0:
                assert label is not None
                yield (start, i, label, link)
                start, label = -1, None
        else:
            cur_label = tag[2:]
            if tag.startswith("B"):
                if start >= 0:
                    assert label is not None
                    yield (start, i, label, link)
                start, label = i, cur_label
            else:
                if cur_label != label:
                    if start >= 0:
                        assert label is not None
                        yield (start, i, label, link)
                    start, label = i, cur_label


def _convert_conll_to_ours(input_file: str) -> tuple[list[dict], list[tuple[str, str]]]:
    """
    Convert a CoNLL formatted file to our specific format.

    Args:
        input_file (str): Path to the input CoNLL file.

    Returns:
        tuple: A list of dictionaries in our format and a list of titles.
    """
    dataset = []
    titles = []

    for i, (id, sentences) in enumerate(read_conll(input_file, delimiter='\t')):
        examples = []
        id = str(i) if not id else str(id)
        for si, sentence in enumerate(sentences):
            text = sentence['text']
            entities = sentence['entities']
            titles.extend([(ent['label'][0], ent['title'][0]) for ent in sentence['entities']])
            examples.append({
                "id": f"{id}-{si}",
                "text": text,
                "entities": entities,
            })
        dataset.append({"id": id, "examples": examples})

    return dataset, titles


def convert_data_to_ours(args: Namespace) -> list[tuple[str, str]]:
    """
    Convert a JSONL file to our specific format.

    Args:
        args (Namespace): Command line arguments containing input and output file paths.
    """
    input_dir = args.input_dir
    output_dir = args.output_dir
    all_titles = []

    input_file = os.path.join(input_dir, "train_data/zelda_train.conll")
    output_path = os.path.join(output_dir, "zelda_train.jsonl")
    print(f"Processing {input_file} to {output_path}")
    dataset, titles = _convert_conll_to_ours(input_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    all_titles.extend(titles)

    for dname in TEST_DATASET_NAME:
        input_file = os.path.join(input_dir, f"test_data/conll/test_{dname}.conll")
        output_path = os.path.join(output_dir, f"test_{dname}.jsonl")
        print(f"Processing {input_file} to {output_path}")
        dataset, titles = _convert_conll_to_ours(input_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        all_titles.extend(titles)

    return list(set(all_titles))


def convert_description_to_dictionary(args: Namespace, titles: list[tuple[str, str]]) -> None:
    """
    Convert a description file to a dictionary format.

    Args:
        args (Namespace): Command line arguments containing input and output file paths.
    """
    description_file = os.path.join(args.input_dir, "other/entity_descriptions.jsonl")
    output_path = os.path.join(args.output_dir, "zelda_dictionary.jsonl")

    descriptions = {}
    for line in open(description_file, 'r', encoding='utf-8'):
        item = json.loads(line.strip())
        descriptions[str(item['wikipedia_id'])] = {
            "name": item["wikipedia_title"],
            "description": item.get("description", "")
        }

    with open(output_path, 'w', encoding='utf-8') as f:
        for id, title in titles:
            if id in descriptions:
                f.write(json.dumps({
                    "id": str(id),
                    "name": descriptions[id]["name"],
                    "description": descriptions[id]["description"]
                }, ensure_ascii=False) + '\n')
            else:
                f.write(json.dumps({
                    "id": str(id),
                    "name": title,
                    "description": ""
                }, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    parser = ArgumentParser(description="Zelda Entity Disambiguation Script")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--output_dir", "-o", type=str, default='./zelda', help="Path to the output text file")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    titles = convert_data_to_ours(args)
    convert_description_to_dictionary(args, titles)
