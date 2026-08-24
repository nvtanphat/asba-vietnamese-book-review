"""Migrated from 00_preprocessing_debug.ipynb.

This file preserves the original experiment code for audit/reproducibility.
The production benchmark uses the modular ml/ package instead.
"""


# %% [code cell 1]
import importlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from IPython.display import display

PROJECT_ROOT = None
for candidate in [Path.cwd(), *Path.cwd().parents]:
    if (candidate / 'src').is_dir():
        PROJECT_ROOT = candidate
        break
if PROJECT_ROOT is None:
    raise FileNotFoundError('Could not locate project root containing src/')
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import emoji_norm, formatters, noise_cleaner, pipeline, quality_filter, split_dataset, unicode_norm, vocab_norm

emoji_norm = importlib.reload(emoji_norm)
formatters = importlib.reload(formatters)
noise_cleaner = importlib.reload(noise_cleaner)
quality_filter = importlib.reload(quality_filter)
unicode_norm = importlib.reload(unicode_norm)
vocab_norm = importlib.reload(vocab_norm)
pipeline = importlib.reload(pipeline)
split_dataset = importlib.reload(split_dataset)


# %% [code cell 2]
pd.set_option('display.max_colwidth', 120)
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 160)


# %% [code cell 3]
def lowercase_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: value.lower() if isinstance(value, str) else value)


def stepwise_preview(series: pd.Series, min_chars: int = 5) -> pd.DataFrame:
    preview = pd.DataFrame({'raw': series})
    preview['unicode'] = unicode_norm.normalize_series(preview['raw'])
    preview['noise'] = noise_cleaner.normalize_series(preview['unicode'])
    preview['emoji'] = emoji_norm.normalize_series(preview['noise'])
    preview['vocab'] = vocab_norm.normalize_series(preview['emoji'])
    preview['format'] = formatters.normalize_series(preview['vocab'])
    preview['clean'] = lowercase_series(preview['format'])
    preview['meaningful'] = preview['clean'].map(lambda value: quality_filter.is_meaningful_text(value, min_chars=min_chars))
    preview['group_key'] = preview['clean'].map(quality_filter.normalize_for_duplicate)
    return preview


def summarize_frame(df: pd.DataFrame, text_column: str = 'content') -> pd.DataFrame:
    summary = {'rows': len(df), 'columns': len(df.columns)}
    if text_column in df.columns:
        summary['missing_text'] = int(df[text_column].isna().sum())
        summary['duplicate_normalized_texts'] = int(df[text_column].map(quality_filter.normalize_for_duplicate).duplicated().sum())
    return pd.DataFrame([summary])


# %% [code cell 4]
calls: list[str] = []


def make_step(name: str):
    def step(series: pd.Series) -> pd.Series:
        calls.append(name)
        return pd.Series([f'{series.iloc[0]}->{name}'], index=series.index)

    return step


with patch.object(pipeline.unicode_norm, 'normalize_series', make_step('unicode')):
    with patch.object(pipeline.noise_cleaner, 'normalize_series', make_step('noise')):
        with patch.object(pipeline.emoji_norm, 'normalize_series', make_step('emoji')):
            with patch.object(pipeline.vocab_norm, 'normalize_series', make_step('vocab')):
                with patch.object(pipeline.formatters, 'normalize_series', make_step('format')):
                    ordered_result = pipeline.clean_text_series(pd.Series(['start']))

display(pd.DataFrame({'result': ordered_result}))
print(calls)

assert ordered_result.tolist() == ['start->unicode->noise->emoji->vocab->format']
assert calls == ['unicode', 'noise', 'emoji', 'vocab', 'format']


# %% [code cell 5]
# Unicode
with patch.object(unicode_norm, 'fix_text', lambda text: text):
    raw = 'Cafe\u0301\x00\tA\u200bB\nC\u200dD'
    assert unicode_norm.normalize_unicode(raw) == 'Caf\u00e9\tAB\nC\u200dD'

assert unicode_norm.normalize_unicode(None) is None
assert unicode_norm.normalize_unicode(float('nan')) is None

unicode_frame = pd.DataFrame({'content': ['A' + chr(0) + 'B', None], 'other': [1, 2]})
unicode_normalized = unicode_norm.normalize_dataframe(unicode_frame, 'content', output_column='clean')
display(unicode_normalized)

assert unicode_normalized['clean'].iloc[0] == 'AB'
assert pd.isna(unicode_normalized['clean'].iloc[1])
assert unicode_frame['content'].iloc[0] == 'A' + chr(0) + 'B'
assert pd.isna(unicode_frame['content'].iloc[1])
assert 'clean' not in unicode_frame.columns

# Noise
assert noise_cleaner.strip_html('<div>Hello <b>world</b></div>') == 'Hello world'
noise_text = '<p>Visit https://example.com or email me@example.com or call +84 912 345 678.</p>'
assert noise_cleaner.normalize_noise(noise_text) == 'Visit __url__ or email __email__ or call +__phone__.'

noise_series = pd.Series(['Hello <b>world</b>', None])
noise_normalized = noise_cleaner.normalize_series(noise_series)
assert noise_normalized.iloc[0] == 'Hello world'
assert pd.isna(noise_normalized.iloc[1])


# %% [code cell 6]
# Emoji, vocab and formatters
assert emoji_norm.demojize_text('Hello :V. world :(((') == 'Hello :V. world :((('

with patch.object(emoji_norm, 'EMOJI_MAP', {}):
    assert emoji_norm.demojize_text('Nice \U0001F600') == 'Nice emoji_grinning_face'

with patch.object(emoji_norm, 'EMOJI_MAP', {'grinning_face': 'mat_cuoi'}):
    assert emoji_norm.demojize_text('Nice \U0001F600') == 'Nice mat_cuoi'

with patch.object(emoji_norm, 'EMOJI_MAP', {}):
    emoji_series = pd.Series(['A \U0001F600', None])
    emoji_normalized = emoji_norm.normalize_series(emoji_series)
    assert emoji_normalized.iloc[0] == 'A emoji_grinning_face'
    assert pd.isna(emoji_normalized.iloc[1])

with patch.object(vocab_norm, 'VOCAB_MAP', {'ko': 'khong'}):
    assert vocab_norm.normalize_vocab('ko') == 'khong'

assert vocab_norm.normalize_vocab('ngonnnn qua hayyyy') == 'ngon qua hayy'

with patch.object(vocab_norm, 'VOCAB_MAP', {}):
    vocab_series = pd.Series(['ngonnnn', None])
    vocab_normalized = vocab_norm.normalize_series(vocab_series)
    assert vocab_normalized.iloc[0] == 'ngon'
    assert pd.isna(vocab_normalized.iloc[1])

assert formatters.normalize_format('Xin\u200b chao!!!   ban???\n') == 'Xin chao!! ban??'
assert formatters.normalize_format(None) is None

format_series = pd.Series(['A\u200bB', None])
format_normalized = formatters.normalize_series(format_series)
assert format_normalized.iloc[0] == 'AB'
assert pd.isna(format_normalized.iloc[1])


# %% [code cell 7]
sample_texts = pd.Series(
    [
        'Cafe\u0301\x00\tA\u200bB\nC\u200dD',
        '<p>Visit https://example.com or email me@example.com or call +84 912 345 678.</p>',
        'Hello :V. world :(((',
        'Nice \U0001F600',
        'ngonnnn qua hayyyy',
        'Xin\u200b chao!!!   ban???\n',
        'null',
        None,
    ],
    name='raw',
)

preview = stepwise_preview(sample_texts, min_chars=5)
display(preview)

assert preview.loc[2, 'clean'] == 'hello :v. world :((('
assert preview.loc[4, 'clean'] == 'ngon qua hayy'
assert preview.loc[5, 'clean'] == 'xin chao!! ban??'
assert not preview.loc[6, 'meaningful']


# %% [code cell 8]
assert quality_filter.normalize_for_duplicate('  Xin   Chao\n') == 'xin chao'
assert not quality_filter.is_meaningful_text('null', min_chars=5)
assert not quality_filter.is_meaningful_text('123456', min_chars=5)
assert not quality_filter.is_meaningful_text('!!!', min_chars=5)
assert quality_filter.is_meaningful_text('meaningful text', min_chars=5)
assert quality_filter.is_digit_only('123456')
assert quality_filter.is_symbol_only('!!!')

noise_frame = pd.DataFrame(
    {
        'content': [
            'Hello world',
            'hello   world',
            '1234567890',
            '!!!???',
            'meaningful text',
        ],
        'label': [1, 2, 3, 4, 5],
    }
)

filtered = quality_filter.drop_noise_rows(
    noise_frame,
    text_column='content',
    min_chars=5,
    drop_duplicates=True,
)

display(noise_frame)
display(filtered)

assert filtered['content'].tolist() == ['Hello world', 'meaningful text']
assert filtered['label'].tolist() == [1, 5]
assert filtered.index.tolist() == [0, 1]


# %% [code cell 9]
frame = pd.DataFrame(
    {
        'review_id': [1, 2],
        'content': ['alpha', 'beta'],
        'sentiment': [0, 1],
        'extra': [9, 8],
    }
)

with patch.object(pipeline, 'clean_text_series', lambda series: series.str.upper()):
    transformed = pipeline.preprocess_dataframe(
        frame,
        text_column='content',
        output_column='clean_content',
        keep_raw=True,
        min_chars=1,
        drop_duplicates=False,
        keep_columns=['review_id', 'content_raw', 'clean_content'],
    )

display(transformed)

assert transformed.columns.tolist() == ['review_id', 'content_raw', 'clean_content']
assert transformed['review_id'].tolist() == [1, 2]
assert transformed['content_raw'].tolist() == ['alpha', 'beta']
assert transformed['clean_content'].tolist() == ['ALPHA', 'BETA']


# %% [code cell 10]
with TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    input_path = tmpdir / 'input.csv'
    output_path = tmpdir / 'output.csv'

    file_frame = pd.DataFrame({'review_id': [1], 'content': ['hello world'], 'sentiment': [1]})
    file_frame.to_csv(input_path, index=False, encoding='utf-8-sig')

    cleaned = pipeline.preprocess_file(
        input_path,
        output_path,
        text_column='content',
        keep_raw=True,
        min_chars=5,
        drop_duplicates=True,
        keep_columns=['review_id', 'content_raw', 'content'],
    )

    saved = pd.read_csv(output_path)
    display(cleaned)
    display(saved)

    assert output_path.exists()
    assert cleaned.columns.tolist() == ['review_id', 'content_raw', 'content']
    assert saved.columns.tolist() == ['review_id', 'content_raw', 'content']
    assert cleaned['content_raw'].tolist() == ['hello world']
    assert saved['content_raw'].tolist() == ['hello world']
    assert cleaned['content'].tolist() == ['hello world']
    assert saved['content'].tolist() == ['hello world']


# %% [code cell 11]
raw_candidates = [
    PROJECT_ROOT / 'data' / 'raw' / 'tiki-book-review.json',
    PROJECT_ROOT / 'data' / 'interim' / 'raw_train' / 'train.json',
    PROJECT_ROOT / 'data' / 'interim' / 'raw_val' / 'val.json',
    PROJECT_ROOT / 'data' / 'interim' / 'raw_test' / 'test.json',
]
raw_path = next((path for path in raw_candidates if path.exists()), None)
if raw_path is None:
    raise FileNotFoundError('No raw dataset found in data/raw or data/interim')

raw_df = pd.read_json(raw_path)
display(summarize_frame(raw_df, 'content'))
display(raw_df.head(3))

if 'sentiment' in raw_df.columns:
    display(raw_df['sentiment'].value_counts(dropna=False).sort_index())

content_rows = raw_df.loc[raw_df['content'].notna(), ['review_id', 'content', 'sentiment']].copy()
sample = content_rows.sample(n=min(12, len(content_rows)), random_state=42).reset_index(drop=True)
sample_preview = stepwise_preview(sample['content'], min_chars=quality_filter.SHORT_TEXT_MIN_CHARS)
sample_preview.insert(0, 'review_id', sample['review_id'])
sample_preview.insert(1, 'sentiment', sample['sentiment'])
display(sample_preview[['review_id', 'sentiment', 'raw', 'unicode', 'noise', 'emoji', 'vocab', 'format', 'clean', 'meaningful']])

changed = sample_preview[sample_preview['raw'].astype('string') != sample_preview['clean'].astype('string')]
if not changed.empty:
    display(changed[['review_id', 'raw', 'format', 'clean', 'meaningful']])


# %% [code cell 12]
split_frame = pd.DataFrame(
    {
        'review_id': list(range(1, 15)),
        'content': [
            'Alpha product',
            'Alpha   product ',
            'Bravo product',
            'bravo product',
            'Charlie product',
            'Delta product',
            'Echo product',
            'Foxtrot product',
            'Golf product',
            'Hotel product',
            'India product',
            'Juliet product',
            'Kilo product',
            'Lima product',
        ],
        'sentiment': [0, 0, 1, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
    }
)

with patch.object(split_dataset, 'clean_text_series', lambda series: series):
    raw_train, raw_val, raw_test = split_dataset._split_raw_rows(split_frame)

train_keys = {quality_filter.normalize_for_duplicate(value) for value in raw_train['content']}
val_keys = {quality_filter.normalize_for_duplicate(value) for value in raw_val['content']}
test_keys = {quality_filter.normalize_for_duplicate(value) for value in raw_test['content']}

split_overview = pd.DataFrame(
    {
        'split': ['train', 'val', 'test'],
        'rows': [len(raw_train), len(raw_val), len(raw_test)],
        'unique_keys': [len(train_keys), len(val_keys), len(test_keys)],
    }
)
display(split_overview)
display(pd.DataFrame([
    {
        'train_val_overlap': len(train_keys & val_keys),
        'train_test_overlap': len(train_keys & test_keys),
        'val_test_overlap': len(val_keys & test_keys),
    }
]))

assert train_keys.isdisjoint(val_keys)
assert train_keys.isdisjoint(test_keys)
assert val_keys.isdisjoint(test_keys)

duplicate_key = quality_filter.normalize_for_duplicate('Alpha product')
membership = sum(duplicate_key in split_keys for split_keys in [train_keys, val_keys, test_keys])
assert membership == 1
