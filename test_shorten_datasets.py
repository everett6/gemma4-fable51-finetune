"""Synthetic checks; no downloads, model weights, or GPU."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest

from shorten_datasets import encode_trace, normalize_row, trace_identity, write_source


class FakeGemma:
    def apply_chat_template(self, messages, **kwargs):
        return "<bos>" + "".join(f"<|turn>{'model' if m['role'] == 'assistant' else m['role']}\n{m['content']}<turn|>\n"
                                 for m in messages)

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text], "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class ShorteningTests(unittest.TestCase):
    def test_sharegpt_and_messages_have_same_identity(self):
        arcee = {"conversations": [{"from": "human", "value": "Question"}, {"from": "gpt", "value": "Answer"}]}
        fable = {"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "Answer"}]}
        self.assertEqual(trace_identity(normalize_row(arcee), 3407, 0.05),
                         trace_identity(normalize_row(fable), 3407, 0.05))

    def test_short_and_long_traces_obey_bound_and_keep_final_answer(self):
        fake = FakeGemma()
        for answer in ("Short", "reasoning " * 100 + "FINAL ANSWER"):
            messages = [{"role": "user", "content": "Question"}, {"role": "assistant", "content": answer}]
            original_length, chunks = encode_trace(messages, fake, fake, 64, 8)
            chunks = list(chunks)
            self.assertTrue(chunks)
            self.assertTrue(all(len(c["input_ids"]) <= 64 for c in chunks))
            supervised = "".join(chr(x) for c in chunks for x in c["labels"] if x != -100)
            self.assertEqual(supervised, answer + "<turn|>\n")
            self.assertGreater(original_length, len(answer))

    def test_files_keep_trace_in_single_split_and_deduplicate_across_sources(self):
        fake = FakeGemma()
        rows = [{"messages": [{"role": "user", "content": f"Question {i}"},
                              {"role": "assistant", "content": "reasoning " * 40 + "Final"}]}
                for i in range(20)]
        args = Namespace(context=64, overlap=8, max_rows=0, seed=3407, eval_fraction=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seen = set()
            stats = write_source(rows, "fable", root, fake, fake, args, seen)
            self.assertEqual(stats["accepted_traces"], 20)
            trace_sets = []
            for split in ("train", "validation"):
                examples = [json.loads(line) for line in (root / "fable" / f"{split}.jsonl").read_text().splitlines()]
                self.assertTrue(examples)
                self.assertTrue(all(len(e["input_ids"]) <= 64 for e in examples))
                trace_sets.append({e["trace_id"] for e in examples})
            self.assertFalse(trace_sets[0] & trace_sets[1])
            unique = {"conversations": [{"from": "human", "value": "New question"},
                                         {"from": "gpt", "value": "New answer"}]}
            other = write_source(rows[:1] + [unique, {"bad": "row"}], "arcee", root, fake, fake, args, seen)
            self.assertEqual(other["duplicates"], 1)
            self.assertEqual(other["rejected"], 1)


if __name__ == "__main__":
    unittest.main()
