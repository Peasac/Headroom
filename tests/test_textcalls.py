import pytest

from headroom.runner.textcalls import detect_text_tool_call


@pytest.mark.parametrize("text", [
    '{"name": "cat", "parameters": {"file": "shopping.txt"}}',  # seen from llama3.2:3b
    'Sure! {"name": "add_event", "arguments": {"title": "x"}}',
    '```json\n{"type": "function", "name": "read_file", "parameters": {"path": "a"}}\n```',
    '[{"name": "read_file", "arguments": "{\\"path\\": \\"a\\"}"}]',
    '{"function": {"name": "read_file", "arguments": {"path": "a"}}}',
    "<tool_call>read_file a.txt</tool_call>",
    "<|python_tag|>read_file(path='a.txt')",
])
def test_detects_tool_calls_written_as_text(text):
    assert detect_text_tool_call(text)


@pytest.mark.parametrize("text", [
    "",
    "Moved the file to archive/plan.md.",
    "I added {milk} to the list.",
    'The note says {"title": "Ideas"} in JSON.',
    '{"name": "Sam"}',  # a name but no parameters/arguments
    "Booked 'Sync with Sam' from 11:00 to 12:00.",
])
def test_ordinary_replies_are_not_flagged(text):
    assert detect_text_tool_call(text) is None
