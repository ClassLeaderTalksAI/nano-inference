import pytest
from pydantic import ValidationError

from nano_inference.protocol import ChatRequest, has_content, sse_data


async def test_sse_multiline_comments_and_incomplete_eof():
    async def lines():
        for line in [": ping", "event: message", 'data: {"x":', "data: 1}", "", "data: lost"]:
            yield line

    assert [data async for data in sse_data(lines())] == ['{"x":\n1}']


def test_empty_role_delta_is_not_first_content():
    assert not has_content({"choices": [{"delta": {"role": "assistant", "content": ""}}]})
    assert has_content({"choices": [{"delta": {"content": "hello"}}]})


@pytest.mark.parametrize(
    "extra",
    [{"max_tokens": 0}, {"messages": []}, {"tools": []}, {"stream_options": {"unknown": True}}],
)
def test_unsupported_fields_and_invalid_requests_are_rejected(extra):
    with pytest.raises(ValidationError):
        ChatRequest.model_validate(
            {"model": "nano-model", "messages": [{"role": "user", "content": "hello"}], **extra}
        )
