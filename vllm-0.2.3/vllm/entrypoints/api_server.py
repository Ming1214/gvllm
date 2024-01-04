import argparse
import json
from typing import AsyncGenerator

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()
engine = None


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/tokenize_text_count")
async def tokenize_count(req: Request) -> Response:
    """Count number of tokens."""
    req_dict = await req.json()
    text = req_dict.pop("text", "")
    tokens = engine.engine.tokenizer.tokenize(text)
    ret = {"count": len(tokens)}
    return JSONResponse(ret)


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)

    use_chat = request_dict.pop("chat", False)
    meta_info = request_dict.pop("meta", "You are an AI assistant. Your response should be helpful, harmless and honest.")
    if use_chat:
        prompt = f"<|meta_start|> {meta_info} <|meta_end|>\n <|start|> <|human|> {prompt} <|end|>\n <|assistant|> "
    
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()

    results_generator = engine.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        prev_text = ""
        async for request_output in results_generator:
            #prompt = request_output.prompt
            #text_outputs = [
            #    prompt + output.text for output in request_output.outputs
            #]
            #ret = {"text": text_outputs}
            #yield (json.dumps(ret) + "\0").encode("utf-8")
            output = request_output.outputs[0]
            token_id = output.token_ids[-1]
            text = output.text
            token = text[len(prev_text): ]
            prev_text = text
            resp = {
                "token": {
                    "id": token_id, 
                    "text": token, 
                    "logprob": None, 
                    "special": False
                }
            }
            resp["generated_text"] = text if request_output.finished else None
            yield f"data:{json.dumps(resp, ensure_ascii=False)}\n\n"

    async def abort_request() -> None:
        await engine.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks, headers={"Content-Type": "text/event-stream"})

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        #if await request.is_disconnected():
        #    # Abort the request if the client disconnects.
        #    await engine.abort(request_id)
        #    return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    #prompt = final_output.prompt
    #text_outputs = [prompt + output.text for output in final_output.outputs]
    #ret = {"text": text_outputs}
    texts = [output.text for output in request_output.outputs]
    ret = {"generated_text": texts if len(texts) > 1 else texts[0]}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=8000)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)