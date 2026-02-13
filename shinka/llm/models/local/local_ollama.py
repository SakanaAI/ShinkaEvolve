import backoff
import openai
import logging
import re  # 正規表現を追加
from ..result import QueryResult

logger = logging.getLogger(__name__)

def backoff_handler(details):
    logger.info(
        f"Ollama - Retry {details['tries']} due to error: {details.get('exception')}. "
        f"Waiting {details['wait']:0.1f}s..."
    )
    
@backoff.on_exception(
    backoff.expo,
    (openai.APIConnectionError, openai.APITimeoutError),
    max_tries=3,
    on_backoff=backoff_handler,
)
def query_local_ollama(
    client,
    model,
    msg,
    system_msg,
    msg_history,
    output_model=None,
    model_posteriors=None,
    **kwargs,
) -> QueryResult:
    
    # --- モデル名のクリーニング処理を追加 ---
    # もし model が "local-NAME-http://..." 形式なら "NAME" だけを取り出す
    if model.startswith("local-"):
        pattern = r"https?://"
        match = re.search(pattern, model)
        if match:
            url_start = match.start()
            # "local-" (6文字) の後から、URLの前のハイフンまでを抽出
            model = model[6:url_start-1]
    # ---------------------------------------

    new_msg_history = msg_history + [{"role": "user", "content": msg}]
    
    response = client.chat.completions.create(
        model=model,  # ここで掃除された名前（例: local-model）が送られる
        messages=[
            {"role": "system", "content": system_msg},
            *new_msg_history,
        ],
        **kwargs,
        n=1,
    )

    content = response.choices[0].message.content
    thought = getattr(response.choices[0].message, "reasoning_content", "")
    new_msg_history.append({"role": "assistant", "content": content})

    return QueryResult(
        content=content,
        msg=msg,
        system_msg=system_msg,
        new_msg_history=new_msg_history,
        model_name=model,
        kwargs=kwargs,
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        cost=0.0,
        input_cost=0.0,
        output_cost=0.0,
        thought=thought,
        model_posteriors=model_posteriors,
    )