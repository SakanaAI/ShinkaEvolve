
# 🧩 Integrating Local LLMs into **ShinkaEvolve**

## 🧠 Overview

The original **ShinkaEvolve** code does **not** include built-in support for running **local LLMs**.
To enable this functionality, parts of the codebase can be modified to integrate locally hosted models.

---

## 🏗️ Code Organization

**ShinkaEvolve** uses a **modular architecture** that supports multiple **LLM providers**.
The relevant code for LLM interaction is located in the **`LLM/`** folder, which manages all model communications.
ShinkaEvolve distinguishes between two LLM types:

* **Regular LLMs**
* **Embedding LLMs**

---

## ⚙️ Adding a Regular LLM

To add support for a **regular LLM**, follow these steps. They will show an example of adding support for gpt-oss models running with unsloth, which provides an API compatible with OpenAI API (v1/completions).
This LLM can then be specified in the configuration variables:

```yaml
llm_models:
meta_llm_models:
```

---

### 🔧 Step 1: Modify the Client

The file **`client.py`** is responsible for creating clients that interact with LLMs.
Each client instance is later used to query a specific model.

To add a local model, introduce a new client configuration.
The API URL is extracted from the model name, which follows this format:

```
local-gptoss-unsloth-url
```

#### Example

```python
elif "local-gptoss-unsloth" in model_name:
    # Extract URL from model name
    pattern = r"https?://"
    match = re.search(pattern, model_name)
    if match:
        start_index = match.start()
        url = model_name[start_index:]
    else:
        raise ValueError(f"Invalid URL in model name: {model_name}")
    
    # Create OpenAI-compatible client
    client = openai.OpenAI(
        api_key="filler",
        base_url=url
    )

    # Structured output mode (if required)
    if structured_output:
        client = instructor.from_openai(
            client,
            mode=instructor.Mode.JSON,
        )
```

---

### 📁 Step 2: Create the Local Query Function

Inside the **`models/`** folder, create a new subfolder to store the query functions for your local models:

```
LLM/models/local/
```

> Don’t forget to include an empty `__init__.py` file.

This folder should contain a **custom query function** for the local model. I called my file local_gptoss_unsloth.py.
It should follow the same structure as other functions in `LLM/models/`, but with small adjustments.

#### My Key Adjustments

* Replace `max_output_tokens` with **`max_tokens`** to match the local API.
* Extract additional response metadata such as:

  * `total_tokens`
  * `thinking_tokens` (if your model includes reasoning traces)

This function is later imported and registered in **`query.py`**.

---

### 🧩 Step 3: Update `__init__.py`

Configure **`__init__.py`** to include and expose the new local query function, so it can be imported elsewhere.

```
from .local.local_gptoss_unsloth import query_local_gptoss_unsloth            # ADDED THIS LINE
from .result import QueryResult

__all__ = [
    "query_anthropic",
    "query_openai",
    "query_deepseek",
    "query_gemini",
    "query_local_gptoss_unsloth",              # ADDED THIS LINE
    "QueryResult",
]
```

---

### 📬 Step 4: Update `query.py`

Import and register the new local query function in query.py.

#### Imports

```python
from .models import (
    query_anthropic,
    query_openai,
    query_deepseek,
    query_gemini,
    query_local_gptoss_unsloth,  # ADDED THIS LINE
    QueryResult,
)
```

#### Model Selection Logic

```python
elif "local-gptoss-unsloth" in model_name:  # ADDED THIS LINE
    query_fn = query_local_gptoss_unsloth
```

---

### 🧠 Step 5: Other Observations

The file **`query.py`** also defines functions such as:

* `sample_model_kwargs`
* `sample_batch_kwargs`

However, these are **not referenced anywhere else** in the repository, so no modifications are required here for now.

---

### ✅ Summary

| Step | File                                         | Change               | Description                                              |
| ---- | -------------------------------------------- | -------------------- | -------------------------------------------------------- |
| 1    | `client.py`                                  | Add new client block | Create OpenAI-compatible client for local LLM            |
| 2    | `models/local/query_local_gptoss_unsloth.py` | New function         | Query local model, adjust tokens, extract reasoning info |
| 3    | `__init__.py`                                | Add import           | Expose new query function                                |
| 4    | `query.py`                                   | Register model       | Add conditional for local LLM                            |
| 5    | —                                            | Review only          | Ignored unused functions                                 |

---

## 🧬 Adding a Local Embedding Model

For embedding models, you can use **Ollama**, which follows the **OpenAI API** format.
The only relevant file is **`embedding.py`**.

### Code Addition

```python
elif model_name.startswith("local-"):
    # Pattern: local-(model-name)-(http or https url)
    match = re.match(r"local-(.+?)-(https?://.+)", model_name)
    if match:
        model_to_use = match.group(1)
        url = match.group(2)
    else:
        raise ValueError(f"Invalid local model format: {model_name}")

    client = openai.OpenAI(
        base_url=url,
        api_key="filler"
    )
```

#### Notes

* Compatible with **any Ollama model**.
* The model name must follow this convention:

  ```
  local-model-name-url
  ```
* The code extracts both `model-name` and `url`, and uses them to query Ollama.

---

### Query Logic

The existing line in **`embedding.py`** remains unchanged:

```python
response = self.client.embeddings.create(
    model=self.model,
    input=code,
    encoding_format="float"
)
```

For local embedding models, `self.model` corresponds to the extracted model name.
The only addition to the **Embedding Client** class:

```python
elif self.model_name.startswith("local-"):
    cost = 0.0
```

---

## 🚀 Result

ShinkaEvolve can now connect to **locally hosted LLMs** and **embedding models** through **OpenAI-compatible APIs**.
This setup supports **Ollama** and other frameworks such as **gpt-oss** under **Unsloth**.

If your model has different requirements, follow the same pattern with a distinct model identifier and your own custom logic.

