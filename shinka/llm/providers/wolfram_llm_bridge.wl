(* Wolfram-side bridge for the shinka.llm.providers.wolfram_llm Python provider.

   Reads a JSON spec from <input>, opens a Wolfram service connection
   (using a passed APIKey or whatever credentials Wolfram has on file),
   calls ServiceExecute[..., "ChatService", ...] against the target LLM
   service, and writes the assistant response (or an error description) to
   <output> as JSON.

   "ChatService" rather than "Chat" because the per-service paclets'
   processResult["ChatService"] is a superset of processResult["Chat"]:
   same Content / Role / Model / FinishReason, plus a "Usage" association
   carrying real input and output token counts. GoogleGemini's "Chat"
   omits Usage; "ChatService" includes it. OpenAI / Anthropic / AzureOpenAI
   / TogetherAI map "ChatService" to "Chat" so the shape is unchanged.

   Default path uses ServiceExecute directly rather than LLMSynthesize
   because the current LLMSynthesize wrapper does a mandatory LLMKit
   subscription check that fails on script kernels not signed in to a
   Wolfram ID — even when an explicit Authentication is supplied.
   ServiceExecute is the lower-level path that the LLMServices paclet
   ultimately delegates to, and works with any Wolfram-registered LLM
   service.

   Setting "useLLMSynthesize": true in the input JSON opts into
   LLMSynthesize + LLMConfiguration instead, for users on a Wolfram setup
   where LLMKit is available. This path is not exercised by the project's
   automated tests.

   Usage:
     wolframscript -file wolfram_llm_bridge.wl <input.json> <output.json>

   Input JSON schema:
     {
       "service":          "OpenAI" | "Anthropic" | "GoogleGemini" | "DeepSeek"
                         | "Groq" | "MistralAI" | "Cohere" | "AlephAlpha"
                         | "TogetherAI",
       "model":            string,
       "messages":         [ {"role": "system"|"user"|"assistant",
                              "content": string}, ... ],
       "temperature":      number   (optional),
       "maxTokens":        integer  (optional),
       "apiKey":           string   (optional, sourced from Python-side env var
                                     if present; otherwise we fall back to
                                     whatever credentials Wolfram has stored
                                     for the service),
       "useLLMSynthesize": boolean  (optional, default false; opts into the
                                     LLMSynthesize path)
     }

   Output JSON schema (success):
     {
       "content":         string,
       "service":         string,
       "model":           string,
       "finishReason":    string  (optional),
       "inputTokens":     integer (optional, present when the service paclet
                                   surfaced Usage),
       "outputTokens":    integer (optional, same)
     }

   Output JSON schema (error):
     {
       "error":           string,
       "service":         string  (only present after the input is parsed),
       "model":           string  (only present after the input is parsed),
       "raw":             string  (optional)
     }
*)

If[Length[$ScriptCommandLine] < 3,
  WriteString["stderr",
    "Usage: wolframscript -file wolfram_llm_bridge.wl <input.json> <output.json>\n"];
  Exit[1]];

inPath  = $ScriptCommandLine[[2]];
outPath = $ScriptCommandLine[[3]];

spec = Quiet @ Check[Import[inPath, "RawJSON"], $Failed];
If[spec === $Failed,
  Export[outPath, <|"error" -> "failed to import input JSON"|>, "JSON"];
  Exit[1]];

service          = Lookup[spec, "service", $Failed];
model            = Lookup[spec, "model", $Failed];
rawMessages      = Lookup[spec, "messages", $Failed];
temperature      = Lookup[spec, "temperature", 0.7];
maxTokens        = Lookup[spec, "maxTokens", 8192];
apiKey           = Lookup[spec, "apiKey", None];
useLLMSynthesize = TrueQ[Lookup[spec, "useLLMSynthesize", False]];

If[service === $Failed || model === $Failed || rawMessages === $Failed,
  Export[outPath,
    <|"error" -> "input JSON missing one of: service, model, messages"|>, "JSON"];
  Exit[1]];

(* Wolfram chat APIs take a list of <|"Role" -> ..., "Content" -> ...|>
   associations with capitalized role names; the Python side speaks the
   OpenAI/Anthropic lowercase convention. Canonicalize here so callers
   can use either. *)
canonRole[r_String] := Switch[ToLowerCase[r],
  "system",    "System",
  "user",      "User",
  "assistant", "Assistant",
  _,           Capitalize[r]
];
toMsg[m_?AssociationQ] := <|
  "Role"    -> canonRole[ToString[Lookup[m, "role", "user"]]],
  "Content" -> ToString[Lookup[m, "content", ""]]
|>;
messages = toMsg /@ rawMessages;

If[Length[messages] == 0,
  Export[outPath,
    <|"error" -> "messages list is empty"|>, "JSON"];
  Exit[1]];

If[useLLMSynthesize,
  (* LLMSynthesize + LLMConfiguration path. Per the LLMSynthesize reference,
     the configuration is passed via the LLMEvaluator option and credentials
     via the Authentication option — the second positional argument is a
     result-property selector, not a configuration. *)
  config = LLMConfiguration[<|
    "Service"     -> service,
    "Model"       -> model,
    "Temperature" -> temperature,
    "MaxTokens"   -> maxTokens
  |>];
  (* LLMSynthesize takes a single string; collapse role-tagged messages
     into a labelled transcript. The default ChatService path passes
     the structured messages list through unchanged. *)
  syntheticInput = StringJoin @ Riffle[
    Map[(#["Role"] <> ": " <> #["Content"]) &, messages],
    "\n\n"
  ];
  response = If[StringQ[apiKey] && StringLength[apiKey] > 0,
    Check[
      LLMSynthesize[syntheticInput, LLMEvaluator -> config,
                    Authentication -> <|"APIKey" -> apiKey|>],
      $Failed
    ],
    Check[LLMSynthesize[syntheticInput, LLMEvaluator -> config], $Failed]
  ];
  If[response === $Failed || Head[response] =!= String,
    Export[outPath,
      <|"error"   -> ("LLMSynthesize failed for " <> service <>
                      " (LLMKit subscription gating is the usual cause)"),
        "raw"     -> ToString[response],
        "service" -> service,
        "model"   -> model|>, "JSON"];
    Exit[1]];
  Export[outPath,
    <|"content"      -> response,
      "service"      -> service,
      "model"        -> model,
      "finishReason" -> ""|>,
    "JSON"];
  Exit[0]
];

(* Default path: ServiceConnect + ServiceExecute["ChatService"]. With an
   explicit APIKey we use "New" to force a fresh connection bound to that
   key, bypassing Wolfram's local credential vault and the LLMKit
   subscription check. Without an APIKey we fall back to whatever Wolfram
   has on file. *)
conn = Check[
  If[StringQ[apiKey] && StringLength[apiKey] > 0,
    ServiceConnect[service, "New",
                   Authentication -> <|"APIKey" -> apiKey|>],
    ServiceConnect[service]
  ],
  $Failed
];

If[conn === $Failed || Head[conn] =!= ServiceObject,
  Export[outPath,
    <|"error"   -> ("ServiceConnect failed for " <> service <>
                    ": no usable credential (check apiKey or Wolfram-side ServiceConnect)"),
      "raw"     -> ToString[conn],
      "service" -> service,
      "model"   -> model|>, "JSON"];
  Exit[1]];

params = {
  "Messages"    -> messages,
  "Model"       -> model,
  "Temperature" -> temperature,
  "MaxTokens"   -> maxTokens
};

response = Check[
  ServiceExecute[conn, "ChatService", params],
  $Failed
];

(* A successful response is an Association with at least "Content". Anything
   else is treated as a failure. *)
If[response === $Failed || !AssociationQ[response] || !KeyExistsQ[response, "Content"],
  Export[outPath,
    <|"error"   -> "ServiceExecute[..., \"ChatService\", ...] did not return a Chat result",
      "raw"     -> ToString[response],
      "service" -> service,
      "model"   -> model|>, "JSON"];
  Exit[1]];

(* Token counts come from response["Usage"] when the service paclet
   surfaces them, as Quantity[N, "Tokens"] under "History" (input) and
   "Completion" (output). Strip the Quantity wrapper to a plain integer
   so the Python side can use it directly; omit the keys if usage is
   not available from this service. *)
usage = Lookup[response, "Usage", <||>];
tokenInt[q_Quantity] := QuantityMagnitude[q];
tokenInt[n_?IntegerQ] := n;
tokenInt[_] := Missing[];
inTokens = tokenInt[Lookup[usage, "History", Missing[]]];
outTokens = tokenInt[Lookup[usage, "Completion", Missing[]]];

baseOut = <|
  "content"      -> response["Content"],
  "service"      -> service,
  "model"        -> Lookup[response, "Model", model],
  "finishReason" -> ToString[Lookup[response, "FinishReason", ""]]
|>;
If[IntegerQ[inTokens],  AssociateTo[baseOut, "inputTokens" -> inTokens]];
If[IntegerQ[outTokens], AssociateTo[baseOut, "outputTokens" -> outTokens]];

Export[outPath, baseOut, "JSON"];
