(* Evaluator-invoked harness for the wolfram_simplify example.

   Usage:
     wolframscript -file harness.wl <seed.wl> <candidate.wl> <out.json>

   For each of {seed, candidate} the harness:
     1. Reads the file's EVOLVE-BLOCK source text.
     2. Parses it with `ToExpression[..., InputForm, Hold]` and reports
        `leafCount = LeafCount[heldForm]`. This counts the LLM-written
        form, not the evaluated form — so wrapping in `FullSimplify`
        / `Evaluate[...]` inflates the count rather than shrinking it.
     3. Scans the held form for blocklisted heads `Simplify`,
        `FullSimplify` (the prompt forbids them; we enforce it).
     4. `Get`s the file to install the definitions.
     5. Runs t1..t5 on a fixed input batch with `TimeConstrained` per call.

   Output JSON:
     { "seed":      {"leafCount": N, "blocklisted": [...], "parseError": s|null,
                     "outputs": {"tN": [v1, ...]}, "timeouts": [...]},
       "candidate": { same shape } }

   Tolerance and the seed/candidate output comparison are handled in Python. *)

If[Length[$ScriptCommandLine] < 4,
  WriteString["stderr",
    "Usage: wolframscript -file harness.wl <seed.wl> <candidate.wl> <out.json>\n"];
  Exit[1]];

seedPath      = $ScriptCommandLine[[2]];
candidatePath = $ScriptCommandLine[[3]];
outPath       = $ScriptCommandLine[[4]];

(* Fixed numeric inputs per task. Deterministic, seeded once at design time,
   chosen so seed and target forms produce identical values within float
   tolerance and so no input hits an undefined boundary (T5 mean of empty
   list is intentionally excluded). *)
$inputs = <|
  "t1" -> {0.0, 1.0, -1.0, 2.5, -3.7, 10.0, 0.137, -7.21},
  "t2" -> {0.0, 1.0, 1.5707963267948966, -2.5, 0.5, 0.7853981633974483, 3.14159, -1.1},
  "t3" -> {0, 1, 2, 5, 10, 50, 100},
  "t4" -> {0.0, 1.0, 2.0, -3.5, 0.5, 7.0, -0.25, 11.0},
  "t5" -> {{1.0}, {1.0, 2.0}, {1.0, 2.0, 3.0, 4.0, 5.0},
           {-1.5, 2.5, 0.0, 3.5}, {10.0, -10.0, 5.0, -5.0, 0.0}}
|>;

$perCallSeconds = 0.05;
$blocklist      = {Simplify, FullSimplify};

extractEvolveBlock[path_] := Module[{src, m},
  src = Quiet @ Check[Import[path, "Text"], $Failed];
  If[src === $Failed, Return[Missing["importFailed"]]];
  m = StringCases[
    src,
    StartOfLine ~~ Shortest["(* EVOLVE-BLOCK-START *)" ~~ body__ ~~
      "(* EVOLVE-BLOCK-END *)"] :> body,
    1
  ];
  If[Length[m] != 1, Return[Missing["markersMissing"]]];
  First[m]
];

parseHeld[src_String] := Module[{held},
  held = Quiet @ Check[ToExpression[src, InputForm, Hold], $Failed];
  If[held === $Failed, Missing["parseError"], held]
];

findBlocklisted[held_Hold] :=
  DeleteDuplicates[
    SymbolName /@ Cases[held,
      sym_Symbol /; MemberQ[$blocklist, sym],
      Infinity,
      Heads -> True]];

findBlocklisted[_] := {};

(* Stable JSON-friendly atom serializer: integers and reals pass through;
   other heads (e.g. rationals, ComplexInfinity, $Failed) get stringified
   so Python can compare them as opaque tokens. *)
toJsonValue[n_Integer]               := n;
toJsonValue[r_Real]                  := r;
toJsonValue[xs_List]                 := toJsonValue /@ xs;
toJsonValue[Indeterminate]           := "Indeterminate";
toJsonValue[ComplexInfinity]         := "ComplexInfinity";
toJsonValue[other_]                  := ToString[other, InputForm];

runOne[fnName_, input_] := Module[{result, fn},
  fn   = Symbol[fnName];
  result = TimeConstrained[Quiet @ Check[fn[input], "$EvalFailed"],
    $perCallSeconds, "$Timeout"];
  toJsonValue[result]
];

evaluateOne[fnName_] := AssociationThread[
  {"inputs", "outputs"},
  {toJsonValue /@ $inputs[fnName], runOne[fnName, #] & /@ $inputs[fnName]}
];

analyze[path_] := Module[{src, held, leafCount, blocklisted, outputs},
  src = extractEvolveBlock[path];
  If[MissingQ[src],
    Return[<|"leafCount" -> Missing["NA"], "blocklisted" -> {},
             "parseError" -> First[src], "outputs" -> <||>|>]];

  held = parseHeld[src];
  If[MissingQ[held],
    Return[<|"leafCount" -> Missing["NA"], "blocklisted" -> {},
             "parseError" -> "parseError", "outputs" -> <||>|>]];

  leafCount   = LeafCount[held];
  blocklisted = findBlocklisted[held];

  If[blocklisted =!= {},
    Return[<|"leafCount" -> leafCount, "blocklisted" -> blocklisted,
             "parseError" -> Null, "outputs" -> <||>|>]];

  (* Install definitions. Get evaluates; any runtime error from a definition
     is captured as the function's evaluation results showing $EvalFailed. *)
  Quiet @ Check[Get[path], Null];

  outputs = AssociationMap[evaluateOne, {"t1", "t2", "t3", "t4", "t5"}];

  <|"leafCount" -> leafCount,
    "blocklisted" -> blocklisted,
    "parseError" -> Null,
    "outputs" -> outputs|>
];

result = <|
  "seed" -> analyze[seedPath],
  "candidate" -> analyze[candidatePath]
|>;

Export[outPath, result, "JSON"];
