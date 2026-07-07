(* Task: compute S(N) = sum_{i=1..N, j=1..N} GCD(i, j) for N = 300.
   Expected result: 336784.
   Goal: rewrite solve[] so it produces the same result, faster.
   Score = baseline / your_time. *)

(* Problem size — fixed, outside the EVOLVE-BLOCK so it cannot be mutated. *)
n = 300;

(* EVOLVE-BLOCK-START *)
(* Replace solve[] with a faster implementation that produces the same integer.
   - solve[] must return an Integer (no symbolic, no real).
   - n is fixed by the harness above; the harness verifies correctness
     against the known answer.
   - Lower runtime → higher score. Score climbs continuously, no plateau.
   - Any built-in Wolfram function and any algorithmic approach is fair game. *)

solve[] := Module[{s = 0, i, j, a, b, t},
  For[i = 1, i <= n, i++,
    For[j = 1, j <= n, j++,
      Range[n];                          (* recomputed each iteration; pure waste *)
      a = i; b = j;                      (* manual Euclidean GCD *)
      While[b > 0, t = Mod[a, b]; a = b; b = t];
      s = s + a
    ]
  ];
  s
];
(* EVOLVE-BLOCK-END *)


(* ============================================================ *)
(* I/O harness — DO NOT MODIFY (outside EVOLVE-BLOCK)            *)
(* ============================================================ *)

If[Length[$ScriptCommandLine] < 2,
  WriteString["stderr", "Usage: wolframscript -file initial.wl <out.json>\n"];
  Exit[1]];

outPath = $ScriptCommandLine[[2]];

(* Cap candidate runtime. WOLFRAM_GCD_MAX_SECONDS overrides the default,
   so slow machines can give candidates more headroom. *)
maxSecondsRaw = Environment["WOLFRAM_GCD_MAX_SECONDS"];
maxSeconds = If[StringQ[maxSecondsRaw] && StringLength[maxSecondsRaw] > 0,
  Quiet @ Check[ToExpression[maxSecondsRaw], 30.0],
  30.0
];
If[!NumericQ[maxSeconds] || maxSeconds <= 0, maxSeconds = 30.0];

(* One discarded warm-up evaluation. The first call to solve[] in a fresh
   kernel pays lazy-loading and first-evaluation cost for the builtins it
   touches that later calls do not. Running it once here, untimed, lets the
   RepeatedTiming below measure steady-state execution rather than reporting
   a single cold run. Bounded by maxSeconds so a runaway candidate cannot
   hang the warm-up. *)
TimeConstrained[solve[], maxSeconds];

(* RepeatedTiming auto-repeats solve[] and returns the average per-call
   time, which is far more stable than a single AbsoluteTiming sample. *)
{wallTime, result} = TimeConstrained[
  RepeatedTiming[solve[]],
  maxSeconds,
  {maxSeconds, "TIMEOUT"}
];

Export[outPath, <|
  "result"  -> ToString[result],
  "time_ms" -> 1000.0 * wallTime,
  "n"       -> n
|>, "JSON"];
