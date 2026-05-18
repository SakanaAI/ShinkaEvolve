(* Task: rewrite each of t1..t5 below into a smaller (lower LeafCount)
   Wolfram expression that returns the same value for every input.
   Score = seed_total_LeafCount / your_total_LeafCount (across the whole
   EVOLVE-BLOCK below). Any output mismatch, timeout, or use of
   `Simplify`/`FullSimplify` in your source scores -1.0. *)

(* EVOLVE-BLOCK-START *)

(* T1 — single expression: this polynomial factors. *)
t1[x_] := x^3 - 3 x^2 + 3 x - 1;

(* T2 — trig: an expanded identity. *)
t2[x_] := Sin[x]^4 + 2 Sin[x]^2 Cos[x]^2 + Cos[x]^4;

(* T3 — imperative summation with a known closed form. *)
t3[n_Integer] := Module[{s, i},
  s = 0;
  For[i = 1, i <= n, i = i + 1,
    s = s + i (i + 1)
  ];
  s
];

(* T4 — top-level Sequence: two helpers and a wrapper that composes them. *)
sq[x_]     := x * x;
double[x_] := x + x;
t4[x_]     := double[sq[x]] / 2;

(* T5 — verbose mean of a list. *)
t5[xs_List] := Module[{n, s, i},
  n = Length[xs];
  s = 0;
  For[i = 1, i <= n, i++, s = s + xs[[i]]];
  s / n
];

(* EVOLVE-BLOCK-END *)
