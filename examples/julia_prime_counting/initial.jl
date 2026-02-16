# EVOLVE-BLOCK-START
function is_prime(n::Int)::Bool
    if n < 2
        return false
    end
    if n == 2
        return true
    end
    if n % 2 == 0
        return false
    end

    d = 3
    while d * d <= n
        if n % d == 0
            return false
        end
        d += 2
    end
    return true
end


function count_primes_upto(n::Int)::Int
    if n < 2
        return 0
    end
    count = 1  # prime number 2
    k = 3
    while k <= n
        if is_prime(k)
            count += 1
        end
        k += 2
    end
    return count
end


function solve_prime_counts(queries::Vector{Int})::Vector{Int}
    return [count_primes_upto(q) for q in queries]
end
# EVOLVE-BLOCK-END


function _read_queries(input_path::String)::Vector{Int}
    queries = Int[]
    for line in eachline(input_path)
        stripped = strip(line)
        if !isempty(stripped)
            push!(queries, parse(Int, stripped))
        end
    end
    return queries
end


function _write_answers(output_path::String, answers::Vector{Int})
    open(output_path, "w") do io
        for (idx, value) in enumerate(answers)
            if idx > 1
                write(io, "\n")
            end
            write(io, string(value))
        end
    end
end


function _main()
    if length(ARGS) != 2
        println(stderr, "Usage: julia initial.jl <input_path> <output_path>")
        exit(1)
    end

    input_path = ARGS[1]
    output_path = ARGS[2]
    queries = _read_queries(input_path)
    answers = solve_prime_counts(queries)

    if length(answers) != length(queries)
        println(
            stderr,
            "Length mismatch: got $(length(answers)) answers for $(length(queries)) queries",
        )
        exit(1)
    end

    _write_answers(output_path, answers)
end


if abspath(PROGRAM_FILE) == @__FILE__
    _main()
end
