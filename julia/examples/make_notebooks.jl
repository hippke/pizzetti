# Generate the executed Jupyter notebooks in notebooks/ from the Literate-style scripts
# J0_*.jl … J6_*.jl of this folder.
#
#     julia --project=. make_notebooks.jl            # all scripts
#     julia --project=. make_notebooks.jl J1 J6      # a subset
#
# The scripts follow the Literate.jl conventions (`# ` lines are Markdown, `#-` starts a new
# cell, lines ending in `#src` are dropped, `#nb` keeps a line for the notebook only, `#jl`
# for the script only), so `Literate.notebook(file; execute = true)` produces the same
# notebooks. This file is a dependency-free stand-in: it uses only the standard library and
# JSON, executes every code cell in a fresh module, and stores the printed output, the
# returned value (figures as PNG, Markdown as Markdown) and any error in the notebook.

using JSON, Base64, Markdown

const HERE = @__DIR__
const OUT = joinpath(HERE, "notebooks")
const FILE = Ref("")
const KERNEL = Dict("display_name" => "Julia $(VERSION.major).$(VERSION.minor)",
                    "language" => "julia", "name" => "julia-$(VERSION.major).$(VERSION.minor)")

# ---------------------------------------------------------------- parsing
struct Cell
    kind::Symbol          # :markdown or :code
    source::String
end

function parse_literate(path)
    cells = Cell[]
    buf = String[]
    kind = :none
    flush!() = (if !isempty(buf) && any(!isempty ∘ strip, buf)
                    s = join(buf, "\n")
                    s = kind == :code ? strip(s, '\n') : strip(s)
                    push!(cells, Cell(kind, s))
                end; empty!(buf))
    for raw in readlines(path)
        line = rstrip(raw)
        endswith(line, "#src") && continue
        if endswith(line, "#jl")
            continue
        elseif endswith(line, "#nb")
            line = rstrip(line[1:end-3])
        end
        if line == "#-"
            flush!()
            continue
        end
        m = match(r"^#( (.*))?$", line)
        if m !== nothing && !startswith(line, "#=")
            kind == :code && flush!()
            kind = :markdown
            push!(buf, m.captures[2] === nothing ? "" : m.captures[2])
        else
            kind == :markdown && flush!()
            kind = :code
            push!(buf, raw)
        end
    end
    flush!()
    return cells
end

# ---------------------------------------------------------------- execution
function run_cell(mod, code)
    stdout_file = tempname()
    out = Dict{String,Any}[]
    value = nothing
    err = nothing
    open(stdout_file, "w") do io
        redirect_stdout(io) do
            redirect_stderr(io) do
                try
                    expr = Meta.parseall(code; filename = FILE[])
                    value = Core.eval(mod, expr)
                catch e
                    err = e
                end
            end
        end
    end
    txt = read(stdout_file, String)
    rm(stdout_file; force = true)
    if !isempty(txt)
        push!(out, Dict("output_type" => "stream", "name" => "stdout", "text" => txt))
    end
    suppressed = endswith(rstrip(code), ";")
    if err !== nothing
        msg = sprint(showerror, err)
        push!(out, Dict("output_type" => "error", "ename" => string(typeof(err)), "evalue" => msg,
                        "traceback" => [msg]))
    elseif value !== nothing && !suppressed
        push!(out, display_value(value))
    end
    return out, err
end

function display_value(x)
    data = Dict{String,Any}()
    if showable(MIME("image/png"), x)
        buf = IOBuffer()
        show(buf, MIME("image/png"), x)
        data["image/png"] = base64encode(take!(buf))
        data["text/plain"] = "Plot"
    elseif x isa Markdown.MD
        data["text/markdown"] = sprint(show, MIME("text/markdown"), x)
        data["text/plain"] = sprint(show, MIME("text/plain"), x)
    else
        data["text/plain"] = sprint(show, MIME("text/plain"), x; context = :limit => true)
    end
    return Dict("output_type" => "execute_result", "execution_count" => nothing, "metadata" => Dict(), "data" => data)
end

function build(name)
    src = joinpath(HERE, name)
    FILE[] = src
    cells = parse_literate(src)
    mod = Module(Symbol(replace(splitext(name)[1], r"\W" => "_")))
    Core.eval(mod, :(const NOTEBOOK = true))
    Core.eval(mod, :(eval(x) = Core.eval($mod, x)))
    nb = Dict{String,Any}[]
    nerr = 0
    count = 0
    t0 = time()
    for c in cells
        if c.kind == :markdown
            push!(nb, Dict("cell_type" => "markdown", "metadata" => Dict(), "source" => c.source))
        else
            count += 1
            out, err = run_cell(mod, c.source)
            err === nothing || (nerr += 1; @warn "error in cell $count of $name" err)
            push!(nb, Dict("cell_type" => "code", "metadata" => Dict(), "execution_count" => count,
                           "source" => c.source, "outputs" => out))
        end
    end
    doc = Dict("cells" => nb, "nbformat" => 4, "nbformat_minor" => 5,
               "metadata" => Dict("kernelspec" => KERNEL,
                                  "language_info" => Dict("name" => "julia", "version" => string(VERSION),
                                                          "file_extension" => ".jl", "mimetype" => "application/julia")))
    mkpath(OUT)
    dest = joinpath(OUT, splitext(name)[1] * ".ipynb")
    open(dest, "w") do io
        JSON.print(io, doc, 1)
    end
    @info "$(basename(dest)): $count code cells, $nerr errors, $(round(time() - t0, digits = 1)) s"
    return nerr
end

function main(args)
    ENV["GKSwstype"] = "100"
    files = sort(filter(f -> occursin(r"^J\d_.*\.jl$", f), readdir(HERE)))
    if !isempty(args)
        files = filter(f -> any(a -> startswith(f, a), args), files)
    end
    total = 0
    for f in files
        total += build(f)
    end
    total == 0 || error("$total cells raised errors")
end

main(ARGS)
